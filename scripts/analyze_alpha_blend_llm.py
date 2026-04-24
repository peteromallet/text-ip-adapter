#!/usr/bin/env python3
"""LLM-judge analysis of α-blend probe outputs.

Reads `alpha_blend_results.jsonl` from probe_capabilities.py output and asks Claude
which generation each sample is more stylistically similar to (own_ref vs swap_ref).

This verifies whether α-blend interpolation is along the STYLE axis (the C3 claim)
or just along some arbitrary text-similarity axis.

Expected pattern if C3 holds:
- α=0.0 (pure swap_z) → judge picks swap_ref ~most often
- α=1.0 (pure own_z)  → judge picks own_ref ~most often
- α=0.5 → roughly 50/50 (true intermediate style)
- Monotonic transition between

If degenerate (e.g. flat at 50% across all α, or sharp jump at one α boundary), C3 is
not strongly supported despite the textual-Jaccard monotonicity.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT = """You are judging which of two REFERENCE passages a GENERATION is more stylistically similar to. Focus on style: voice, register, rhythm, vocabulary, syntax. Ignore content/topic — they may be similar or different and that doesn't matter.

REFERENCE A:
{ref_a}

REFERENCE B:
{ref_b}

GENERATION:
{gen}

Which reference's STYLE does the generation more closely match? Respond with EXACTLY one of: A, B, or TIE. No explanation."""


def judge_one(client, gen_text: str, ref_a: str, ref_b: str, model: str = "claude-haiku-4-5") -> str:
    msg = PROMPT.format(ref_a=ref_a[:1200], ref_b=ref_b[:1200], gen=gen_text[:1500])
    resp = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": msg}],
    )
    raw = resp.content[0].text.strip().upper().split()[0] if resp.content else "TIE"
    raw = raw.rstrip(".:,;\"'`")
    if raw not in ("A", "B", "TIE"):
        raw = "TIE"
    return raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-results", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        return 2
    client = anthropic.Anthropic(api_key=api_key)

    # Load α-blend probe outputs (one row per probe with all α generations)
    rows = [json.loads(l) for l in open(args.alpha_results) if l.strip()]
    probes_by_id = {p["probe_id"]: p for p in [json.loads(l) for l in open(args.probes) if l.strip()]}
    print(f"[llm-alpha] {len(rows)} probes, judging across alphas")

    # For each (probe, alpha), get a judgment of "is gen more like own_ref or swap_ref"
    # Randomize A/B assignment to defeat position bias.
    judgments = []  # one record per (probe_id, alpha)

    def task(row, alpha_str, gen_text):
        probe = probes_by_id.get(row["probe_id"])
        if not probe:
            return None
        own = probe["reference_text"]
        swap = probe["swap_reference_text"]
        rng = random.Random(hash((row["probe_id"], alpha_str)) & 0xffff)
        own_is_a = rng.random() < 0.5
        ref_a = own if own_is_a else swap
        ref_b = swap if own_is_a else own
        try:
            verdict = judge_one(client, gen_text, ref_a, ref_b, args.model)
        except Exception as e:
            return {"probe_id": row["probe_id"], "alpha": alpha_str, "error": str(e)[:200]}
        if verdict == "A":
            picked = "own" if own_is_a else "swap"
        elif verdict == "B":
            picked = "swap" if own_is_a else "own"
        else:
            picked = "tie"
        return {"probe_id": row["probe_id"], "alpha": alpha_str, "picked": picked, "raw_verdict": verdict, "own_was_a": own_is_a}

    # Submit all (probe, alpha) tasks
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in rows:
            for alpha_str, gen_text in row.get("generations", {}).items():
                futures.append(ex.submit(task, row, alpha_str, gen_text))
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                judgments.append(r)

    # Aggregate per-alpha distribution
    from collections import Counter
    by_alpha = {}
    for j in judgments:
        if "picked" not in j:
            continue
        a = j["alpha"]
        by_alpha.setdefault(a, Counter())[j["picked"]] += 1

    summary = {
        "n_probes": len(rows),
        "n_judgments": len(judgments),
        "model": args.model,
        "per_alpha": {},
    }
    for alpha_str in sorted(by_alpha.keys(), key=lambda x: float(x)):
        c = by_alpha[alpha_str]
        total = sum(c.values())
        summary["per_alpha"][alpha_str] = {
            "own": c["own"], "swap": c["swap"], "tie": c["tie"], "n": total,
            "frac_own": c["own"] / max(total, 1),
            "frac_swap": c["swap"] / max(total, 1),
        }

    # Diagnostic: is the trend monotonic (frac_own ↑ as α moves from 0 to 1)?
    alphas_sorted = sorted(summary["per_alpha"].keys(), key=lambda x: float(x))
    fracs_own = [summary["per_alpha"][a]["frac_own"] for a in alphas_sorted]
    monotonic = all(fracs_own[i] <= fracs_own[i+1] + 0.05 for i in range(len(fracs_own)-1))  # allow 5pp slack
    summary["monotonic_own_with_alpha"] = monotonic
    summary["frac_own_at_alpha_0"] = fracs_own[0] if fracs_own else None
    summary["frac_own_at_alpha_1"] = fracs_own[-1] if fracs_own else None
    summary["c3_signal_strength"] = (fracs_own[-1] - fracs_own[0]) if len(fracs_own) >= 2 else 0.0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "judgments": judgments}, f, indent=2)

    print("\n=== LLM-JUDGE α-BLEND ANALYSIS ===")
    print(f"n_probes: {len(rows)}, total judgments: {len(judgments)}, model: {args.model}")
    print(f"\nFor each α, fraction of generations the judge labeled as 'more like OWN reference':")
    for a in alphas_sorted:
        s = summary["per_alpha"][a]
        bar = "█" * int(s["frac_own"] * 20)
        print(f"  α={a}  own={s['own']:>2}  swap={s['swap']:>2}  tie={s['tie']:>2}   frac_own={s['frac_own']:.2f}  {bar}")
    print(f"\nmonotonic_own_with_alpha: {monotonic}  (frac_own should grow as α moves 0→1)")
    print(f"C3 signal strength: {summary['c3_signal_strength']:.2f}  (0.0 = no axis, 1.0 = perfect axis)")
    if summary['c3_signal_strength'] > 0.4 and monotonic:
        print("VERDICT: STRONG C3 EVIDENCE — α-blend interpolates along the style axis")
    elif summary['c3_signal_strength'] > 0.15:
        print("VERDICT: WEAK C3 EVIDENCE — there's a style-axis signal but small")
    else:
        print("VERDICT: NO C3 EVIDENCE — α-blend produces variation that isn't style-shaped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
