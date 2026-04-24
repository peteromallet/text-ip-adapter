#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


# ----------------------------- IO helpers -----------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ----------------------------- text features -----------------------------

ARCHAIC = {
    "thee", "thou", "thy", "thine", "ye", "o'er", "e'er", "ne'er", "'tis", "'twas",
    "doth", "hath", "hast", "art", "wast", "wert", "ere", "nay", "forsooth", "whence", "whither", "hither",
}

STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","on","at","for","with","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","can","could","should","may","might","that","this","these","those",
    "i","you","he","she","it","we","they","my","your","his","her","its","our","their","how","what","when","where","why","who","which",
    "if","as","not","no","so","just","like","than","then","there","here",
}


def tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", s.lower())


def ngrams(s: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = tokens(s)
    if len(toks) < n:
        return set()
    return set(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def surface_features(text: str) -> dict[str, float]:
    toks = tokens(text)
    n = max(len(toks), 1)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n_lines = max(len(lines), 1)
    return {
        "avg_line_len_chars": mean(len(ln) for ln in lines) if lines else 0.0,
        "avg_line_len_tokens": n / n_lines,
        "em_dash_rate": text.count("—") / n,
        "em_dash_ascii_rate": text.count("--") / n,
        "exclaim_rate": text.count("!") / n,
        "question_rate": text.count("?") / n,
        "archaic_rate": sum(1 for t in toks if t in ARCHAIC) / n,
        "cap_i_rate": sum(1 for t in toks if t == "i") / n,  # Whitman / Dickinson / free verse tell
        "ttr": len(set(toks)) / n,
        "comma_per_token": text.count(",") / n,
        "semicolon_per_token": text.count(";") / n,
    }


def cosine_sim_dict(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    va = [a.get(k, 0.0) for k in keys]
    vb = [b.get(k, 0.0) for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ----------------------------- tests -----------------------------

def test1_discrimination(samples: list[dict]) -> dict:
    # Cosine of 3-gram Jaccard between adapter and adapter_swap across steps.
    by_step_probe: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in samples:
        if isinstance(r["step"], int):
            by_step_probe[(r["step"], r["probe_id"])][r["variant"]] = r["text"]
    per_step: dict[int, list[float]] = defaultdict(list)
    for (step, _pid), variants in by_step_probe.items():
        if "adapter" in variants and "adapter_swap" in variants:
            j = jaccard(ngrams(variants["adapter"], 3), ngrams(variants["adapter_swap"], 3))
            # High Jaccard = outputs very similar = BAD (we want differentiation).
            per_step[step].append(j)
    summary = {
        int(step): {"mean_jaccard": mean(v), "n": len(v), "max": max(v), "min": min(v)}
        for step, v in sorted(per_step.items()) if v
    }
    # Final verdict uses the LAST step.
    if summary:
        last = max(summary)
        final = summary[last]["mean_jaccard"]
        # Thresholds: <0.25 pass, >0.5 fail, in between weak.
        if final < 0.25:
            verdict = "PASS"
        elif final < 0.5:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        return {"per_step": summary, "final_step": last, "final_mean_jaccard": final, "verdict": verdict,
                "note": "Jaccard of 3-grams between adapter and adapter_swap. Lower = more discrimination."}
    return {"per_step": {}, "verdict": "NO_DATA"}


def test3_style_carryover(samples: list[dict], probes: list[dict], author_corpus: dict[str, list[str]]) -> dict:
    # For each probe at the last step, compare adapter-output surface features to the OWN reference text
    # vs the SWAP reference text. Author-split means probe authors aren't in train corpus, so we use
    # the reference text itself as the style profile (more direct anyway: the adapter gets that exact ref).
    last_step = max((r["step"] for r in samples if isinstance(r["step"], int)), default=None)
    if last_step is None:
        return {"verdict": "NO_DATA"}
    probe_by_id = {p["probe_id"]: p for p in probes}
    results = []
    for r in samples:
        if r["step"] != last_step or r["variant"] != "adapter":
            continue
        p = probe_by_id.get(r["probe_id"])
        if not p:
            continue
        own_ref_feats = surface_features(p["reference_text"])
        swap_ref_feats = surface_features(p["swap_reference_text"])
        gen_feats = surface_features(r["text"])
        sim_own = cosine_sim_dict(gen_feats, own_ref_feats)
        sim_swap = cosine_sim_dict(gen_feats, swap_ref_feats)
        results.append({
            "probe_id": r["probe_id"], "own_author": p["author"], "swap_author": p["swap_reference_author"],
            "sim_own": sim_own, "sim_swap": sim_swap, "advantage_own": sim_own - sim_swap,
        })
    if not results:
        return {"verdict": "NO_DATA"}
    mean_adv = mean(r["advantage_own"] for r in results)
    wins = sum(1 for r in results if r["advantage_own"] > 0)
    verdict = "PASS" if mean_adv > 0.05 else ("WEAK" if mean_adv > 0 else "FAIL")
    return {
        "per_probe": results, "mean_advantage": mean_adv, "own_wins": wins, "total": len(results),
        "verdict": verdict,
        "note": "advantage_own = cos(gen_feats, own_ref_feats) - cos(gen_feats, swap_ref_feats). >0.05 = PASS.",
    }


def test4_memorization(samples: list[dict], probes: list[dict], train_pairs: list[dict]) -> dict:
    # a) Target memorization: adapter outputs that contain 10-gram substrings found in any training target.
    # b) Reference leak: adapter outputs containing 5-gram substrings found in their reference.
    target_10grams: set[tuple[str, ...]] = set()
    for p in train_pairs:
        t = p.get("target_text", "")
        target_10grams |= ngrams(t, 10)
    probe_refs = {p["probe_id"]: p["reference_text"] for p in probes}
    last_step = max((r["step"] for r in samples if isinstance(r["step"], int)), default=None)
    adapter_samples = [r for r in samples if r["variant"] == "adapter" and r["step"] == last_step]
    mem_hits = 0
    leak_hits = 0
    per_probe = []
    for r in adapter_samples:
        gen_10 = ngrams(r["text"], 10)
        gen_5 = ngrams(r["text"], 5)
        ref = probe_refs.get(r["probe_id"], "")
        ref_5 = ngrams(ref, 5)
        mem = len(gen_10 & target_10grams)
        leak = len(gen_5 & ref_5)
        per_probe.append({"probe_id": r["probe_id"], "target_10gram_hits": mem, "ref_5gram_hits": leak})
        if mem > 0:
            mem_hits += 1
        if leak > 0:
            leak_hits += 1
    n = max(len(adapter_samples), 1)
    mem_rate = mem_hits / n
    leak_rate = leak_hits / n
    mem_verdict = "PASS" if mem_rate <= 0.05 else ("WEAK" if mem_rate <= 0.20 else "FAIL")
    leak_verdict = "PASS" if leak_rate <= 0.05 else ("WEAK" if leak_rate <= 0.20 else "FAIL")
    return {
        "per_probe": per_probe, "target_memorization_rate": mem_rate, "ref_leak_rate": leak_rate,
        "memorization_verdict": mem_verdict, "leak_verdict": leak_verdict,
        "note": "memorization: fraction of adapter outputs with any 10-gram in training targets. leak: 5-gram overlap with probe's own reference.",
    }


def test5_loss_curve(train_log: list[dict]) -> dict:
    losses = [r for r in train_log if "loss" in r and "step" in r]
    if len(losses) < 5:
        return {"verdict": "NO_DATA"}
    losses.sort(key=lambda r: r["step"])
    first_q = [r["loss"] for r in losses[: max(1, len(losses) // 4)]]
    last_q = [r["loss"] for r in losses[-max(1, len(losses) // 4):]]
    first_mean = mean(first_q)
    last_mean = mean(last_q)
    improvement = first_mean - last_mean
    improvement_pct = improvement / max(first_mean, 1e-9)
    # Rolling min in last half — did loss still drop in the last half?
    half = len(losses) // 2
    last_half_min = min(r["loss"] for r in losses[half:])
    first_half_min = min(r["loss"] for r in losses[:half])
    still_learning = last_half_min < first_half_min - 0.05
    if improvement_pct > 0.25 and still_learning:
        verdict = "HEALTHY"
    elif improvement_pct > 0.10:
        verdict = "SLOW"
    else:
        verdict = "PLATEAU"
    return {
        "n_steps": len(losses),
        "first_quartile_loss_mean": first_mean,
        "last_quartile_loss_mean": last_mean,
        "improvement_pct": improvement_pct,
        "still_learning_last_half": still_learning,
        "verdict": verdict,
        "note": "HEALTHY: >25% loss drop + still improving. PLATEAU: <10% drop.",
    }


def test2_llm_judge(samples: list[dict], probes: list[dict], api_key: str, n_max: int = 8) -> dict:
    try:
        import anthropic  # type: ignore
    except Exception as exc:
        return {"verdict": "SKIPPED", "reason": f"anthropic SDK unavailable: {exc}"}
    client = anthropic.Anthropic(api_key=api_key)
    last_step = max((r["step"] for r in samples if isinstance(r["step"], int)), default=None)
    if last_step is None:
        return {"verdict": "NO_DATA"}
    probe_by_id = {p["probe_id"]: p for p in probes}
    adapter_by_probe = {r["probe_id"]: r["text"] for r in samples if r["step"] == last_step and r["variant"] == "adapter"}
    baseline_by_probe = {r["probe_id"]: r["text"] for r in samples if r["variant"] == "prompted_baseline"}
    shared = sorted(set(adapter_by_probe) & set(baseline_by_probe))[: n_max]
    if not shared:
        return {"verdict": "NO_DATA", "reason": "no shared probe_ids between adapter and prompted_baseline"}

    prompt_template = (
        "You are a careful critic comparing two short generated texts against a REFERENCE passage. "
        "Judge which generation better matches the REFERENCE in STYLE (voice, rhythm, vocabulary, register, syntax), "
        "independent of content. Ignore differences in what they are about; compare HOW they write.\n\n"
        "REFERENCE (style to match):\n{reference}\n\n"
        "GENERATION A:\n{gen_a}\n\n"
        "GENERATION B:\n{gen_b}\n\n"
        "Respond with EXACTLY one of: A, B, or TIE. No explanation."
    )
    verdicts: list[dict] = []
    for pid in shared:
        probe = probe_by_id[pid]
        # Randomize A/B assignment to avoid position bias.
        import random
        random.seed(hash(pid) & 0xffff)
        adapter_is_a = random.random() < 0.5
        gen_a = adapter_by_probe[pid] if adapter_is_a else baseline_by_probe[pid]
        gen_b = baseline_by_probe[pid] if adapter_is_a else adapter_by_probe[pid]
        msg = prompt_template.format(reference=probe["reference_text"][:1200], gen_a=gen_a[:1200], gen_b=gen_b[:1200])
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=8,
                messages=[{"role": "user", "content": msg}],
            )
            raw = resp.content[0].text.strip().upper().split()[0] if resp.content else "TIE"
        except Exception as exc:
            verdicts.append({"probe_id": pid, "winner": "ERROR", "error": str(exc)[:200]})
            continue
        raw = raw.rstrip(".:,;")
        if raw == "A":
            winner = "adapter" if adapter_is_a else "baseline"
        elif raw == "B":
            winner = "baseline" if adapter_is_a else "adapter"
        else:
            winner = "tie"
        verdicts.append({"probe_id": pid, "adapter_was": "A" if adapter_is_a else "B", "raw": raw, "winner": winner})
    adapter_wins = sum(1 for v in verdicts if v.get("winner") == "adapter")
    baseline_wins = sum(1 for v in verdicts if v.get("winner") == "baseline")
    ties = sum(1 for v in verdicts if v.get("winner") == "tie")
    total = adapter_wins + baseline_wins + ties
    win_rate = adapter_wins / max(total, 1)
    if win_rate >= 0.60:
        verdict = "PASS"
    elif win_rate >= 0.45:
        verdict = "TIE"
    else:
        verdict = "FAIL"
    return {
        "per_probe": verdicts,
        "adapter_wins": adapter_wins, "baseline_wins": baseline_wins, "ties": ties,
        "adapter_win_rate": win_rate,
        "verdict": verdict,
        "note": "PASS >=0.60 win rate over prompted_baseline. TIE 0.45-0.60. FAIL <0.45.",
    }


def test3_llm_judge_style_match(samples: list[dict], probes: list[dict], api_key: str, n_max: int = 20) -> dict:
    try:
        import anthropic  # type: ignore
    except Exception as exc:
        return {"verdict": "SKIPPED", "reason": f"anthropic SDK unavailable: {exc}"}
    client = anthropic.Anthropic(api_key=api_key)
    last_step = max((r["step"] for r in samples if isinstance(r["step"], int)), default=None)
    if last_step is None:
        return {"verdict": "NO_DATA"}
    probe_by_id = {p["probe_id"]: p for p in probes}
    adapter_by_probe = {r["probe_id"]: r["text"] for r in samples if r["step"] == last_step and r["variant"] == "adapter"}
    swap_by_probe = {r["probe_id"]: r["text"] for r in samples if r["step"] == last_step and r["variant"] == "adapter_swap"}
    shared = sorted(set(adapter_by_probe) & set(swap_by_probe) & set(probe_by_id))[: n_max]
    if not shared:
        return {"verdict": "NO_DATA", "reason": "no shared probe_ids between adapter and adapter_swap"}
    prompt_template = (
        "You are judging which of two generated texts better matches the STYLE of a REFERENCE passage. "
        "Focus purely on style — voice, register, rhythm, vocabulary, syntactic patterns. Ignore content differences.\n\n"
        "REFERENCE (style to match):\n{reference_text}\n\n"
        "GENERATION A:\n{gen_a}\n\n"
        "GENERATION B:\n{gen_b}\n\n"
        "Which generation better matches the reference's style? Respond with EXACTLY one of: A, B, or TIE. No explanation."
    )
    import random
    verdicts: list[dict] = []
    for pid in shared:
        probe = probe_by_id[pid]
        random.seed(hash(pid) & 0xffff)
        adapter_is_a = random.random() < 0.5
        gen_a = adapter_by_probe[pid] if adapter_is_a else swap_by_probe[pid]
        gen_b = swap_by_probe[pid] if adapter_is_a else adapter_by_probe[pid]
        msg = prompt_template.format(reference_text=probe["reference_text"][:1200], gen_a=gen_a[:1200], gen_b=gen_b[:1200])
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8,
                messages=[{"role": "user", "content": msg}],
            )
            raw = resp.content[0].text.strip().upper().split()[0] if resp.content else "TIE"
        except Exception as exc:
            verdicts.append({"probe_id": pid, "winner": "ERROR", "error": str(exc)[:200]})
            continue
        raw = raw.rstrip(".:,;")
        if raw == "A":
            winner = "adapter" if adapter_is_a else "swap"
        elif raw == "B":
            winner = "swap" if adapter_is_a else "adapter"
        else:
            winner = "tie"
        verdicts.append({"probe_id": pid, "adapter_was": "A" if adapter_is_a else "B", "raw": raw, "winner": winner})
    adapter_wins = sum(1 for v in verdicts if v.get("winner") == "adapter")
    swap_wins = sum(1 for v in verdicts if v.get("winner") == "swap")
    ties = sum(1 for v in verdicts if v.get("winner") == "tie")
    total = adapter_wins + swap_wins + ties
    win_rate = adapter_wins / max(total, 1)
    if win_rate >= 0.60:
        verdict = "PASS"
    elif win_rate >= 0.50:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    return {
        "per_probe": verdicts,
        "adapter_wins": adapter_wins, "swap_wins": swap_wins, "ties": ties,
        "adapter_win_rate": win_rate,
        "verdict": verdict,
        "note": "LLM-judge style match: adapter vs adapter_swap output compared to probe reference. PASS >=0.60, WEAK 0.50-0.60, FAIL <0.50.",
    }


# ----------------------------- main -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="checkpoints/stage1_gemma")
    parser.add_argument("--probe-path", default="data/pairs/probes.jsonl")
    parser.add_argument("--train-pairs", default="data/pairs/train.jsonl")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--n-judge", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    samples = load_jsonl(run_dir / "samples.jsonl")
    train_log = load_jsonl(run_dir / "train_log.jsonl")
    probes = load_jsonl(Path(args.probe_path))
    train_pairs = load_jsonl(Path(args.train_pairs))

    if not samples:
        print(f"ERROR: no samples at {run_dir}/samples.jsonl", file=sys.stderr)
        return 2
    print(f"[eval] samples={len(samples)} train_log={len(train_log)} probes={len(probes)} train_pairs={len(train_pairs)}")

    # Build author corpus from train_pairs for style profiles.
    author_corpus: dict[str, list[str]] = defaultdict(list)
    for p in train_pairs:
        author_corpus[p.get("author", "unknown")].append(p.get("target_text", ""))
        author_corpus[p.get("author", "unknown")].append(p.get("ref_text", ""))

    report: dict = {}
    print("\n--- Test 1: Reference discrimination (adapter vs adapter_swap) ---")
    report["test1_discrimination"] = test1_discrimination(samples)
    for step, v in list(report["test1_discrimination"].get("per_step", {}).items())[-5:]:
        print(f"  step {step:>5}: mean_jaccard={v['mean_jaccard']:.3f} (n={v['n']})")
    print(f"  VERDICT: {report['test1_discrimination'].get('verdict')}")

    print("\n--- Test 3: Style carryover (surface features match own author > swap author?) ---")
    report["test3_style_carryover"] = test3_style_carryover(samples, probes, author_corpus)
    t3 = report["test3_style_carryover"]
    if t3.get("verdict") != "NO_DATA":
        print(f"  own wins: {t3['own_wins']}/{t3['total']}  mean_advantage={t3['mean_advantage']:.3f}")
    print(f"  VERDICT: {t3.get('verdict')}")

    print("\n--- Test 3b: LLM-judge style match (adapter vs adapter_swap against reference) ---")
    if args.skip_llm_judge:
        report["test3b_llm_style_match"] = {"verdict": "SKIPPED", "reason": "--skip-llm-judge"}
        print("  SKIPPED")
    else:
        api_key_t3b = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key_t3b:
            report["test3b_llm_style_match"] = {"verdict": "SKIPPED", "reason": "ANTHROPIC_API_KEY not set"}
            print("  SKIPPED (no ANTHROPIC_API_KEY)")
        else:
            t3b = test3_llm_judge_style_match(samples, probes, api_key_t3b, n_max=args.n_judge)
            report["test3b_llm_style_match"] = t3b
            if t3b.get("verdict") not in ("NO_DATA", "SKIPPED"):
                print(f"  adapter_wins: {t3b['adapter_wins']}  swap_wins: {t3b['swap_wins']}  ties: {t3b['ties']}  win_rate: {t3b['adapter_win_rate']:.2f}")
            print(f"  VERDICT: {t3b.get('verdict')}")

    print("\n--- Test 4: Memorization / leak ---")
    report["test4_memorization"] = test4_memorization(samples, probes, train_pairs)
    t4 = report["test4_memorization"]
    if t4.get("verdict") != "NO_DATA":
        print(f"  target memorization: {t4['target_memorization_rate']*100:.1f}%  ref leak: {t4['ref_leak_rate']*100:.1f}%")
    print(f"  VERDICT mem: {t4.get('memorization_verdict')}  leak: {t4.get('leak_verdict')}")

    print("\n--- Test 5: Loss curve shape ---")
    report["test5_loss_curve"] = test5_loss_curve(train_log)
    t5 = report["test5_loss_curve"]
    if t5.get("verdict") != "NO_DATA":
        print(f"  first-Q loss: {t5['first_quartile_loss_mean']:.3f}  last-Q loss: {t5['last_quartile_loss_mean']:.3f}  improvement: {t5['improvement_pct']*100:.1f}%")
    print(f"  VERDICT: {t5.get('verdict')}")

    print("\n--- Test 2: LLM-judge (adapter vs prompted_baseline) ---")
    if args.skip_llm_judge:
        report["test2_llm_judge"] = {"verdict": "SKIPPED", "reason": "--skip-llm-judge"}
        print("  SKIPPED")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            report["test2_llm_judge"] = {"verdict": "SKIPPED", "reason": "ANTHROPIC_API_KEY not set"}
            print("  SKIPPED (no ANTHROPIC_API_KEY)")
        else:
            t2 = test2_llm_judge(samples, probes, api_key, n_max=args.n_judge)
            report["test2_llm_judge"] = t2
            if t2.get("verdict") not in ("NO_DATA", "SKIPPED"):
                print(f"  adapter_wins: {t2['adapter_wins']}  baseline_wins: {t2['baseline_wins']}  ties: {t2['ties']}  win_rate: {t2['adapter_win_rate']:.2f}")
            print(f"  VERDICT: {t2.get('verdict')}")

    # Decision matrix summary.
    print("\n=== SUMMARY ===")
    t1v = report["test1_discrimination"].get("verdict")
    t2v = report.get("test2_llm_judge", {}).get("verdict")
    t3v = report["test3_style_carryover"].get("verdict")
    t4mv = report["test4_memorization"].get("memorization_verdict")
    t4lv = report["test4_memorization"].get("leak_verdict")
    t5v = report["test5_loss_curve"].get("verdict")
    print(f"  T1 (discrimination):        {t1v}")
    print(f"  T2 (vs prompted baseline):  {t2v}")
    print(f"  T3 (style carryover):       {t3v}")
    print(f"  T3b (LLM-judge style match):    {report['test3b_llm_style_match'].get('verdict')}")
    print(f"  T4 (memorization):          {t4mv}")
    print(f"  T4 (ref leak):              {t4lv}")
    print(f"  T5 (loss curve):            {t5v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[eval] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
