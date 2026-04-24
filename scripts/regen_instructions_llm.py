#!/usr/bin/env python3
"""Regenerate (target-only) instructions for all pairs via Claude.

Reads data/pairs/{train,val,test}.jsonl; rewrites instructions from the target_text only
(never sees the reference). Writes to data/pairs/{train,val,test}.llm.jsonl.

Cost estimate at ~1000 pairs, 100-200 input tokens + ~30 output tokens per call,
claude-haiku-4-5 at $0.25/1M in + $1.25/1M out: ~$0.50-1.00 total.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYSTEM_PROMPT = """You write short, content-only instructions. Given a TARGET TEXT, you produce ONE imperative instruction that describes WHAT CONTENT the text conveys (its topic/subject/situation) without describing HOW it is written (no references to style, author, tone, voice, or rhythm).

Rules:
- Output ONLY the instruction, one short sentence (10-25 words).
- Do NOT mention style, voice, register, rhythm, or form.
- Do NOT reuse 5-grams from the target.
- Match the register implicit in the text (a scene instruction for screenplays, a speech instruction for speeches, etc.) but stay content-focused.

Examples:
TARGET: "Once it was over, the silence did not return all at once..."
INSTRUCTION: "Describe the lingering quiet that follows an overwhelming disruption."

TARGET: "Everyone moved faster than usual, not because panic helped..."
INSTRUCTION: "Describe a group preparing urgently for a difficult task."

TARGET: "ACT III. Slowly recovering from the rapture..."
INSTRUCTION: "Describe a garden in spring emerging from a period of intense bloom."
"""


def build_user_prompt(target_text: str, register: str) -> str:
    return f"REGISTER: {register}\n\nTARGET:\n{target_text[:1500]}\n\nINSTRUCTION:"


def call_claude(client, target_text: str, register: str, model: str) -> str:
    import anthropic

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=80,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(target_text, register)}],
        )
        text = resp.content[0].text if resp.content else ""
        text = text.strip().strip('"').strip("'").strip()
        # Take first line if multi-line.
        text = text.split("\n")[0].strip()
        return text
    except anthropic.APIError as exc:
        return f"[ERR:{type(exc).__name__}]"


def process_pair(client, pair: dict, model: str) -> dict:
    target = pair.get("target_text", "")
    register = pair.get("register", "poetry")
    if not target:
        return pair
    new_instr = call_claude(client, target, register, model)
    pair = dict(pair)
    pair["instruction_rule_based"] = pair.get("instruction", "")
    pair["instruction"] = new_instr
    return pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="data/pairs")
    parser.add_argument("--out-dir", default="data/pairs")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    args = parser.parse_args()

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

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_cost_approx = 0.0
    for split in args.splits:
        in_path = in_dir / f"{split}.jsonl"
        out_path = out_dir / f"{split}.llm.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        pairs = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs = pairs[: args.max_pairs]
        print(f"[{split}] processing {len(pairs)} pairs with {args.workers} workers via {args.model}")
        t0 = time.time()
        done = 0
        out_pairs: list[dict] = [None] * len(pairs)  # type: ignore
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_pair, client, p, args.model): i for i, p in enumerate(pairs)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    out_pairs[i] = fut.result()
                except Exception as exc:
                    out_pairs[i] = pairs[i]
                    out_pairs[i]["instruction"] = f"[ERR:{exc}]"
                done += 1
                if done % 25 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(pairs) - done) / rate if rate > 0 else 0
                    print(f"  [{split}] {done}/{len(pairs)} ({rate:.1f}/s, eta {eta:.0f}s)")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in out_pairs:
                f.write(json.dumps(p) + "\n")
        n_err = sum(1 for p in out_pairs if str(p.get("instruction", "")).startswith("[ERR"))
        # Rough cost: 300 input tokens + 30 output per call, haiku pricing.
        approx_cost = len(pairs) * (300 / 1e6 * 0.25 + 30 / 1e6 * 1.25)
        total_cost_approx += approx_cost
        elapsed = time.time() - t0
        print(f"[{split}] wrote {out_path} in {elapsed:.1f}s ({len(pairs)-n_err} ok, {n_err} errors, ~${approx_cost:.3f})")
    print(f"[done] approx total cost ~${total_cost_approx:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
