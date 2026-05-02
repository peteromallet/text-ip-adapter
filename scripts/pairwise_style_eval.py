#!/usr/bin/env python3
"""Pairwise style/prompt-adherence eval for sampled adapter generations.

This is a deterministic evaluator, not a replacement for a human/LLM judge. It
compares variants pairwise on the same probe using reference-style feature
similarity plus simple prompt-adherence penalties.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


META_RE = re.compile(
    r"(?i)(write a|reference passage|reference style|instruction:|response:|"
    r"here is|example:|analysis:|exercise|no response|not qualified|what do they mean|"
    r"this is what i wrote|any advice|poem is written|the poet|stanza)"
)
PROSE_RE = re.compile(r"(?i)(chapter|lesson|province|cuisine|persuasive writing|faroese|not been able)")
TOKEN_RE = re.compile(r"[A-Za-z']+")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word in tokens(text):
        padded = f" {word} "
        for n in range(n_min, n_max + 1):
            for i in range(max(0, len(padded) - n + 1)):
                counts[padded[i : i + n]] += 1
    return counts


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / max(1e-9, na * nb)


def lineation_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        return 1.0
    if len(lines) >= 2:
        return 0.6
    return 0.0


def prompt_adherence(text: str) -> dict[str, float | int | bool]:
    toks = tokens(text)
    chars = len(text)
    meta = bool(META_RE.search(text))
    prose = bool(PROSE_RE.search(text))
    too_short = chars < 40
    too_long = chars > 900
    line_score = lineation_score(text)
    penalty = 0.0
    if meta:
        penalty += 0.35
    if prose:
        penalty += 0.25
    if too_short:
        penalty += 0.20
    if too_long:
        penalty += 0.10
    if line_score == 0.0:
        penalty += 0.10
    return {
        "chars": chars,
        "tokens": len(toks),
        "lineation_score": line_score,
        "meta": meta,
        "prose": prose,
        "too_short": too_short,
        "too_long": too_long,
        "penalty": penalty,
    }


def score_sample(sample: dict, probe: dict) -> dict:
    text = sample.get("text", "")
    own = probe.get("reference_text", "")
    swap = probe.get("swap_reference_text", "")
    text_vec = char_ngrams(text)
    own_sim = cosine(text_vec, char_ngrams(own))
    swap_sim = cosine(text_vec, char_ngrams(swap))
    adherence = prompt_adherence(text)
    style_advantage = own_sim - swap_sim
    score = style_advantage + 0.12 * float(adherence["lineation_score"]) - float(adherence["penalty"])
    return {
        "probe_id": sample.get("probe_id"),
        "variant": sample.get("variant"),
        "own_sim": own_sim,
        "swap_sim": swap_sim,
        "style_advantage": style_advantage,
        "adherence": adherence,
        "score": score,
    }


def compare(a: dict, b: dict, margin: float) -> str:
    delta = a["score"] - b["score"]
    if delta > margin:
        return a["variant"]
    if delta < -margin:
        return b["variant"]
    return "tie"


def summarize_pairwise(records: list[dict], left: str, right: str, margin: float) -> dict:
    by_probe: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in records:
        by_probe[rec["probe_id"]][rec["variant"]] = rec
    outcomes = []
    for pid, variants in sorted(by_probe.items()):
        if left not in variants or right not in variants:
            continue
        winner = compare(variants[left], variants[right], margin)
        outcomes.append(
            {
                "probe_id": pid,
                "left": left,
                "right": right,
                "winner": winner,
                "left_score": variants[left]["score"],
                "right_score": variants[right]["score"],
                "delta": variants[left]["score"] - variants[right]["score"],
            }
        )
    n = len(outcomes)
    if not n:
        return {"n": 0, "left": left, "right": right}
    left_wins = sum(1 for row in outcomes if row["winner"] == left)
    right_wins = sum(1 for row in outcomes if row["winner"] == right)
    ties = sum(1 for row in outcomes if row["winner"] == "tie")
    return {
        "left": left,
        "right": right,
        "n": n,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "left_win_rate": left_wins / n,
        "right_win_rate": right_wins / n,
        "mean_delta_left_minus_right": mean(row["delta"] for row in outcomes),
        "per_probe": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.samples))
    probes = {row["probe_id"]: row for row in load_jsonl(Path(args.probes))}
    scored = [score_sample(sample, probes[sample["probe_id"]]) for sample in samples if sample.get("probe_id") in probes]
    comparisons = {
        "adapter_vs_no_ref": summarize_pairwise(scored, "adapter", "no_ref", args.margin),
        "adapter_prompted_vs_prompted_baseline": summarize_pairwise(
            scored, "adapter_prompted", "prompted_baseline", args.margin
        ),
        "adapter_vs_adapter_swap": summarize_pairwise(scored, "adapter", "adapter_swap", args.margin),
    }
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for rec in scored:
        by_variant[rec["variant"]].append(rec)
    variant_summary = {
        variant: {
            "n": len(rows),
            "mean_score": mean(row["score"] for row in rows),
            "mean_style_advantage": mean(row["style_advantage"] for row in rows),
            "meta_hits": sum(1 for row in rows if row["adherence"]["meta"]),
            "prose_hits": sum(1 for row in rows if row["adherence"]["prose"]),
        }
        for variant, rows in sorted(by_variant.items())
    }
    out = {"variant_summary": variant_summary, "comparisons": comparisons, "per_sample": scored}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"variant_summary": variant_summary, "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "per_probe"} for k, v in comparisons.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
