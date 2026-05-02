#!/usr/bin/env python3
"""Offline own-vs-swap stylometric audit for generated samples.

This is not a replacement for an LLM judge. It is a stronger cheap diagnostic
than the small hand-built surface feature metric in eval_probes.py: compare
generated outputs to own vs swap references and heldout author prototypes using
character and word n-gram TF-IDF cosine similarity.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_for_pair(row: dict) -> str:
    return "\n\n".join(part for part in (row.get("ref_text", ""), row.get("target_text", "")) if part)


def fit_vectors(texts: list[str], mode: str):
    if mode == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
    elif mode == "word":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True)
    else:
        raise ValueError(mode)
    mat = vec.fit_transform(texts)
    return vec, mat


def cos(vec, a: str, b: str) -> float:
    mat = vec.transform([a, b])
    return float(cosine_similarity(mat[0], mat[1])[0, 0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--heldout", action="append", required=True, help="val/test JSONL; may be repeated")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["char", "word"], default="char")
    args = parser.parse_args()

    samples = load_jsonl(Path(args.samples))
    probes = load_jsonl(Path(args.probes))
    heldout_rows: list[dict] = []
    for path in args.heldout:
        heldout_rows.extend(load_jsonl(Path(path)))

    probe_by_id = {row["probe_id"]: row for row in probes}

    author_docs: dict[str, dict[str, str]] = defaultdict(dict)
    for row in heldout_rows:
        author = row.get("author")
        if not author:
            continue
        for key in ("ref_doc_id", "target_doc_id"):
            doc_id = row.get(key)
            if not doc_id:
                continue
            text = row.get("ref_text" if key == "ref_doc_id" else "target_text", "")
            if text:
                author_docs[author][doc_id] = text

    corpus_texts = []
    for row in probes:
        corpus_texts.extend([row.get("reference_text", ""), row.get("swap_reference_text", ""), row.get("expected_target", "")])
    corpus_texts.extend(sample.get("text", "") for sample in samples)
    for docs in author_docs.values():
        corpus_texts.extend(docs.values())
    vec, _ = fit_vectors(corpus_texts, args.mode)

    def prototype(author: str, exclude: set[str]) -> str:
        docs = [txt for doc_id, txt in sorted(author_docs.get(author, {}).items()) if doc_id not in exclude]
        if not docs:
            return ""
        return "\n\n".join(docs[:12])

    records = []
    for sample in samples:
        if sample.get("variant") not in {"adapter", "adapter_swap", "no_ref", "prompted_baseline"}:
            continue
        if not isinstance(sample.get("step"), int) and sample.get("variant") != "prompted_baseline":
            continue
        probe = probe_by_id.get(sample.get("probe_id"))
        if not probe:
            continue
        own_author = probe["author"]
        swap_author = probe["swap_reference_author"]
        own_ref = probe.get("reference_text", "")
        swap_ref = probe.get("swap_reference_text", "")
        gen = sample.get("text", "")
        own_exclude = {probe.get("ref_doc_id", ""), probe.get("target_doc_id", "")}
        swap_exclude = {probe.get("swap_ref_doc_id", "")}
        own_proto = prototype(own_author, own_exclude) or own_ref
        swap_proto = prototype(swap_author, swap_exclude) or swap_ref
        rec = {
            "probe_id": sample["probe_id"],
            "variant": sample["variant"],
            "register": probe.get("register"),
            "own_author": own_author,
            "swap_author": swap_author,
            "own_ref_sim": cos(vec, gen, own_ref),
            "swap_ref_sim": cos(vec, gen, swap_ref),
            "own_proto_sim": cos(vec, gen, own_proto),
            "swap_proto_sim": cos(vec, gen, swap_proto),
        }
        rec["ref_advantage"] = rec["own_ref_sim"] - rec["swap_ref_sim"]
        rec["proto_advantage"] = rec["own_proto_sim"] - rec["swap_proto_sim"]
        if sample["variant"] == "adapter_swap":
            rec["swap_ref_advantage"] = rec["swap_ref_sim"] - rec["own_ref_sim"]
            rec["swap_proto_advantage"] = rec["swap_proto_sim"] - rec["own_proto_sim"]
        records.append(rec)

    def summarize(rows: list[dict], advantage_key: str) -> dict:
        if not rows:
            return {"n": 0}
        vals = [row[advantage_key] for row in rows]
        return {
            "n": len(rows),
            "wins": sum(1 for val in vals if val > 0),
            "win_rate": sum(1 for val in vals if val > 0) / len(vals),
            "mean_advantage": mean(vals),
            "median_advantage": sorted(vals)[len(vals) // 2],
            "min_advantage": min(vals),
            "max_advantage": max(vals),
        }

    summary: dict[str, object] = {"mode": args.mode}
    for variant, key in [
        ("adapter", "ref_advantage"),
        ("adapter", "proto_advantage"),
        ("adapter_swap", "swap_ref_advantage"),
        ("adapter_swap", "swap_proto_advantage"),
        ("prompted_baseline", "ref_advantage"),
        ("prompted_baseline", "proto_advantage"),
        ("no_ref", "ref_advantage"),
        ("no_ref", "proto_advantage"),
    ]:
        rows = [row for row in records if row["variant"] == variant]
        summary[f"{variant}_{key}"] = summarize(rows, key)
        for register in sorted({row["register"] for row in rows}):
            reg_rows = [row for row in rows if row["register"] == register]
            summary[f"{variant}_{register}_{key}"] = summarize(reg_rows, key)

    out = {"summary": summary, "per_sample": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
