#!/usr/bin/env python3
"""Build a stricter poetry pair set gated on distinctive, coherent style.

The v5.7 pair audit removed obvious artifacts. This script adds a different
gate: prefer medium-length poem chunks that are stylometrically closer to their
own author prototype than to other-author prototypes, and whose ref/target pair
is not wildly incoherent.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z']+")
GENERIC_RE = re.compile(
    r"\b(sun|moon|stars?|birds?|flowers?|trees?|grass|wind|rain|heart|love|"
    r"dream|song|sing|sweet|dear|fair|bright|night|day|sky|sea)\b",
    re.IGNORECASE,
)
PROSE_CUE_RE = re.compile(
    r"\b(chapter|section|volume|copyright|publisher|illustration|contents|"
    r"preface|introduction|transcriber|page|figure|appendix)\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_stats(text: str) -> dict[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    toks = TOKEN_RE.findall(text.lower())
    unique = set(toks)
    generic_hits = len(GENERIC_RE.findall(text))
    chars = len(text)
    return {
        "chars": chars,
        "lines": len(lines),
        "tokens": len(toks),
        "type_token": len(unique) / max(1, len(toks)),
        "avg_line_chars": mean([len(line) for line in lines]) if lines else 0.0,
        "punct_per_100": 100.0 * sum(1 for ch in text if ch in ",;:!?—-") / max(1, chars),
        "quote_count": text.count('"') + text.count("'"),
        "generic_per_100": 100.0 * generic_hits / max(1, len(toks)),
        "prose_cue": 1.0 if PROSE_CUE_RE.search(text) else 0.0,
    }


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter[str]:
    words = TOKEN_RE.findall(text.lower())
    counts: Counter[str] = Counter()
    for word in words:
        padded = f" {word} "
        for n in range(n_min, n_max + 1):
            for i in range(0, max(0, len(padded) - n + 1)):
                counts[padded[i : i + n]] += 1
    return counts


def cos_counts(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(val * b.get(key, 0) for key, val in a.items())
    na = sum(val * val for val in a.values()) ** 0.5
    nb = sum(val * val for val in b.values()) ** 0.5
    return dot / max(1e-9, na * nb)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": median(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": mean(values),
    }


def collect_docs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        for prefix, text_key, title_key, source_key in [
            ("ref", "ref_text", "ref_title", "ref_source_id"),
            ("target", "target_text", "target_title", "target_source_id"),
        ]:
            doc_id = row.get(f"{prefix}_doc_id")
            text = row.get(text_key, "")
            if not doc_id or not text:
                continue
            docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "author": row.get("author"),
                    "author_name": row.get("author_name"),
                    "title": row.get(title_key, ""),
                    "source_id": row.get(source_key, ""),
                    "text": text,
                },
            )
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v5_7_poetry_pair_audited_min25")
    parser.add_argument("--output-dir", default="data/pairs_v5_9_poetry_distinctive_medium")
    parser.add_argument("--report-dir", default="../prompt-adapters/projects/poetry-clean-corpus/distinctive_style_v1")
    parser.add_argument("--min-chars", type=int, default=220)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--min-doc-advantage", type=float, default=0.015)
    parser.add_argument("--min-pair-cohesion", type=float, default=0.035)
    parser.add_argument("--max-generic-per-100", type=float, default=9.0)
    args = parser.parse_args()

    repo = Path.cwd()
    input_dir = repo / args.input_dir
    output_dir = repo / args.output_dir
    report_dir = (repo / args.report_dir).resolve()

    train = load_jsonl(input_dir / "train.jsonl")
    val = load_jsonl(input_dir / "val.jsonl")
    test = load_jsonl(input_dir / "test.jsonl")
    probes = load_jsonl(input_dir / "probes_balanced_n24.jsonl")

    docs = collect_docs(train)
    by_author: dict[str, list[str]] = defaultdict(list)
    for doc in docs.values():
        by_author[doc["author"]].append(doc["doc_id"])

    doc_vecs = {doc_id: char_ngrams(doc["text"]) for doc_id, doc in docs.items()}
    author_proto: dict[str, str] = {
        author: "\n\n".join(docs[doc_id]["text"] for doc_id in doc_ids)
        for author, doc_ids in by_author.items()
    }
    author_vecs = {author: char_ngrams(text) for author, text in author_proto.items()}

    doc_scores: dict[str, dict[str, Any]] = {}
    for doc_id, doc in docs.items():
        author = doc["author"]
        own_other = [other for other in by_author[author] if other != doc_id]
        own_proto = "\n\n".join(docs[other]["text"] for other in own_other) or doc["text"]
        own_vec = char_ngrams(own_proto)
        own_sim = cos_counts(doc_vecs[doc_id], own_vec)
        other_sims = [
            (other_author, cos_counts(doc_vecs[doc_id], proto_vec))
            for other_author, proto_vec in author_vecs.items()
            if other_author != author
        ]
        nearest_author, nearest_sim = max(other_sims, key=lambda item: item[1])
        stats = text_stats(doc["text"])
        score = {
            **doc,
            **stats,
            "own_author_sim": own_sim,
            "nearest_other_author": nearest_author,
            "nearest_other_author_sim": nearest_sim,
            "author_advantage": own_sim - nearest_sim,
        }
        score["medium_length"] = args.min_chars <= stats["chars"] <= args.max_chars and stats["lines"] >= args.min_lines
        score["distinctive"] = score["author_advantage"] >= args.min_doc_advantage
        score["not_too_generic"] = stats["generic_per_100"] <= args.max_generic_per_100
        score["no_prose_cue"] = stats["prose_cue"] == 0.0
        score["keep_doc"] = bool(
            score["medium_length"]
            and score["distinctive"]
            and score["not_too_generic"]
            and score["no_prose_cue"]
        )
        doc_scores[doc_id] = score

    kept: list[dict[str, Any]] = []
    pair_scores: list[dict[str, Any]] = []
    reject_reasons = Counter()
    for row in train:
        ref_id = row["ref_doc_id"]
        tgt_id = row["target_doc_id"]
        ref = doc_scores[ref_id]
        tgt = doc_scores[tgt_id]
        cohesion = cos_counts(doc_vecs[ref_id], doc_vecs[tgt_id])
        min_adv = min(ref["author_advantage"], tgt["author_advantage"])
        reasons = []
        if not ref["keep_doc"]:
            reasons.append("ref_doc_not_distinctive_medium")
        if not tgt["keep_doc"]:
            reasons.append("target_doc_not_distinctive_medium")
        if cohesion < args.min_pair_cohesion:
            reasons.append("low_ref_target_cohesion")
        if ref["chars"] < args.min_chars or tgt["chars"] < args.min_chars:
            reasons.append("too_short")
        if ref["chars"] > args.max_chars or tgt["chars"] > args.max_chars:
            reasons.append("too_long")
        keep = not reasons
        if keep:
            out = dict(row)
            out["distinctive_style_score"] = {
                "ref_author_advantage": ref["author_advantage"],
                "target_author_advantage": tgt["author_advantage"],
                "min_author_advantage": min_adv,
                "ref_target_cohesion": cohesion,
                "ref_chars": ref["chars"],
                "target_chars": tgt["chars"],
                "ref_generic_per_100": ref["generic_per_100"],
                "target_generic_per_100": tgt["generic_per_100"],
            }
            kept.append(out)
        else:
            reject_reasons.update(reasons)
        pair_scores.append(
            {
                "pair_id": row["pair_id"],
                "author": row["author"],
                "keep": keep,
                "reasons": reasons,
                "ref_doc_id": ref_id,
                "target_doc_id": tgt_id,
                "ref_author_advantage": ref["author_advantage"],
                "target_author_advantage": tgt["author_advantage"],
                "min_author_advantage": min_adv,
                "ref_target_cohesion": cohesion,
                "ref_chars": ref["chars"],
                "target_chars": tgt["chars"],
                "ref_generic_per_100": ref["generic_per_100"],
                "target_generic_per_100": tgt["generic_per_100"],
            }
        )

    kept_by_author = Counter(row["author"] for row in kept)
    final_train = [row for row in kept if kept_by_author[row["author"]] >= 20]
    dropped_low_author = {
        author: count for author, count in sorted(kept_by_author.items()) if count < 20
    }
    final_author_counts = Counter(row["author"] for row in final_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", final_train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "probes_balanced_n24.jsonl", probes)
    write_jsonl(output_dir / "probes_balanced.jsonl", probes)

    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "doc_scores.jsonl", sorted(doc_scores.values(), key=lambda r: (r["author"], r["doc_id"])))
    write_jsonl(report_dir / "pair_scores.jsonl", pair_scores)

    summary = {
        "source": str(input_dir),
        "output": str(output_dir),
        "thresholds": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_lines": args.min_lines,
            "min_doc_advantage": args.min_doc_advantage,
            "min_pair_cohesion": args.min_pair_cohesion,
            "max_generic_per_100": args.max_generic_per_100,
            "min_pairs_per_author_after_filter": 20,
        },
        "docs": {
            "total": len(doc_scores),
            "keep": sum(1 for row in doc_scores.values() if row["keep_doc"]),
            "author_advantage": summarize([row["author_advantage"] for row in doc_scores.values()]),
            "chars": summarize([row["chars"] for row in doc_scores.values()]),
            "generic_per_100": summarize([row["generic_per_100"] for row in doc_scores.values()]),
        },
        "pairs": {
            "source_train": len(train),
            "kept_before_min_author": len(kept),
            "final_train": len(final_train),
            "source_authors": len(set(row["author"] for row in train)),
            "final_authors": len(final_author_counts),
            "reject_reasons": dict(reject_reasons.most_common()),
            "dropped_low_author_after_filter": dropped_low_author,
            "final_author_counts": dict(sorted(final_author_counts.items())),
            "min_author_advantage": summarize([row["min_author_advantage"] for row in pair_scores]),
            "ref_target_cohesion": summarize([row["ref_target_cohesion"] for row in pair_scores]),
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "description": "v5.7 pair-audited train split filtered to medium-length, stylometrically distinctive/coherent poetry pairs.",
        "source_pair_set": str(input_dir),
        "distinctive_style_report": str(report_dir),
        "splits": {},
    }
    for split in ["train", "val", "test", "probes_balanced", "probes_balanced_n24"]:
        path = output_dir / f"{split}.jsonl"
        rows = load_jsonl(path)
        manifest["splits"][split] = {
            "rows": len(rows),
            "authors": len(set(row.get("author") for row in rows if row.get("author"))),
            "sha256": sha256_file(path),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Distinctive style filter v1",
        "",
        f"Source train rows: {len(train)}",
        f"Kept before min-author gate: {len(kept)}",
        f"Final train rows: {len(final_train)}",
        f"Final train authors: {len(final_author_counts)}",
        "",
        "## Thresholds",
        "",
        json.dumps(summary["thresholds"], indent=2),
        "",
        "## Main reject reasons",
        "",
    ]
    for reason, count in reject_reasons.most_common():
        md.append(f"- `{reason}`: {count}")
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is intentionally stricter than artifact cleanup. It keeps rows whose chunk-level",
            "character n-gram style is closer to same-author context than other-author context,",
            "then keeps pairs whose reference and target are medium-length and not too far apart",
            "stylometrically. It is a candidate training split, not a final proof of style quality.",
            "",
        ]
    )
    (report_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
