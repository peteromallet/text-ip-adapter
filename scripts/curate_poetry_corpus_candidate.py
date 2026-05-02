#!/usr/bin/env python3
"""Apply deterministic author/source curation to a poetry corpus candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ARTIFACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "gutenberg_boilerplate": re.compile(r"Project Gutenberg|START OF (THIS|THE) PROJECT|END OF (THIS|THE) PROJECT", re.I),
    "toc_index": re.compile(r"\b(table of contents|contents|index|bibliography)\b", re.I),
    "editorial": re.compile(r"\b(preface|introduction|editor'?s note|transcriber'?s note|footnote)\b", re.I),
    "classroom_prompt": re.compile(r"\b(write a poem|submit|teacher|assessment|assignment|use these words|following lines)\b", re.I),
    "html_url": re.compile(r"<[^>]+>|https?://|www\.", re.I),
    "placeholder": re.compile(r"_\(\d+\)_|\[[^\]]{1,40}\]"),
    "ocr_noise": re.compile(r"(?:[|}{~]{2,}|[A-Za-z]{20,}| {5,})"),
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def norm_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def sha256_text(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def artifact_flags(text: str) -> list[str]:
    return [name for name, pattern in ARTIFACT_PATTERNS.items() if pattern.search(text)]


def source_title(row: dict) -> str:
    return str(row.get("source_work_title") or row.get("title") or "")


def row_title(row: dict) -> str:
    return str(row.get("title") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-corpus", type=Path, required=True)
    parser.add_argument("--input-source-results", type=Path)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cleaning-version", default="poetry_corpus_v0_curation_v0")
    args = parser.parse_args()

    curation = json.loads(args.curation.read_text(encoding="utf-8"))
    merges = curation.get("merge_author_ids", {})
    reject_authors = curation.get("reject_author_ids", {})
    reject_sources = curation.get("reject_source_ids", {})
    title_res = [re.compile(pattern, re.I) for pattern in curation.get("reject_source_title_regex", [])]
    row_title_res = [re.compile(pattern, re.I) for pattern in curation.get("reject_row_title_regex", [])]
    text_res = [re.compile(pattern, re.I | re.M) for pattern in curation.get("reject_text_regex", [])]
    drop_flagged = bool(curation.get("drop_audit_flagged_rows", True))

    source_results = list(read_jsonl(args.input_source_results)) if args.input_source_results and args.input_source_results.exists() else []
    source_meta = {row.get("source_id"): row for row in source_results}

    kept: list[dict] = []
    rejected: list[dict] = []
    seen_hashes: set[str] = set()
    reason_counts: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    merge_counts: Counter[str] = Counter()

    for row_index, row in enumerate(read_jsonl(args.input_corpus)):
        original_author_id = str(row.get("author_id") or "")
        source_id = str(row.get("source_id") or "")
        title = source_title(row)
        title_guess = row_title(row)
        reasons: list[str] = []

        if original_author_id in reject_authors:
            reasons.append("reject_author_id")
        if source_id in reject_sources:
            reasons.append("reject_source_id")
        if any(pattern.search(title) for pattern in title_res):
            reasons.append("reject_source_title_regex")
        if any(pattern.search(title_guess) for pattern in row_title_res):
            reasons.append("reject_row_title_regex")
        text = str(row.get("text") or "")
        if any(pattern.search(text) for pattern in text_res):
            reasons.append("reject_text_regex")
        if drop_flagged:
            flags = artifact_flags(text)
            reasons.extend(f"audit_{flag}" for flag in flags)

        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            rejected.append(
                {
                    "row_index": row_index,
                    "corpus_id": row.get("corpus_id"),
                    "author_id": original_author_id,
                    "source_id": source_id,
                    "source_work_title": title,
                    "reasons": sorted(set(reasons)),
                    "text_preview": str(row.get("text") or "")[:300],
                }
            )
            continue

        row = dict(row)
        merge = merges.get(original_author_id)
        if merge:
            row["original_author_id"] = original_author_id
            row["original_author_name"] = row.get("author_name")
            row["author_id"] = merge["author_id"]
            row["author_name"] = merge["author_name"]
            row["author_normalization_reason"] = merge.get("reason")
            merge_counts[f"{original_author_id}->{row['author_id']}"] += 1

        clean_hash = sha256_text(text)
        if clean_hash in seen_hashes:
            reason_counts["duplicate_after_curation"] += 1
            rejected.append(
                {
                    "row_index": row_index,
                    "corpus_id": row.get("corpus_id"),
                    "author_id": row.get("author_id"),
                    "source_id": source_id,
                    "source_work_title": title,
                    "reasons": ["duplicate_after_curation"],
                    "text_preview": text[:300],
                }
            )
            continue
        seen_hashes.add(clean_hash)

        row["clean_sha256"] = clean_hash
        row["cleaning_version"] = args.cleaning_version
        row["corpus_id"] = f"{row['author_id']}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
        row["curation_source"] = str(args.curation)
        row["source_fetch_status"] = source_meta.get(source_id, {}).get("fetch_status", row.get("source_fetch_status"))
        kept.append(row)
        author_counts[str(row.get("author_id"))] += 1
        source_counts[source_id] += 1

    low_authors = {author: count for author, count in sorted(author_counts.items()) if count < 25}
    summary = {
        "input_corpus": str(args.input_corpus),
        "curation": str(args.curation),
        "rows_in": len(kept) + len(rejected),
        "rows_kept": len(kept),
        "rows_rejected": len(rejected),
        "authors": len(author_counts),
        "sources": len(source_counts),
        "authors_ge_25": sum(count >= 25 for count in author_counts.values()),
        "low_authors_under_25": low_authors,
        "reason_counts": dict(reason_counts.most_common()),
        "merge_counts": dict(merge_counts.most_common()),
        "top_author_counts": author_counts.most_common(40),
        "top_source_counts": source_counts.most_common(40),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "corpus.jsonl", kept)
    write_jsonl(args.output_dir / "rejected_rows.jsonl", rejected)
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
