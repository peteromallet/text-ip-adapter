#!/usr/bin/env python3
"""Audit a clean-poetry corpus JSONL candidate.

This checks corpus-first records, not training pairs. It is intentionally
deterministic and conservative: anything that looks like source boilerplate,
classroom prompts, HTML, URL residue, duplicates, or prose drift is surfaced for
review before pair generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


REQUIRED_FIELDS = (
    "corpus_id",
    "author_id",
    "author_name",
    "text",
    "source_id",
    "source_name",
    "source_url",
    "license",
    "public_domain_basis",
    "clean_sha256",
)

WORD_RE = re.compile(r"[A-Za-z']+")
ARTIFACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "gutenberg_boilerplate": re.compile(r"Project Gutenberg|START OF (THIS|THE) PROJECT|END OF (THIS|THE) PROJECT", re.I),
    "toc_index": re.compile(r"\b(table of contents|contents|index|bibliography)\b", re.I),
    "editorial": re.compile(r"\b(preface|introduction|editor'?s note|transcriber'?s note|footnote)\b", re.I),
    "classroom_prompt": re.compile(r"\b(write a poem|submit|teacher|assessment|assignment|use these words|following lines)\b", re.I),
    "html_url": re.compile(r"<[^>]+>|https?://|www\.", re.I),
    "placeholder": re.compile(r"_\(\d+\)_|\[[^\]]{1,40}\]"),
    "ocr_noise": re.compile(r"(?:[|}{~]{2,}|[A-Za-z]{20,}| {5,})"),
}


def read_jsonl(path: Path) -> Iterable[tuple[int, dict | Exception]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    yield line_no, json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    yield line_no, exc


def normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def verse_stats(text: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = WORD_RE.findall(text)
    line_words = [line_word_count(line) for line in lines]
    if not line_words:
        return {
            "line_count": 0,
            "word_count": len(words),
            "median_line_words": 0,
            "long_line_count": 0,
            "prose_like": True,
        }
    median_line_words = median(line_words)
    long_line_count = sum(count > 30 for count in line_words)
    prose_like = len(words) < 20 or len(lines) < 4 or median_line_words > 18 or long_line_count >= 3
    return {
        "line_count": len(lines),
        "word_count": len(words),
        "median_line_words": median_line_words,
        "long_line_count": long_line_count,
        "prose_like": prose_like,
    }


def audit_row(row: dict) -> list[str]:
    flags: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] in ("", None):
            flags.append(f"missing_{field}")
    text = str(row.get("text", ""))
    if "clean_sha256" in row and row.get("clean_sha256") != sha256_text(text):
        flags.append("clean_sha256_mismatch")
    if not str(row.get("license", "")).startswith("public_domain"):
        flags.append("license_not_public_domain")
    if not row.get("public_domain_basis"):
        flags.append("missing_public_domain_basis")
    for name, pattern in ARTIFACT_PATTERNS.items():
        if pattern.search(text):
            flags.append(name)
    stats = verse_stats(text)
    if stats["prose_like"]:
        flags.append("prose_like")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_jsonl", type=Path)
    parser.add_argument("--min-authors", type=int, default=25)
    parser.add_argument("--min-records-per-author", type=int, default=10)
    parser.add_argument("--max-artifact-rate", type=float, default=0.01)
    parser.add_argument("--write-flagged", type=Path)
    parser.add_argument("--write-summary", type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    parse_errors: list[dict] = []
    for line_no, row in read_jsonl(args.corpus_jsonl):
        if isinstance(row, Exception):
            parse_errors.append({"line_no": line_no, "error": repr(row)})
        else:
            rows.append(row)

    clean_hashes: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    flagged_rows: list[dict] = []
    by_author_flags: dict[str, Counter[str]] = defaultdict(Counter)

    for row_index, row in enumerate(rows):
        author_id = str(row.get("author_id", ""))
        author_counts[author_id] += 1
        clean_hash = str(row.get("clean_sha256") or sha256_text(str(row.get("text", ""))))
        clean_hashes[clean_hash] += 1
        flags = audit_row(row)
        if clean_hashes[clean_hash] > 1:
            flags.append("duplicate_clean_text")
        for flag in flags:
            flag_counts[flag] += 1
            by_author_flags[author_id][flag] += 1
        if flags:
            flagged_rows.append({"row_index": row_index, "corpus_id": row.get("corpus_id"), "author_id": author_id, "flags": flags})

    retained_author_count = sum(count >= args.min_records_per_author for count in author_counts.values())
    artifact_flags = set(ARTIFACT_PATTERNS) | {"duplicate_clean_text", "clean_sha256_mismatch"}
    artifact_hits = sum(count for flag, count in flag_counts.items() if flag in artifact_flags)
    artifact_rate = artifact_hits / max(len(rows), 1)
    low_authors = {author: count for author, count in sorted(author_counts.items()) if count < args.min_records_per_author}

    gates = {
        "min_authors": retained_author_count >= args.min_authors,
        "min_records_per_author": not low_authors,
        "max_artifact_rate": artifact_rate <= args.max_artifact_rate,
        "no_parse_errors": not parse_errors,
        "no_duplicate_clean_text": flag_counts.get("duplicate_clean_text", 0) == 0,
        "all_public_domain": flag_counts.get("license_not_public_domain", 0) == 0
        and flag_counts.get("missing_public_domain_basis", 0) == 0,
    }
    summary = {
        "path": str(args.corpus_jsonl),
        "rows": len(rows),
        "authors": len(author_counts),
        "retained_authors_at_min_records": retained_author_count,
        "min_records_per_author": args.min_records_per_author,
        "low_authors": low_authors,
        "unique_clean_hashes": len(clean_hashes),
        "duplicate_clean_hash_groups": sum(count > 1 for count in clean_hashes.values()),
        "flag_counts": dict(sorted(flag_counts.items())),
        "artifact_rate": artifact_rate,
        "parse_errors": parse_errors[:20],
        "gates": gates,
        "passed": all(gates.values()),
        "top_author_counts": author_counts.most_common(20),
        "top_author_flags": {
            author: dict(flags.most_common(10))
            for author, flags in sorted(by_author_flags.items())
            if flags
        },
    }

    if args.write_flagged:
        args.write_flagged.parent.mkdir(parents=True, exist_ok=True)
        args.write_flagged.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in flagged_rows),
            encoding="utf-8",
        )
    if args.write_summary:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
