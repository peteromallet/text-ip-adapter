#!/usr/bin/env python3
"""Create source-level acceptance decisions for a poetry corpus candidate."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


BAD_TITLE_RE = re.compile(
    r"\b(lecture|criticism|magazine|monthly|journal|review|antholog|treasury|"
    r"selected|selections|reliques|miscellaneous|catalog|bibliograph|index|"
    r"manual|school|college|book list|library of|poets and poetry|lexicon|"
    r"extracts|miscellanies|minstrelsy|dramatic works|works of|translations?|"
    r"eminent hands|biographical|critical notices|copious notes|mysticism|"
    r"concordance|grammar|prose and poetry|history of|student|testament|"
    r"dictionary|interspersed with some pieces|edited by|hymns?|prayers?|"
    r"eucharistica|messianica|sacred verses|religious|romance of|novels?|"
    r"literature|old testament|plays for|schools?)\b",
    re.I,
)
BAD_AUTHOR_RE = re.compile(
    r"\b(editor|edited|students?|company|anonymous|various|publisher|academy|"
    r"gunston|dr\.?\s*seuss|father|sister|rev|s\.?\s*j|ed|comp|compiler|"
    r"translator|auteur)\b|;|/|&",
    re.I,
)
GOOD_TITLE_RE = re.compile(r"\b(poems|poetry|lyrics|sonnets|songs|ballads|lays|verses|verse|odes)\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: object) -> str:
    return "" if value is None else str(value).strip()


def decide_source(row: dict, *, min_records: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    source_id = norm(row.get("source_id"))
    source_name = norm(row.get("source_name"))
    if not source_name:
        if source_id.startswith("gutenberg_"):
            source_name = "project_gutenberg"
        elif source_id.startswith("internet_archive_"):
            source_name = "internet_archive"
    author = norm(row.get("author_name"))
    title = norm(row.get("source_work_title"))
    records = int(row.get("records") or 0)
    fetch_status = norm(row.get("fetch_status"))

    if fetch_status != "ok":
        return "reject", [f"fetch_{fetch_status or 'not_ok'}"]
    if records < min_records:
        return "reject", ["too_few_records"]
    if source_name == "project_gutenberg":
        return "accept", ["curated_project_gutenberg"]

    if not author:
        reasons.append("missing_author")
    if BAD_AUTHOR_RE.search(author):
        reasons.append("risky_author")
    if not title:
        reasons.append("missing_title")
    if BAD_TITLE_RE.search(title):
        reasons.append("risky_title")
    if not GOOD_TITLE_RE.search(title):
        reasons.append("title_not_poetry_collection")
    if "internet_archive" not in source_id:
        reasons.append("unknown_non_gutenberg_source")

    if reasons:
        return "review_or_reject", reasons
    return "accept", ["strict_single_author_poetry_source"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--min-records", type=int, default=25)
    args = parser.parse_args()

    rows = read_jsonl(args.source_results)
    decisions: list[dict] = []
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        decision, reasons = decide_source(row, min_records=args.min_records)
        counts[decision] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        decisions.append(
            {
                "source_id": row.get("source_id"),
                "source_name": row.get("source_name"),
                "author_name": row.get("author_name"),
                "source_work_title": row.get("source_work_title"),
                "records": row.get("records", 0),
                "rejects": row.get("rejects", 0),
                "fetch_status": row.get("fetch_status"),
                "decision": decision,
                "reasons": reasons,
            }
        )

    decisions.sort(key=lambda row: (row["decision"] != "accept", -int(row.get("records") or 0), str(row.get("source_id"))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    summary = {
        "source_results": str(args.source_results),
        "sources": len(decisions),
        "decision_counts": dict(counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "accepted_records": sum(int(row.get("records") or 0) for row in decisions if row["decision"] == "accept"),
        "accepted_sources": sum(row["decision"] == "accept" for row in decisions),
        "review_sources": sum(row["decision"] == "review_or_reject" for row in decisions),
        "rejected_sources": sum(row["decision"] == "reject" for row in decisions),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
