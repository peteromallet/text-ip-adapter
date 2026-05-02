#!/usr/bin/env python3
"""Create an author/source review queue for a poetry corpus candidate."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RISK_AUTHOR_RE = re.compile(
    r"\b(ed|comp|compiler|translator|fellow|auteur|rev|s_j|sister|father|"
    r"pope|baron|earl|sir|lady|publisher)\b|_and_|_\d{3,4}",
    re.I,
)
RISK_TITLE_RE = re.compile(
    r"\b(antholog|library of|poets and poetry|lexicon|extracts|miscellanies|"
    r"minstrelsy|reliques|dramatic works|works of|translations?|"
    r"biographical|critical notices|copious notes|history of|student|"
    r"testament|dictionary|concordance|grammar|prose and poetry)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.corpus)
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author_id"])].append(row)

    queue: list[dict] = []
    reason_counts: Counter[str] = Counter()
    for author_id, author_rows in sorted(by_author.items()):
        source_ids = sorted({str(row.get("source_id")) for row in author_rows})
        source_names = sorted({str(row.get("source_name")) for row in author_rows})
        titles = sorted({str(row.get("source_work_title") or "") for row in author_rows if row.get("source_work_title")})
        display_names = sorted({str(row.get("author_name") or "") for row in author_rows if row.get("author_name")})
        reasons: list[str] = []
        if "project_gutenberg" not in source_names:
            reasons.append("ia_or_non_gutenberg_author")
        if RISK_AUTHOR_RE.search(author_id):
            reasons.append("risky_author_string")
        if any(RISK_TITLE_RE.search(title) for title in titles):
            reasons.append("risky_source_title")
        if len(source_ids) == 1 and len(author_rows) > 500:
            reasons.append("single_source_high_volume")
        if len(display_names) > 1:
            reasons.append("multiple_display_names")
        if len(author_rows) < 25:
            reasons.append("low_record_count")
        for reason in reasons:
            reason_counts[reason] += 1
        queue.append(
            {
                "author_id": author_id,
                "record_count": len(author_rows),
                "source_count": len(source_ids),
                "source_names": source_names,
                "display_names": display_names[:10],
                "source_ids": source_ids[:20],
                "source_titles": titles[:20],
                "review_reasons": reasons,
                "default_decision": "review" if reasons else "keep_candidate",
            }
        )

    queue.sort(key=lambda row: (row["default_decision"] != "review", -row["record_count"], row["author_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queue),
        encoding="utf-8",
    )
    summary = {
        "corpus": str(args.corpus),
        "authors": len(queue),
        "review_authors": sum(row["default_decision"] == "review" for row in queue),
        "keep_candidate_authors": sum(row["default_decision"] == "keep_candidate" for row in queue),
        "reason_counts": dict(reason_counts.most_common()),
        "top_review": [row for row in queue if row["default_decision"] == "review"][:25],
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
