#!/usr/bin/env python3
"""Profile poetry source manifest candidates."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def looks_anthology(row: dict) -> bool:
    title = norm(row.get("source_work_title")).lower()
    author = norm(row.get("author_name")).lower()
    subjects = " ".join(str(x).lower() for x in row.get("ia_subject", []) if x)
    text = " ".join([title, author, subjects])
    return bool(re.search(r"\b(antholog|collection|selected|miscellaneous|various|treasury|magazine|journal)\b", text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_jsonl", type=Path)
    parser.add_argument("--write-summary", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest_jsonl)
    by_source = Counter(norm(r.get("source_name")) for r in rows)
    by_status = Counter(norm(r.get("status")) for r in rows)
    by_license = Counter(norm(r.get("license")) for r in rows)
    by_author = Counter(norm(r.get("author_name")) or "<missing>" for r in rows)
    by_year = Counter(str(r.get("source_publication_year") or "<missing>") for r in rows)
    missing_author = sum(not norm(r.get("author_name")) for r in rows)
    missing_title = sum(not norm(r.get("source_work_title")) for r in rows)
    anthology_like = sum(looks_anthology(r) for r in rows)
    summary = {
        "path": str(args.manifest_jsonl),
        "rows": len(rows),
        "source_names": dict(by_source.most_common()),
        "statuses": dict(by_status.most_common()),
        "licenses": dict(by_license.most_common()),
        "unique_author_names": len(by_author),
        "missing_author": missing_author,
        "missing_title": missing_title,
        "anthology_like": anthology_like,
        "top_authors": by_author.most_common(30),
        "top_years": by_year.most_common(30),
        "sample_rows": rows[:10],
    }
    if args.write_summary:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
