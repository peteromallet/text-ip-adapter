#!/usr/bin/env python3
"""Create an initial poetry source manifest from known source candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_ip_adapter.data.ingest_poetry import GUTENBERG_POETS, gutenberg_url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/poetry_source_manifest_v0_candidates.jsonl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in GUTENBERG_POETS:
        author_id = entry["author"]
        book_id = entry["book_id"]
        rows.append(
            {
                "source_id": f"gutenberg_{book_id}",
                "author_id": author_id,
                "author_name": author_id.replace("_", " ").title(),
                "source_name": "project_gutenberg",
                "source_url": gutenberg_url(book_id),
                "source_work_title": None,
                "source_publication_year": None,
                "license": "public_domain_us",
                "public_domain_basis": "project_gutenberg_public_domain_us_candidate",
                "expected_register": "poetry",
                "risk_notes": "Candidate inherited from legacy poetry ingestor; must pass source-native boundary and artifact audit before acceptance.",
                "status": "candidate",
            }
        )

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.relative_to(repo_root)), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
