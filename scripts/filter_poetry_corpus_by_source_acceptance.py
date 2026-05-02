#!/usr/bin/env python3
"""Filter corpus rows using source acceptance decisions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--source-decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision", default="accept")
    args = parser.parse_args()

    rows = read_jsonl(args.corpus)
    decisions = read_jsonl(args.source_decisions)
    accepted_sources = {
        str(row["source_id"])
        for row in decisions
        if row.get("decision") == args.decision
    }
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        if str(row.get("source_id")) in accepted_sources:
            out = dict(row)
            out["source_acceptance"] = args.decision
            kept.append(out)
        else:
            rejected.append({"corpus_id": row.get("corpus_id"), "source_id": row.get("source_id"), "author_id": row.get("author_id")})

    output = args.output
    write_jsonl(output / "corpus.jsonl", kept)
    write_jsonl(output / "rejected_by_source_acceptance.jsonl", rejected)
    author_counts = Counter(str(row["author_id"]) for row in kept)
    source_counts = Counter(str(row["source_id"]) for row in kept)
    manifest = {
        "source_corpus": str(args.corpus),
        "source_decisions": str(args.source_decisions),
        "decision": args.decision,
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "rows_rejected": len(rejected),
        "sources_kept": len(source_counts),
        "authors": dict(sorted(author_counts.items())),
        "authors_count": len(author_counts),
        "authors_ge_25": sum(count >= 25 for count in author_counts.values()),
        "authors_ge_100": sum(count >= 100 for count in author_counts.values()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
