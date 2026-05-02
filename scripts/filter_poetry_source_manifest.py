#!/usr/bin/env python3
"""Filter poetry source manifests into a high-precision candidate subset."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_TITLE_RE = re.compile(
    r"\b(lecture|influence of poetry|criticism|magazine|monthly|journal|review|"
    r"antholog|treasury|selected|selections|reliques|miscellaneous|"
    r"catalog|bibliograph|index|manual|school|college monthly|book list|"
    r"library of|poets and poetry|lexicon|extracts|miscellanies|minstrelsy|"
    r"dramatic works|works of|translations?|eminent hands|biographical|"
    r"critical notices|copious notes|mysticism|concordance|collected in|"
    r"plays for|hymns? for public|all ages and tongues)\b",
    re.I,
)
BAD_AUTHOR_RE = re.compile(
    r"\b(editor|edited|students?|company|anonymous|various|voices of today|"
    r"publisher|academy|gunston|dr\.?\s*seuss|father|sister|rev|s\.?\s*j|"
    r"comp|compiler|translator|auteur)\b|;|/|&",
    re.I,
)
GOOD_TITLE_RE = re.compile(r"\b(poems|poetry|lyrics|sonnets|songs|ballads|lays|verses|verse)\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: object) -> str:
    return "" if value is None else str(value).strip()


def decision(row: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    title = norm(row.get("source_work_title"))
    author = norm(row.get("author_name"))
    year = row.get("source_publication_year")
    language = " ".join(norm(x).lower() for x in row.get("ia_language", []))
    if not author:
        reasons.append("missing_author")
    if BAD_AUTHOR_RE.search(author):
        reasons.append("author_editor_or_ambiguous")
    if not title:
        reasons.append("missing_title")
    if BAD_TITLE_RE.search(title):
        reasons.append("bad_title_pattern")
    if not GOOD_TITLE_RE.search(title):
        reasons.append("title_not_poetry_book_like")
    if year is None:
        reasons.append("missing_year")
    elif int(year) > 1929:
        reasons.append("post_1929")
    if "handwritten" in language:
        reasons.append("handwritten")
    return ("keep" if not reasons else "reject", reasons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest_jsonl)
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        dec, reasons = decision(row)
        out = dict(row)
        out["filter_decision"] = dec
        out["filter_reasons"] = reasons
        if dec == "keep":
            kept.append(out)
        else:
            rejected.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
        encoding="utf-8",
    )
    if args.rejected:
        args.rejected.parent.mkdir(parents=True, exist_ok=True)
        args.rejected.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejected),
            encoding="utf-8",
        )
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "output": str(args.output),
        "rejected": str(args.rejected) if args.rejected else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
