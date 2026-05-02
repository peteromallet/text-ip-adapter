#!/usr/bin/env python3
"""Discover Internet Archive poetry source candidates.

This writes metadata candidates only. It does not download OCR text or accept
sources into the corpus. Internet Archive is large and noisy, so it must enter
through the source manifest gate.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY = (
    "mediatype:texts AND "
    "(title:poems OR title:poetry OR subject:poetry) AND "
    "(language:English OR language:eng) AND "
    "(rights:publicdomain OR licenseurl:*publicdomain*)"
)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def fetch_page(query: str, *, rows: int, page: int) -> dict:
    fields = ["identifier", "title", "creator", "date", "language", "subject", "rights", "licenseurl"]
    params: list[tuple[str, str | int]] = [
        ("q", query),
        ("rows", rows),
        ("page", page),
        ("output", "json"),
    ]
    for field in fields:
        params.append(("fl[]", field))
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "text-ip-adapter/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--max-year", type=int, default=1929)
    parser.add_argument("--require-year", action="store_true")
    parser.add_argument("--exclude-handwritten", action="store_true")
    parser.add_argument("--output", default="data/poetry_source_manifest_ia_publicdomain_candidates.jsonl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    page = 1
    total = None
    seen: set[str] = set()
    while len(out_rows) < args.limit:
        data = fetch_page(args.query, rows=min(args.rows, args.limit - len(out_rows)), page=page)
        response = data.get("response", {})
        total = response.get("numFound", total)
        docs = response.get("docs", [])
        if not docs:
            break
        for doc in docs:
            identifier = str(doc.get("identifier") or "")
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            creators = as_list(doc.get("creator"))
            title = str(doc.get("title") or "")
            year = None
            date_values = as_list(doc.get("date"))
            for date in date_values:
                if len(date) >= 4 and date[:4].isdigit():
                    year = int(date[:4])
                    break
            languages = as_list(doc.get("language"))
            if args.require_year and year is None:
                continue
            if args.max_year is not None and year is not None and year > args.max_year:
                continue
            if args.exclude_handwritten and any("handwritten" in lang.lower() for lang in languages):
                continue
            out_rows.append(
                {
                    "source_id": f"internet_archive_{identifier}",
                    "author_id": None,
                    "author_name": creators[0] if creators else None,
                    "source_name": "internet_archive",
                    "source_url": f"https://archive.org/details/{identifier}",
                    "source_work_title": title or None,
                    "source_publication_year": year,
                    "license": "public_domain_candidate",
                    "public_domain_basis": "internet_archive_rights_or_license_metadata_candidate",
                    "expected_register": "poetry",
                    "risk_notes": "Internet Archive metadata candidate; requires author normalization, OCR quality audit, rights verification, and poem-boundary parsing before acceptance.",
                    "status": "candidate",
                    "ia_identifier": identifier,
                    "ia_subject": as_list(doc.get("subject"))[:20],
                    "ia_language": languages,
                    "ia_rights": as_list(doc.get("rights")),
                    "ia_licenseurl": as_list(doc.get("licenseurl")),
                }
            )
            if len(out_rows) >= args.limit:
                break
        page += 1

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.relative_to(repo_root)), "rows": len(out_rows), "total_found": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
