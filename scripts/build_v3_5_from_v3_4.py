#!/usr/bin/env python3
"""Build v3.5 by applying stricter artifact cleanup to v3.4 splits.

This is intentionally a narrow repair over v3.4. It preserves the author-disjoint
split shape while removing artifact blocks that v3.4 missed, especially wrapped
Gutenberg picture/note captions and standalone page/year fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BAD_SUBSTRINGS = [
    "[picture:",
    "[illustration:",
    "[note on text:",
    "lines longer than 78 characters",
    "this etext has been transcribed",
    "voice of the page",
    "a new illustrated edition",
    "the world and its people",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_wrapped_bracket_blocks(lines: list[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        starts_block = lower.startswith(("[picture:", "[illustration:", "[note on text:"))
        if in_block or starts_block:
            removed += 1
            if "]" in stripped:
                in_block = False
            else:
                in_block = True
            continue
        kept.append(line)
    return kept, removed


def _clean_text(text: str, register: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    lines = text.splitlines()
    lines, n_block = _clean_wrapped_bracket_blocks(lines)
    if n_block:
        counts["wrapped_bracket_block"] += n_block

    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        remove_reason = ""
        if register == "poetry":
            if re.fullmatch(r"\d{1,4}\.?", stripped):
                remove_reason = "poetry_page_number"
            elif re.fullmatch(r"\d{3,4}\s*[–-]\.?", stripped):
                remove_reason = "poetry_year_fragment"
            elif lower.startswith(("page ", "note on text", "notes.")):
                remove_reason = "poetry_apparatus_line"
        elif register == "screenplay":
            if re.fullmatch(r"\d{1,4}\.?", stripped):
                remove_reason = "screenplay_page_number"
            elif re.fullmatch(r"\(?continued\)?(?:[:.]| to next page)?|cont'd\.?", lower):
                remove_reason = "screenplay_continuation"
        if remove_reason:
            counts[remove_reason] += 1
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, counts


def _bad_after_clean(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in ("ref_text", "target_text"):
        lower = str(row.get(field, "")).lower()
        for marker in BAD_SUBSTRINGS:
            if marker in lower:
                reasons.append(f"{field}:{marker}")
    return reasons


def _audit_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "author_disjoint": {}}
    split_authors: dict[str, set[tuple[str, str]]] = {}
    for split, rows in splits.items():
        by_register = Counter(row.get("register", "unknown") for row in rows)
        authors_by_register: dict[str, set[str]] = defaultdict(set)
        split_authors[split] = set()
        for row in rows:
            reg = str(row.get("register", "unknown"))
            author = str(row.get("author", "unknown"))
            authors_by_register[reg].add(author)
            split_authors[split].add((reg, author))
        out["splits"][split] = {
            "rows": len(rows),
            "by_register": dict(by_register),
            "authors_by_register": {reg: len(authors) for reg, authors in authors_by_register.items()},
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out["author_disjoint"][f"{a}_{b}"] = not bool(split_authors[a] & split_authors[b])
    return out


def _sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v3_4_artifact_clean_core3")
    parser.add_argument("--output-dir", default="data/pairs_v3_5_artifact_clean_core3")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    cleanup_counts: Counter[str] = Counter()
    removed: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        rows = _read_jsonl(input_dir / f"{split}.jsonl")
        kept: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            register = str(row.get("register", "unknown"))
            for field in ("ref_text", "target_text"):
                cleaned, counts = _clean_text(str(row.get(field, "")), register)
                row[field] = cleaned
                for key, value in counts.items():
                    cleanup_counts[f"{split}:{register}:{field}:{key}"] += value
            reasons = _bad_after_clean(row)
            if reasons or not row.get("ref_text", "").strip() or not row.get("target_text", "").strip():
                removed[split].append(
                    {
                        "author": row.get("author"),
                        "register": register,
                        "ref_doc_id": row.get("ref_doc_id"),
                        "target_doc_id": row.get("target_doc_id"),
                        "reasons": reasons or ["empty_after_clean"],
                    }
                )
                continue
            kept.append(row)
        splits[split] = kept
        _write_jsonl(output_dir / f"{split}.jsonl", kept)

    for probes_name in ("probes_balanced_n15.jsonl", "probes_default_n20.jsonl"):
        src = input_dir / probes_name
        if src.exists():
            probes = _read_jsonl(src)
            repaired: list[dict[str, Any]] = []
            for probe in probes:
                probe = dict(probe)
                register = str(probe.get("register", "unknown"))
                for field in ("reference_text", "expected_target", "swap_reference_text"):
                    if field in probe:
                        probe[field], counts = _clean_text(str(probe[field]), register)
                        for key, value in counts.items():
                            cleanup_counts[f"probe:{register}:{field}:{key}"] += value
                probe_check = {
                    "ref_text": "\n".join(
                        [
                            str(probe.get("reference_text", "")),
                            str(probe.get("swap_reference_text", "")),
                        ]
                    ),
                    "target_text": str(probe.get("expected_target", "")),
                }
                if not _bad_after_clean(probe_check):
                    repaired.append(probe)
            _write_jsonl(output_dir / probes_name, repaired)

    core_paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    manifest = {
        "corpus_version": "v3.5_artifact_clean_core3",
        "derived_from": str(input_dir),
        "cleanup_counts": dict(cleanup_counts),
        "removed": {split: rows for split, rows in removed.items()},
        "removed_counts": {split: len(rows) for split, rows in removed.items()},
        "audit": _audit_splits(splits),
        "sha256_core": _sha(core_paths),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["audit"], indent=2))
    print(f"sha256_core={manifest['sha256_core']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
