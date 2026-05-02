#!/usr/bin/env python3
"""Build a stricter poetry-only dataset from v4.1.

The first poetry-only slice (v4.2) preserved many title/section headers and a
small amount of dramatic verse. For 025, we want the adapter objective to see a
cleaner verse style axis: generic instructions, poetry only, obvious headings
removed, and stage/dialogue-like material excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median
from typing import Iterable


GENERIC_POETRY_INSTRUCTION = "Write a short poem in the style of the reference passage."

HEADING_RE = re.compile(r"^([A-Z][A-Z0-9 .,'’!?:;()_-]{0,48}|[IVXLCDM]+\.?|\d+\.?|[_*].{1,48}[_*])$")
STAGE_RE = re.compile(
    r"\b(BURR|HAMILTON|WASHINGTON|JEFFERSON|ADAMS|MADISON|SCENE|ENTER|EXIT|EXEUNT|ACT\s+[IVX]+)\b"
)
WORD_RE = re.compile(r"[A-Za-z']+")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def strip_heading_lines(text: str) -> tuple[str, int]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stripped = 0
    out: list[str] = []
    seen_body = False
    for raw in lines:
        line = raw.strip()
        if not seen_body and not line:
            continue
        if (
            not seen_body
            and stripped < 6
            and line
            and HEADING_RE.match(line)
            and line_word_count(line) <= 8
        ):
            stripped += 1
            continue
        seen_body = True
        out.append(raw)
    return "\n".join(out).strip(), stripped


def verse_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = WORD_RE.findall(text)
    if len(words) < 32 or len(lines) < 6:
        return False
    line_words = [line_word_count(line) for line in lines]
    # Keep long-line poets, but drop passages that look like prose paragraphs.
    if median(line_words) > 18:
        return False
    if sum(1 for n in line_words if n > 30) >= 3:
        return False
    return True


def clean_text(text: str) -> tuple[str, int]:
    text, stripped = strip_heading_lines(text)
    return text, stripped


def clean_pair(row: dict) -> tuple[dict | None, dict[str, object]]:
    if row.get("register") != "poetry":
        return None, {"reason": "not_poetry"}
    ref_text, ref_stripped = clean_text(row.get("ref_text", ""))
    target_text, target_stripped = clean_text(row.get("target_text", ""))
    combined = f"{ref_text}\n{target_text}"
    if STAGE_RE.search(combined):
        return None, {"reason": "stage_like"}
    if not verse_like(ref_text) or not verse_like(target_text):
        return None, {"reason": "not_verse_like"}
    out = dict(row)
    out["ref_text_original"] = row.get("ref_text", "")
    out["target_text_original"] = row.get("target_text", "")
    out["ref_text"] = ref_text
    out["target_text"] = target_text
    out["instruction"] = GENERIC_POETRY_INSTRUCTION
    out["heading_lines_stripped"] = ref_stripped + target_stripped
    return out, {"reason": "kept", "heading_lines_stripped": ref_stripped + target_stripped}


def make_probes(rows_by_author: dict[str, list[dict]], *, n_per_author: int = 2) -> list[dict]:
    authors = sorted(rows_by_author)
    probes: list[dict] = []
    for author_index, author in enumerate(authors):
        rows = rows_by_author[author][:n_per_author]
        swap_author = authors[(author_index + 1) % len(authors)]
        swap_rows = rows_by_author[swap_author]
        for row_index, row in enumerate(rows):
            swap = swap_rows[row_index % len(swap_rows)]
            probes.append(
                {
                    "probe_id": f"v43_poetry_{author}_{row_index:02d}",
                    "register": "poetry",
                    "author": author,
                    "heldout_split": row.get("split"),
                    "instruction": GENERIC_POETRY_INSTRUCTION,
                    "reference_text": row["ref_text"],
                    "expected_target": row["target_text"],
                    "ref_doc_id": row.get("ref_doc_id"),
                    "target_doc_id": row.get("target_doc_id"),
                    "swap_reference_author": swap_author,
                    "swap_reference_register": "poetry",
                    "swap_heldout_split": swap.get("split"),
                    "swap_reference_text": swap["ref_text"],
                    "swap_ref_doc_id": swap.get("ref_doc_id"),
                }
            )
    return probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/pairs_v4_1_core2_styleclean_genericprobes")
    parser.add_argument("--output", default="data/pairs_v4_3_poetry_strict")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(source.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "description": "strict poetry-only v4.3 with heading stripping and stage/prose filtering",
        "splits": {},
    }
    kept_by_split: dict[str, list[dict]] = {}
    audit: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        kept: list[dict] = []
        counts: dict[str, int] = {}
        heading_stripped_rows = 0
        for row in iter_jsonl(source / f"{split}.jsonl"):
            cleaned, info = clean_pair(row)
            reason = str(info["reason"])
            counts[reason] = counts.get(reason, 0) + 1
            if cleaned is None:
                continue
            cleaned["split"] = split
            if int(cleaned.get("heading_lines_stripped", 0)) > 0:
                heading_stripped_rows += 1
            kept.append(cleaned)
        write_jsonl(output / f"{split}.jsonl", kept)
        kept_by_split[split] = kept
        authors: dict[str, int] = {}
        for row in kept:
            authors[row["author"]] = authors.get(row["author"], 0) + 1
        audit[split] = counts
        manifest["splits"][split] = {
            "rows": len(kept),
            "authors": authors,
            "audit": counts,
            "heading_stripped_rows": heading_stripped_rows,
            "sha256": sha256_path(output / f"{split}.jsonl"),
        }

    heldout_by_author: dict[str, list[dict]] = {}
    for split in ("val", "test"):
        for row in kept_by_split[split]:
            heldout_by_author.setdefault(row["author"], []).append(row)
    probes = make_probes({a: rows for a, rows in heldout_by_author.items() if len(rows) >= 2})
    write_jsonl(output / "probes_balanced.jsonl", probes)
    write_jsonl(output / "probes_balanced_n16.jsonl", probes[:16])
    manifest["probes_balanced.jsonl"] = {
        "rows": len(probes),
        "sha256": sha256_path(output / "probes_balanced.jsonl"),
    }
    manifest["probes_balanced_n16.jsonl"] = {
        "rows": min(16, len(probes)),
        "sha256": sha256_path(output / "probes_balanced_n16.jsonl"),
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
