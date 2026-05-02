#!/usr/bin/env python3
"""Build v4 core2 style-clean data from v3.9.

v3.9 repaired the eval probes, but left the training distribution mostly
unchanged: brittle topic-token instructions and some book/heading artifacts.
This builder keeps the same split/author structure, normalizes instructions to
register-level style requests, strips common source artifacts, drops rows that
remain too short or dirty, and copies the clean balanced probe set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


ROMAN_HEADING_RE = re.compile(r"^\s{0,4}[IVXLCM]{2,}\.?\s*$")
FOOTNOTE_BLOCK_RE = re.compile(r"\[\s*Footnote\s+\d+\s*:.*?\]\s*", re.IGNORECASE | re.DOTALL)
BRACKET_NOTE_RE = re.compile(r"\[(?:\d+|Illustration|Transcriber[^\]]*|Footnote[^\]]*)\]\s*", re.IGNORECASE)
GUTENBERG_LINE_RE = re.compile(
    r"(project gutenberg|gutenberg ebook|end of (?:the )?project gutenberg|transcriber|etext)",
    re.IGNORECASE,
)
TOC_LINE_RE = re.compile(r"^\s*(?:contents|index|list of illustrations|illustrations|preface)\s*$", re.IGNORECASE)
PAGE_LINE_RE = re.compile(r"^\s*(?:page|plate|image|caption|illustration)\b.*$", re.IGNORECASE)
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = FOOTNOTE_BLOCK_RE.sub("", text)
    text = BRACKET_NOTE_RE.sub("", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if GUTENBERG_LINE_RE.search(line):
            continue
        if TOC_LINE_RE.match(line):
            continue
        if PAGE_LINE_RE.match(line):
            continue
        if ROMAN_HEADING_RE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text


def instruction_for(register: str) -> str:
    if register == "screenplay":
        return "Write a short screenplay scene in the style of the reference passage."
    return "Write a short poem in the style of the reference passage."


def is_dirty(text: str) -> bool:
    if not text or len(text.split()) < 40:
        return True
    if GUTENBERG_LINE_RE.search(text):
        return True
    if re.search(r"\[\s*Footnote\b|\bFootnote\s+\d+\b", text, re.IGNORECASE):
        return True
    return False


def transform_row(row: dict) -> tuple[dict | None, str]:
    out = dict(row)
    out["ref_text"] = clean_text(out.get("ref_text", ""))
    out["target_text"] = clean_text(out.get("target_text", ""))
    if is_dirty(out["ref_text"]):
        return None, "dirty_ref"
    if is_dirty(out["target_text"]):
        return None, "dirty_target"
    out["instruction_original"] = out.get("instruction", "")
    out["instruction"] = instruction_for(out.get("register", ""))
    return out, "kept"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_probe(row: dict) -> dict:
    out = dict(row)
    out["instruction_original"] = out.get("instruction", "")
    out["instruction"] = instruction_for(out.get("register", ""))
    out["reference_text"] = clean_text(out.get("reference_text", ""))
    out["swap_reference_text"] = clean_text(out.get("swap_reference_text", ""))
    out["expected_target"] = clean_text(out.get("expected_target", ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/pairs_v3_9_core2_evalclean")
    parser.add_argument("--output", default="data/pairs_v4_core2_styleclean")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(source.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "description": "v3.9 with artifact-stripped text and generic register-level instructions",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        kept: list[dict] = []
        reasons: Counter[str] = Counter()
        before_registers: Counter[str] = Counter()
        after_registers: Counter[str] = Counter()
        for row in iter_jsonl(source / f"{split}.jsonl"):
            before_registers[row.get("register", "")] += 1
            transformed, reason = transform_row(row)
            reasons[reason] += 1
            if transformed is None:
                continue
            kept.append(transformed)
            after_registers[transformed.get("register", "")] += 1
        write_jsonl(output / f"{split}.jsonl", kept)
        manifest["splits"][split] = {
            "source_rows": sum(before_registers.values()),
            "kept_rows": len(kept),
            "before_registers": dict(before_registers),
            "after_registers": dict(after_registers),
            "drop_reasons": dict(reasons),
            "sha256": sha256_path(output / f"{split}.jsonl"),
        }

    for probe_name in ("probes_balanced.jsonl", "probes_balanced_n32.jsonl"):
        probes = [transform_probe(row) for row in iter_jsonl(source / probe_name)]
        write_jsonl(output / probe_name, probes)
        manifest[probe_name] = {
            "derived_from": str((source / probe_name).relative_to(repo_root)),
            "rows": len(probes),
            "instruction_policy": "generic register-level style request",
            "sha256": sha256_path(output / probe_name),
        }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
