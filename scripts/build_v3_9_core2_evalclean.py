#!/usr/bin/env python3
"""Build v3.9 core2 eval-clean probes from v3.8.

v3.8 fixed warmstart author contamination but still allowed dirty reference
documents in probes, notably a Christina Rossetti footnote/Dante block reused
across poetry probes. v3.9 keeps the train/val/test split unchanged and rebuilds
only the balanced probe file, preferring clean heldout target chunks as style
references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_v3_7_core2_repair import read_jsonl, write_jsonl


REGISTERS = ["poetry", "screenplay"]


DIRTY_RE = re.compile(
    r"\[(?:Footnote|Illustration|Transcriber|Note)\b|"
    r"\b(?:CHAPTER|BOOK|ACT|SCENE)\b|"
    r"\b(?:INTEXT QUESTIONS|Exercises?)\b",
    re.I,
)


def clean_enough(text: str, register: str) -> bool:
    if not text or DIRTY_RE.search(text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(re.findall(r"[A-Za-z']+", text)) < 30:
        return False
    if register == "poetry":
        if len(lines) < 4:
            return False
        avg_line = sum(len(line) for line in lines) / max(1, len(lines))
        if avg_line > 65:
            return False
        if max((len(line) for line in lines), default=0) > 130:
            return False
    return True


def sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def build_doc_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pool: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        register = row.get("register", "unknown")
        author = row.get("author", "unknown")
        for field, doc_key in (("target_text", "target_doc_id"), ("ref_text", "ref_doc_id")):
            text = row.get(field, "")
            doc_id = row.get(doc_key)
            if not doc_id or not clean_enough(text, register):
                continue
            pool[author][doc_id] = {
                "doc_id": doc_id,
                "text": text,
                "source_field": field,
                "register": register,
                "heldout_split": row.get("_heldout_split"),
            }
    return pool


def rows_with_split(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        for row in rows:
            row["_heldout_split"] = split
        splits[split] = rows
    return splits


def build_evalclean_probes(splits: dict[str, list[dict[str, Any]]], n_per_register: int) -> list[dict[str, Any]]:
    heldout = [row for split in ("val", "test") for row in splits[split] if row.get("register") in REGISTERS]
    by_register_author: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in heldout:
        by_register_author[row["register"]][row["author"]].append(row)
    doc_pool = build_doc_pool(heldout)

    probes: list[dict[str, Any]] = []
    for register in REGISTERS:
        authors = sorted(by_register_author[register], key=lambda author: (-len(by_register_author[register][author]), author))
        if len(authors) < 2:
            continue
        author_docs = {
            author: sorted(
                [doc for doc in doc_pool.get(author, {}).values() if doc["register"] == register],
                key=lambda doc: doc["doc_id"],
            )
            for author in authors
        }
        missing = [author for author, docs in author_docs.items() if not docs]
        if missing:
            raise RuntimeError(f"no clean docs for {register}: {missing}")
        for i in range(n_per_register):
            author = authors[i % len(authors)]
            own_rows = by_register_author[register][author]
            own = own_rows[(i // len(authors)) % len(own_rows)]
            own_docs = author_docs[author]
            own_doc = own_docs[(i // len(authors)) % len(own_docs)]
            if len(own_docs) > 1 and own_doc["doc_id"] == own.get("target_doc_id"):
                own_doc = own_docs[((i // len(authors)) + 1) % len(own_docs)]

            swap_author = authors[(i + 1) % len(authors)]
            swap_docs = author_docs[swap_author]
            swap_doc = swap_docs[(i // len(authors)) % len(swap_docs)]

            probes.append(
                {
                    "probe_id": f"core2v39_{register}_{i:02d}",
                    "author": own.get("author", "unknown"),
                    "register": register,
                    "reference_text": own_doc["text"],
                    "instruction": own["instruction"],
                    "expected_target": own["target_text"],
                    "heldout_split": own["_heldout_split"],
                    "ref_doc_id": own_doc["doc_id"],
                    "ref_source_field": own_doc["source_field"],
                    "target_doc_id": own.get("target_doc_id"),
                    "swap_reference_text": swap_doc["text"],
                    "swap_reference_author": swap_author,
                    "swap_reference_register": register,
                    "swap_heldout_split": by_register_author[register][swap_author][0]["_heldout_split"],
                    "swap_ref_doc_id": swap_doc["doc_id"],
                    "swap_ref_source_field": swap_doc["source_field"],
                }
            )
    return probes


def audit(probes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(probes),
        "by_register": dict(Counter(probe["register"] for probe in probes)),
        "authors_by_register": {
            reg: len({probe["author"] for probe in probes if probe["register"] == reg}) for reg in REGISTERS
        },
        "unique_refs_by_register": {
            reg: len({probe["ref_doc_id"] for probe in probes if probe["register"] == reg}) for reg in REGISTERS
        },
        "dirty_refs": sum(
            1
            for probe in probes
            for field in ("reference_text", "swap_reference_text")
            if not clean_enough(probe[field], probe["register"])
        ),
        "source_fields": dict(Counter(probe["ref_source_field"] for probe in probes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v3_8_core2_cleanheldout")
    parser.add_argument("--output-dir", default="data/pairs_v3_9_core2_evalclean")
    parser.add_argument("--probes-per-register", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = rows_with_split(input_dir)
    for split in ("train", "val", "test"):
        rows = [dict(row) for row in splits[split]]
        for row in rows:
            row.pop("_heldout_split", None)
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    probes = build_evalclean_probes(splits, args.probes_per_register)
    write_jsonl(output_dir / f"probes_balanced_n{len(probes)}.jsonl", probes)
    write_jsonl(output_dir / "probes_balanced.jsonl", probes)

    for name in ("manifest.json",):
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / f"v3_8_{name}")

    core_paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "probes_balanced.jsonl")]
    manifest = {
        "corpus_version": "v3.9_core2_evalclean",
        "derived_from": str(input_dir),
        "change": "train/val/test unchanged; balanced probes rebuilt with clean heldout target/ref chunks as references",
        "audit": audit(probes),
        "sha256_core": sha(core_paths),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
