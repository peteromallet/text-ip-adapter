#!/usr/bin/env python3
"""Build a seed clean-poetry corpus from an existing pair dataset.

This is a bridge artifact, not the final broad corpus. It extracts unique
reference and target texts from a cleaned pair dataset so the corpus project has
an auditable starting point while source-native ingestors are built.
"""
from __future__ import annotations

import argparse
import hashlib
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


def normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def corpus_id(author_id: str, text: str) -> str:
    return f"{author_id}_{hashlib.sha1(normalized_text(text).encode('utf-8')).hexdigest()[:12]}"


def add_record(records: dict[str, dict], row: dict, *, role: str, split: str) -> None:
    text_key = "ref_text" if role == "reference" else "target_text"
    doc_key = "ref_doc_id" if role == "reference" else "target_doc_id"
    text = normalized_text(str(row.get(text_key, "")))
    if not text:
        return
    author_id = str(row["author"])
    clean_hash = sha256_text(text)
    record_key = f"{author_id}:{clean_hash}"
    if record_key not in records:
        source_name = str(row.get("source_dataset") or "unknown_from_pair_dataset")
        source_id = str(row.get(doc_key) or corpus_id(author_id, text))
        records[record_key] = {
            "corpus_id": corpus_id(author_id, text),
            "author_id": author_id,
            "author_name": author_id.replace("_", " ").title(),
            "title": None,
            "text": text,
            "source_id": source_id,
            "source_name": source_name,
            "source_url": f"pair_dataset:{source_id}",
            "source_work_title": None,
            "source_publication_year": None,
            "license": "public_domain_us",
            "public_domain_basis": "inherited_from_cleaned_pair_dataset_source_metadata",
            "raw_sha256": sha256_text(str(row.get(f"{text_key}_original") or text)),
            "clean_sha256": clean_hash,
            "cleaning_version": "seed_from_pairs_v1",
            "split_hint": split,
            "flags": ["seed_from_pair_dataset"],
            "notes": "Bridge corpus seed extracted from cleaned pair rows; replace with source-native provenance in poetry_corpus_v0.",
            "pair_roles_seen": [],
            "pair_splits_seen": [],
        }
    rec = records[record_key]
    if role not in rec["pair_roles_seen"]:
        rec["pair_roles_seen"].append(role)
    if split not in rec["pair_splits_seen"]:
        rec["pair_splits_seen"].append(split)
    if rec["split_hint"] != split:
        rec["split_hint"] = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default="data/pairs_v4_5_poetry_structural_balanced")
    parser.add_argument("--output", default="data/poetry_corpus_v0_seed_from_v45")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    pairs_dir = (repo_root / args.pairs_dir).resolve()
    output = (repo_root / args.output).resolve()

    records: dict[str, dict] = {}
    input_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(pairs_dir / f"{split}.jsonl")
        input_counts[split] = len(rows)
        for row in rows:
            add_record(records, row, role="reference", split=split)
            add_record(records, row, role="target", split=split)

    out_rows = sorted(records.values(), key=lambda row: (row["author_id"], row["corpus_id"]))
    write_jsonl(output / "corpus.jsonl", out_rows)

    author_counts = Counter(row["author_id"] for row in out_rows)
    manifest = {
        "description": "Seed corpus extracted from v4.5 cleaned/balanced pair rows. Use for bootstrapping audits only.",
        "source_pairs_dir": str(pairs_dir.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "input_pair_rows": input_counts,
        "corpus_rows": len(out_rows),
        "authors": dict(sorted(author_counts.items())),
        "cleaning_version": "seed_from_pairs_v1",
        "limitations": [
            "not source-native",
            "titles often unavailable",
            "source_url is pair-dataset pseudo-provenance",
            "should be replaced by poetry_corpus_v0 from raw source manifests",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
