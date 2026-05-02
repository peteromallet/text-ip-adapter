#!/usr/bin/env python3
"""Build v3.8 core2 with heldout authors not seen by the 006 warmstart train set."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_v3_7_core2_repair import audit, build_balanced_probes, read_jsonl, write_jsonl


HELDOUT_AUTHORS = {
    "val": {
        "poetry": {
            "edwin_arlington_robinson",
            "john_masefield",
            "matthew_arnold",
            "edna_millay",
        },
        "screenplay": {
            "bourne_ultimatum_the",
            "cellular",
            "chasing_amy",
            "birds_the",
        },
    },
    "test": {
        "poetry": {
            "sara_teasdale",
            "stephen_crane",
            "christina_rossetti",
            "emily_dickinson",
        },
        "screenplay": {
            "blood_simple",
            "asteroid_city",
            "50_50",
            "ad_astra",
        },
    },
}


def sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def authors_by_register(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        out[row.get("register", "unknown")].add(row.get("author", "unknown"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v3_7_core2_repaired")
    parser.add_argument("--output-dir", default="data/pairs_v3_8_core2_cleanheldout")
    parser.add_argument("--warmstart-train", default="data/pairs/train.llm.jsonl")
    parser.add_argument("--probes-per-register", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    registers = ["poetry", "screenplay"]

    warmstart_train_authors = authors_by_register(Path(args.warmstart_train))
    contamination: dict[str, dict[str, list[str]]] = {}
    for split, by_reg in HELDOUT_AUTHORS.items():
        contamination[split] = {}
        for reg, split_authors in by_reg.items():
            contamination[split][reg] = sorted(split_authors & warmstart_train_authors.get(reg, set()))
    if any(contamination[split][reg] for split in contamination for reg in contamination[split]):
        raise RuntimeError(f"heldout author overlaps 006 train set: {contamination}")

    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        all_rows.extend(read_jsonl(input_dir / f"{split}.jsonl"))
    all_rows = [row for row in all_rows if row.get("register") in registers]

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in all_rows:
        reg = row["register"]
        author = row.get("author", "unknown")
        if author in HELDOUT_AUTHORS["val"].get(reg, set()):
            splits["val"].append(row)
        elif author in HELDOUT_AUTHORS["test"].get(reg, set()):
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    probes = build_balanced_probes(splits, registers, n_per_register=args.probes_per_register)
    write_jsonl(output_dir / f"probes_balanced_n{len(probes)}.jsonl", probes)
    write_jsonl(output_dir / "probes_balanced.jsonl", probes)

    core_paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "probes_balanced.jsonl")]
    manifest = {
        "corpus_version": "v3.8_core2_cleanheldout",
        "derived_from": str(input_dir),
        "warmstart_train_exclusion": str(args.warmstart_train),
        "registers": registers,
        "heldout_authors": {split: {reg: sorted(authors) for reg, authors in by_reg.items()} for split, by_reg in HELDOUT_AUTHORS.items()},
        "warmstart_contamination": contamination,
        "audit": audit(splits, registers, probes),
        "sha256_core": sha(core_paths),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
