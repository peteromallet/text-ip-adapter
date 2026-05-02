#!/usr/bin/env python3
"""Build a poetry+screenplay core2 dataset from v3.5.

Speech is intentionally excluded because 019 showed speech own/swap pathway
collapse across every checkpoint, while poetry+screenplay remain informative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_balanced_probes(splits: dict[str, list[dict[str, Any]]], registers: list[str], n_per_register: int) -> list[dict[str, Any]]:
    heldout: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for row in splits[split_name]:
            item = dict(row)
            item["_heldout_split"] = split_name
            heldout.append(item)

    by_reg_author: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in heldout:
        by_reg_author[row.get("register", "unknown")][row.get("author", "unknown")].append(row)

    probes: list[dict[str, Any]] = []
    for register in registers:
        authors = sorted(by_reg_author[register], key=lambda author: (-len(by_reg_author[register][author]), author))
        if len(authors) < 2:
            continue
        for i in range(n_per_register):
            author = authors[i % len(authors)]
            own_rows = by_reg_author[register][author]
            own = own_rows[(i // len(authors)) % len(own_rows)]
            swap_author = authors[(i + 1) % len(authors)]
            if swap_author == author:
                swap_author = authors[(i + 2) % len(authors)]
            swap_rows = by_reg_author[register][swap_author]
            swap = swap_rows[(i // len(authors)) % len(swap_rows)]
            probes.append(
                {
                    "probe_id": f"core2_{register}_{i:02d}",
                    "author": own.get("author", "unknown"),
                    "register": register,
                    "reference_text": own["ref_text"],
                    "instruction": own["instruction"],
                    "expected_target": own["target_text"],
                    "heldout_split": own["_heldout_split"],
                    "swap_reference_text": swap["ref_text"],
                    "swap_reference_author": swap.get("author", "unknown"),
                    "swap_reference_register": swap.get("register", "unknown"),
                    "swap_heldout_split": swap["_heldout_split"],
                }
            )
    return probes


def audit(splits: dict[str, list[dict[str, Any]]], registers: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "author_disjoint": {}}
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split, rows in splits.items():
        by_register = Counter(row["register"] for row in rows)
        authors_by_register: dict[str, set[str]] = defaultdict(set)
        split_keys[split] = set()
        for row in rows:
            reg = row["register"]
            author = row.get("author", "unknown")
            authors_by_register[reg].add(author)
            split_keys[split].add((reg, author))
        out["splits"][split] = {
            "rows": len(rows),
            "by_register": {reg: by_register.get(reg, 0) for reg in registers},
            "authors_by_register": {reg: len(authors_by_register[reg]) for reg in registers},
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        out["author_disjoint"][f"{left}_{right}"] = not bool(split_keys[left] & split_keys[right])
    out["pass"] = all(out["author_disjoint"].values())
    return out


def sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v3_5_artifact_clean_core3")
    parser.add_argument("--output-dir", default="data/pairs_v3_6_core2_poetry_screenplay")
    parser.add_argument("--probes-per-register", type=int, default=10)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    registers = ["poetry", "screenplay"]
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in read_jsonl(input_dir / f"{split}.jsonl") if row.get("register") in registers]
        splits[split] = rows
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    probes = build_balanced_probes(splits, registers, n_per_register=args.probes_per_register)
    write_jsonl(output_dir / "probes_balanced_n20.jsonl", probes)

    core_paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "probes_balanced_n20.jsonl")]
    manifest = {
        "corpus_version": "v3.6_core2_poetry_screenplay",
        "derived_from": str(input_dir),
        "registers": registers,
        "audit": audit(splits, registers),
        "balanced_probes": {
            "rows": len(probes),
            "by_register": dict(Counter(probe["register"] for probe in probes)),
            "authors_by_register": {
                reg: len({probe["author"] for probe in probes if probe["register"] == reg}) for reg in registers
            },
        },
        "sha256_core": sha(core_paths),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
