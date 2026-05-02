#!/usr/bin/env python3
"""Build a poetry-only v4.2 dataset from v4.1.

024c made screenplay look pathway-solved but left poetry weak. This dataset
removes screenplay so every train batch, triplet positive, and triplet negative
is poetry. Probes are the v4.1 generic-instruction poetry probes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/pairs_v4_1_core2_styleclean_genericprobes")
    parser.add_argument("--output", default="data/pairs_v4_2_poetry_styleclean")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(source.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "description": "poetry-only slice of v4.1 generic-probe style-clean data",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        rows = [row for row in iter_jsonl(source / f"{split}.jsonl") if row.get("register") == "poetry"]
        write_jsonl(output / f"{split}.jsonl", rows)
        authors: dict[str, int] = {}
        for row in rows:
            authors[row["author"]] = authors.get(row["author"], 0) + 1
        manifest["splits"][split] = {
            "rows": len(rows),
            "authors": authors,
            "sha256": sha256_path(output / f"{split}.jsonl"),
        }

    probes = [row for row in iter_jsonl(source / "probes_balanced.jsonl") if row.get("register") == "poetry"]
    write_jsonl(output / "probes_balanced.jsonl", probes)
    write_jsonl(output / "probes_balanced_n16.jsonl", probes)
    manifest["probes_balanced.jsonl"] = {
        "rows": len(probes),
        "sha256": sha256_path(output / "probes_balanced.jsonl"),
    }
    manifest["probes_balanced_n16.jsonl"] = {
        "rows": len(probes),
        "sha256": sha256_path(output / "probes_balanced_n16.jsonl"),
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
