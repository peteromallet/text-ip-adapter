#!/usr/bin/env python3
"""Create deterministic review shards for v4.3 poetry manual cleanup."""
from __future__ import annotations

import json
from pathlib import Path


DATA_ROOT = Path("text-ip-adapter/data/pairs_v4_3_poetry_strict")
OUT_ROOT = Path("prompt-adapters/experiments/2026-05-text-025-poetry-specific-style-axis/manual_review")

TRAIN_SHARDS = {
    "shard_01_train_swinburne_lowell_drossetti": ["algernon_swinburne", "amy_lowell", "dante_rossetti"],
    "shard_02_train_poe_ebrowning_hopkins": ["edgar_allan_poe", "elizabeth_browning", "gerard_manley_hopkins"],
    "shard_03_train_longfellow_keats": ["henry_longfellow", "john_keats"],
    "shard_04_train_wilde_frost": ["oscar_wilde", "robert_frost"],
    "shard_05_train_kipling_brooke_hardy": ["rudyard_kipling", "rupert_brooke", "thomas_hardy"],
    "shard_06_train_eliot_whitman": ["t_s_eliot", "walt_whitman"],
    "shard_07_train_owen_blake": ["wilfred_owen", "william_blake"],
    "shard_08_train_yeats_wordsworth": ["william_butler_yeats", "william_wordsworth"],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def indexed_rows(split: str) -> list[dict]:
    rows = read_jsonl(DATA_ROOT / f"{split}.jsonl")
    out = []
    for row_index, row in enumerate(rows):
        item = dict(row)
        item["split"] = split
        item["row_index"] = row_index
        out.append(item)
    return out


def main() -> int:
    shards_dir = OUT_ROOT / "shards"
    reviews_dir = OUT_ROOT / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    train_rows = indexed_rows("train")
    manifest: dict[str, object] = {"data_root": str(DATA_ROOT), "shards": {}}

    for shard_id, authors in TRAIN_SHARDS.items():
        rows = [row for row in train_rows if row.get("author") in authors]
        write_jsonl(shards_dir / f"{shard_id}.jsonl", rows)
        manifest["shards"][shard_id] = {
            "path": str(shards_dir / f"{shard_id}.jsonl"),
            "review_path": str(reviews_dir / f"{shard_id}.review.jsonl"),
            "rows": len(rows),
            "authors": authors,
        }

    val_rows = indexed_rows("val")
    write_jsonl(shards_dir / "shard_09_val_all.jsonl", val_rows)
    manifest["shards"]["shard_09_val_all"] = {
        "path": str(shards_dir / "shard_09_val_all.jsonl"),
        "review_path": str(reviews_dir / "shard_09_val_all.review.jsonl"),
        "rows": len(val_rows),
        "authors": sorted({row["author"] for row in val_rows}),
    }

    test_rows = indexed_rows("test")
    probes = read_jsonl(DATA_ROOT / "probes_balanced_n16.jsonl")
    probe_rows = []
    for row_index, row in enumerate(probes):
        item = dict(row)
        item["split"] = "probes_balanced_n16"
        item["row_index"] = row_index
        probe_rows.append(item)
    test_probe_rows = test_rows + probe_rows
    write_jsonl(shards_dir / "shard_10_test_and_probes.jsonl", test_probe_rows)
    manifest["shards"]["shard_10_test_and_probes"] = {
        "path": str(shards_dir / "shard_10_test_and_probes.jsonl"),
        "review_path": str(reviews_dir / "shard_10_test_and_probes.review.jsonl"),
        "rows": len(test_probe_rows),
        "authors": sorted({row["author"] for row in test_rows}),
        "probe_rows": len(probe_rows),
    }

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
