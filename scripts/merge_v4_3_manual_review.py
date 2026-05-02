#!/usr/bin/env python3
"""Validate manual review files and build v4.4 poetry manual-clean data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DATA_ROOT = Path("text-ip-adapter/data/pairs_v4_3_poetry_strict")
REVIEW_ROOT = Path("prompt-adapters/experiments/2026-05-text-025-poetry-specific-style-axis/manual_review")
OUT_ROOT = Path("text-ip-adapter/data/pairs_v4_4_poetry_manual")

VALID_DECISIONS = {"keep", "delete", "edit"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_reviews() -> tuple[dict[tuple[str, int], dict], list[str], Counter]:
    reviews: dict[tuple[str, int], dict] = {}
    errors: list[str] = []
    counts: Counter = Counter()
    for path in sorted((REVIEW_ROOT / "reviews").glob("*.review.jsonl")):
        for line_no, row in enumerate(read_jsonl(path), start=1):
            split = row.get("split")
            row_index = row.get("row_index")
            decision = row.get("decision")
            confidence = row.get("confidence")
            key = (split, row_index)
            if not isinstance(split, str) or not isinstance(row_index, int):
                errors.append(f"{path}:{line_no}: missing/invalid split or row_index")
                continue
            if decision not in VALID_DECISIONS:
                errors.append(f"{path}:{line_no}: invalid decision {decision!r}")
            if confidence not in VALID_CONFIDENCE:
                errors.append(f"{path}:{line_no}: invalid confidence {confidence!r}")
            if key in reviews:
                errors.append(f"{path}:{line_no}: duplicate review key {key}")
            reviews[key] = row
            counts[(split, decision, row.get("reason", ""))] += 1
    return reviews, errors, counts


def apply_review(row: dict, review: dict) -> dict | None:
    decision = review["decision"]
    if decision == "delete":
        return None
    if decision == "keep":
        return row
    out = dict(row)
    for key in (
        "ref_text",
        "target_text",
        "reference_text",
        "expected_target",
        "swap_reference_text",
    ):
        clean_key = f"{key}_clean"
        if clean_key in review and review[clean_key] is not None:
            out[key] = review[clean_key]
    out.setdefault("manual_review", {})
    out["manual_review"] = {
        "decision": decision,
        "reason": review.get("reason"),
        "confidence": review.get("confidence"),
        "notes": review.get("notes", ""),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", default=str(OUT_ROOT))
    args = parser.parse_args()

    output = Path(args.output)
    reviews, errors, counts = load_reviews()
    manifest = json.loads((REVIEW_ROOT / "manifest.json").read_text(encoding="utf-8"))

    expected: set[tuple[str, int]] = set()
    for shard in manifest["shards"].values():
        for row in read_jsonl(Path(shard["path"])):
            expected.add((row["split"], row["row_index"]))
    missing = sorted(expected - set(reviews))
    unexpected = sorted(set(reviews) - expected)
    if missing and not args.allow_incomplete:
        errors.append(f"missing {len(missing)} expected reviews")
    if unexpected:
        errors.append(f"unexpected review keys: {unexpected[:10]}")
    if errors:
        print(json.dumps({"status": "invalid", "errors": errors[:50], "error_count": len(errors)}, indent=2))
        return 1

    deleted_rows: list[dict] = []
    edited_rows: list[dict] = []
    split_outputs: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        out_rows = []
        for row_index, row in enumerate(read_jsonl(DATA_ROOT / f"{split}.jsonl")):
            review = reviews.get((split, row_index))
            if review is None:
                if args.allow_incomplete:
                    out_rows.append(row)
                    continue
                raise AssertionError("unreachable missing review")
            applied = apply_review(row, review)
            if applied is None:
                deleted_rows.append({"split": split, "row_index": row_index, "review": review, "row": row})
                continue
            if review["decision"] == "edit":
                edited_rows.append({"split": split, "row_index": row_index, "review": review})
            out_rows.append(applied)
        split_outputs[split] = out_rows
        write_jsonl(output / f"{split}.jsonl", out_rows)

    probes = read_jsonl(DATA_ROOT / "probes_balanced_n16.jsonl")
    out_probes = []
    for row_index, row in enumerate(probes):
        review = reviews.get(("probes_balanced_n16", row_index))
        if review is None:
            if args.allow_incomplete:
                out_probes.append(row)
                continue
            raise AssertionError("unreachable missing probe review")
        applied = apply_review(row, review)
        if applied is None:
            deleted_rows.append({"split": "probes_balanced_n16", "row_index": row_index, "review": review, "row": row})
            continue
        if review["decision"] == "edit":
            edited_rows.append({"split": "probes_balanced_n16", "row_index": row_index, "review": review})
        out_probes.append(applied)
    write_jsonl(output / "probes_balanced_n16.jsonl", out_probes)
    write_jsonl(output / "probes_balanced.jsonl", out_probes)

    write_jsonl(output / "deleted_rows.jsonl", deleted_rows)
    write_jsonl(output / "edited_rows.jsonl", edited_rows)

    summary = {
        "source": str(DATA_ROOT),
        "output": str(output),
        "review_count": len(reviews),
        "missing_count": len(missing),
        "counts": {str(k): v for k, v in sorted(counts.items())},
        "rows": {split: len(rows) for split, rows in split_outputs.items()},
        "probes": len(out_probes),
        "deleted_rows": len(deleted_rows),
        "edited_rows": len(edited_rows),
    }
    (output / "manual_review_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
