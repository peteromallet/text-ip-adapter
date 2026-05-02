#!/usr/bin/env python3
"""Build v4.5 from v4.4 by fixing structural duplication artifacts.

v4.4 is the manually cleaned poetry dataset. Its text is much cleaner, but the
splits still overrepresent a small number of reference poems: many rows share
the same reference and differ only in target. This builder keeps the manual
edits/deletes, then applies deterministic caps and target deduplication so the
next run tests style conditioning rather than repeated-reference memorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path("data/pairs_v4_4_poetry_manual")
DEFAULT_OUTPUT = Path("data/pairs_v4_5_poetry_structural_balanced")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def ref_key(row: dict) -> str:
    return str(row.get("ref_doc_id") or text_key(row.get("ref_text", "")))


def target_key(row: dict) -> str:
    # target_doc_id can name a source document rather than a specific poem
    # chunk. Use exact normalized text so we only remove true repeated targets.
    return text_key(row.get("target_text", ""))


def round_robin_select(
    rows: list[dict],
    *,
    ref_cap: int,
    author_cap: int | None,
    dedupe_targets: bool,
) -> tuple[list[dict], dict[str, object]]:
    """Select rows deterministically while preserving author coverage.

    The source order is already deterministic, but a plain first-N filter would
    let high-count authors consume too much of the retained split. Round-robin
    by author keeps smaller authors represented before applying caps.
    """
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author"])].append(row)

    selected: list[dict] = []
    ref_counts: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    seen_targets: set[str] = set()
    drop_counts: Counter[str] = Counter()
    max_author_rows = max((len(author_rows) for author_rows in by_author.values()), default=0)

    for offset in range(max_author_rows):
        for author in sorted(by_author):
            author_rows = by_author[author]
            if offset >= len(author_rows):
                continue
            row = author_rows[offset]
            this_ref_key = ref_key(row)
            this_target_key = target_key(row)
            if author_cap is not None and author_counts[author] >= author_cap:
                drop_counts["author_cap"] += 1
                continue
            if ref_counts[this_ref_key] >= ref_cap:
                drop_counts["ref_cap"] += 1
                continue
            if dedupe_targets and this_target_key in seen_targets:
                drop_counts["target_duplicate"] += 1
                continue
            out = dict(row)
            out["structural_balance"] = {
                "source_dataset": "pairs_v4_4_poetry_manual",
                "ref_cap": ref_cap,
                "author_cap": author_cap,
                "dedupe_targets": dedupe_targets,
            }
            selected.append(out)
            ref_counts[this_ref_key] += 1
            author_counts[author] += 1
            seen_targets.add(this_target_key)

    audit = {
        "input_rows": len(rows),
        "output_rows": len(selected),
        "dropped": dict(sorted(drop_counts.items())),
        "authors": dict(sorted(author_counts.items())),
        "unique_refs": len(ref_counts),
        "max_ref_repeat": max(ref_counts.values(), default=0),
        "unique_targets": len(seen_targets),
    }
    return selected, audit


def make_probes(rows_by_author: dict[str, list[dict]], *, n_per_author: int = 2) -> list[dict]:
    authors = sorted(author for author, rows in rows_by_author.items() if len(rows) >= n_per_author)
    probes: list[dict] = []
    for author_index, author in enumerate(authors):
        swap_author = authors[(author_index + 1) % len(authors)]
        for row_index, row in enumerate(rows_by_author[author][:n_per_author]):
            swap = rows_by_author[swap_author][row_index % len(rows_by_author[swap_author])]
            probes.append(
                {
                    "probe_id": f"v45_poetry_{author}_{row_index:02d}",
                    "register": "poetry",
                    "author": author,
                    "heldout_split": row.get("split"),
                    "instruction": row["instruction"],
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


def duplicate_stats(rows: list[dict]) -> dict[str, object]:
    ref_counts = Counter(ref_key(row) for row in rows)
    target_counts = Counter(target_key(row) for row in rows)
    author_counts = Counter(str(row["author"]) for row in rows)
    return {
        "rows": len(rows),
        "authors": dict(sorted(author_counts.items())),
        "unique_refs": len(ref_counts),
        "max_ref_repeat": max(ref_counts.values(), default=0),
        "refs_gt_30": sum(count > 30 for count in ref_counts.values()),
        "unique_targets": len(target_counts),
        "target_duplicate_groups": sum(count > 1 for count in target_counts.values()),
        "max_target_repeat": max(target_counts.values(), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-ref-cap", type=int, default=40)
    parser.add_argument("--train-author-cap", type=int, default=60)
    parser.add_argument("--heldout-ref-cap", type=int, default=30)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(source.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "description": "manual-clean poetry data with target dedupe, repeated-reference caps, and rebuilt balanced probes",
        "caps": {
            "train_ref_cap": args.train_ref_cap,
            "train_author_cap": args.train_author_cap,
            "heldout_ref_cap": args.heldout_ref_cap,
            "dedupe_targets": True,
        },
        "splits": {},
    }

    split_rows: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(source / f"{split}.jsonl")
        selected, audit = round_robin_select(
            rows,
            ref_cap=args.train_ref_cap if split == "train" else args.heldout_ref_cap,
            author_cap=args.train_author_cap if split == "train" else None,
            dedupe_targets=True,
        )
        split_rows[split] = selected
        out_path = output / f"{split}.jsonl"
        write_jsonl(out_path, selected)
        manifest["splits"][split] = {
            **audit,
            "post_stats": duplicate_stats(selected),
            "sha256": sha256_path(out_path),
        }

    heldout_by_author: dict[str, list[dict]] = defaultdict(list)
    for split in ("val", "test"):
        for row in split_rows[split]:
            heldout_by_author[str(row["author"])].append(row)
    probes = make_probes(heldout_by_author)
    write_jsonl(output / "probes_balanced.jsonl", probes)
    write_jsonl(output / "probes_balanced_n16.jsonl", probes[:16])
    manifest["probes_balanced.jsonl"] = {
        "rows": len(probes),
        "authors": dict(sorted(Counter(row["author"] for row in probes).items())),
        "swap_authors": dict(sorted(Counter(row["swap_reference_author"] for row in probes).items())),
        "sha256": sha256_path(output / "probes_balanced.jsonl"),
    }
    manifest["probes_balanced_n16.jsonl"] = {
        "rows": len(probes[:16]),
        "authors": dict(sorted(Counter(row["author"] for row in probes[:16]).items())),
        "swap_authors": dict(sorted(Counter(row["swap_reference_author"] for row in probes[:16]).items())),
        "sha256": sha256_path(output / "probes_balanced_n16.jsonl"),
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
