#!/usr/bin/env python3
"""Derive capped author-disjoint poetry pairs from a clean corpus JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


GENERIC_INSTRUCTION = "Write a short poem in the style of the reference passage."


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


def stable_doc_sort(row: dict) -> tuple[str, str, str]:
    return (str(row.get("source_id") or ""), str(row.get("title") or ""), str(row.get("corpus_id") or ""))


def select_diverse_docs(rows: list[dict], cap: int) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=stable_doc_sort):
        by_source[str(row.get("source_id") or "")].append(row)
    selected: list[dict] = []
    max_len = max((len(source_rows) for source_rows in by_source.values()), default=0)
    for offset in range(max_len):
        for source_id in sorted(by_source):
            source_rows = by_source[source_id]
            if offset < len(source_rows):
                selected.append(source_rows[offset])
                if len(selected) >= cap:
                    return selected
    return selected


def make_author_pairs(author: str, rows: list[dict], *, max_pairs: int, split: str) -> list[dict]:
    docs = select_diverse_docs(rows, cap=max(max_pairs + 1, 2))
    if len(docs) < 2:
        return []
    pairs: list[dict] = []
    seen_targets: set[str] = set()
    n = len(docs)
    for offset in range(1, n):
        for i, ref in enumerate(docs):
            target = docs[(i + offset) % n]
            if ref["corpus_id"] == target["corpus_id"]:
                continue
            target_hash = text_key(target["text"])
            if target_hash in seen_targets:
                continue
            seen_targets.add(target_hash)
            pair_id = f"pairs_v5_0_{split}_{author}_{len(pairs):04d}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "register": "poetry",
                    "author": author,
                    "author_name": ref.get("author_name"),
                    "instruction": GENERIC_INSTRUCTION,
                    "ref_doc_id": ref["corpus_id"],
                    "target_doc_id": target["corpus_id"],
                    "ref_text": ref["text"],
                    "target_text": target["text"],
                    "ref_source_id": ref.get("source_id"),
                    "target_source_id": target.get("source_id"),
                    "ref_title": ref.get("title"),
                    "target_title": target.get("title"),
                    "split": split,
                    "source_corpus_id": ref.get("cleaning_version", "poetry_corpus_v0_curation_unknown"),
                }
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def split_authors(authors: list[str]) -> dict[str, list[str]]:
    ordered = sorted(authors, key=lambda author: hashlib.sha1(author.encode("utf-8")).hexdigest())
    n = len(ordered)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test : n_test + n_val],
        "train": ordered[n_test + n_val :],
    }


def make_probes(rows: list[dict], *, n_per_author: int = 1, limit: int | None = None) -> list[dict]:
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author"])].append(row)
    authors = sorted(author for author, author_rows in by_author.items() if len(author_rows) >= n_per_author)
    probes: list[dict] = []
    for author_index, author in enumerate(authors):
        swap_author = authors[(author_index + 1) % len(authors)]
        for row_index, row in enumerate(by_author[author][:n_per_author]):
            swap = by_author[swap_author][row_index % len(by_author[swap_author])]
            probes.append(
                {
                    "probe_id": f"pairs_v5_0_poetry_{author}_{row_index:02d}",
                    "register": "poetry",
                    "author": author,
                    "heldout_split": row["split"],
                    "instruction": row["instruction"],
                    "reference_text": row["ref_text"],
                    "expected_target": row["target_text"],
                    "ref_doc_id": row["ref_doc_id"],
                    "target_doc_id": row["target_doc_id"],
                    "swap_reference_author": swap_author,
                    "swap_reference_register": "poetry",
                    "swap_heldout_split": swap["split"],
                    "swap_reference_text": swap["ref_text"],
                    "swap_ref_doc_id": swap["ref_doc_id"],
                }
            )
            if limit is not None and len(probes) >= limit:
                return probes
    return probes


def split_stats(rows: list[dict]) -> dict[str, object]:
    author_counts = Counter(str(row["author"]) for row in rows)
    ref_counts = Counter(str(row["ref_doc_id"]) for row in rows)
    target_counts = Counter(text_key(str(row["target_text"])) for row in rows)
    return {
        "rows": len(rows),
        "authors": len(author_counts),
        "author_counts": dict(sorted(author_counts.items())),
        "unique_refs": len(ref_counts),
        "max_ref_repeat": max(ref_counts.values(), default=0),
        "unique_targets": len(target_counts),
        "target_duplicate_groups": sum(count > 1 for count in target_counts.values()),
        "max_target_repeat": max(target_counts.values(), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/poetry_corpus_v0_source_native_curated_candidate_v9/corpus.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/pairs_v5_0_poetry_corpus_curated"))
    parser.add_argument("--train-pairs-per-author", type=int, default=80)
    parser.add_argument("--heldout-pairs-per-author", type=int, default=24)
    parser.add_argument("--probe-limit", type=int, default=24)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = (repo_root / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    output = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output

    corpus_rows = read_jsonl(corpus_path)
    by_author: dict[str, list[dict]] = defaultdict(list)
    for row in corpus_rows:
        by_author[str(row["author_id"])].append(row)
    author_splits = split_authors([author for author, rows in by_author.items() if len(rows) >= 25])

    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split, authors in author_splits.items():
        cap = args.train_pairs_per_author if split == "train" else args.heldout_pairs_per_author
        for author in authors:
            split_rows[split].extend(make_author_pairs(author, by_author[author], max_pairs=cap, split=split))

    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_corpus": str(corpus_path.relative_to(repo_root)),
        "output": str(output.relative_to(repo_root)),
        "description": "author-disjoint capped poetry pairs derived from high-precision source-native corpus",
        "caps": {
            "train_pairs_per_author": args.train_pairs_per_author,
            "heldout_pairs_per_author": args.heldout_pairs_per_author,
        },
        "author_splits": author_splits,
        "splits": {},
    }

    for split, rows in split_rows.items():
        out_path = output / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        manifest["splits"][split] = {
            **split_stats(rows),
            "sha256": sha256_path(out_path),
        }

    heldout_rows = split_rows["val"] + split_rows["test"]
    probes = make_probes(heldout_rows, n_per_author=1)
    probes_limited = probes[: args.probe_limit]
    write_jsonl(output / "probes_balanced.jsonl", probes)
    write_jsonl(output / "probes_balanced_n24.jsonl", probes_limited)
    manifest["probes_balanced.jsonl"] = {
        "rows": len(probes),
        "authors": dict(sorted(Counter(row["author"] for row in probes).items())),
        "sha256": sha256_path(output / "probes_balanced.jsonl"),
    }
    manifest["probes_balanced_n24.jsonl"] = {
        "rows": len(probes_limited),
        "authors": dict(sorted(Counter(row["author"] for row in probes_limited).items())),
        "sha256": sha256_path(output / "probes_balanced_n24.jsonl"),
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
