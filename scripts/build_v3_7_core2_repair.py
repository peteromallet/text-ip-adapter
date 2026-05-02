#!/usr/bin/env python3
"""Build v3.7 core2 with screenplay artifact cleanup and less repetitive probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_DIM_RE = re.compile(r"\b\d{2,5}[ \t]*x[ \t]*\d{2,5}\b", re.I)
FILE_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:K|KB|M|MB)\b")
TIMECODE_ARROW_RE = re.compile(r"\b\d{1,2}[.:]\d{2}\s*=>\s*\d{1,2}[.:]\d{2}\b")
BARE_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_PAIR_RE = re.compile(r"^\s*\d{1,4}\s+\d{1,4}\s*$")
REVISION_RE = re.compile(r"^\s*(?:REVISED|REV\.)\b", re.I)
CONT_PAGE_RE = re.compile(r"^\s*(?:\d{1,4}(?:-[A-Z])?\s+)?(?:CONT\.?|CONTINUED)\s*$", re.I)
SINGLE_X_RE = re.compile(r"^\s*X\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def clean_screenplay_text(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if TIMECODE_ARROW_RE.search(stripped):
            counts["drop_timecode_line"] += 1
            continue
        if IMAGE_DIM_RE.search(stripped):
            counts["drop_image_dim_line"] += 1
            continue
        if FILE_SIZE_RE.fullmatch(stripped) or FILE_SIZE_RE.search(stripped):
            counts["drop_file_size_line"] += 1
            continue
        if BARE_PAGE_RE.fullmatch(stripped):
            counts["drop_bare_page_line"] += 1
            continue
        if PAGE_PAIR_RE.fullmatch(stripped):
            counts["drop_page_pair_line"] += 1
            continue
        if REVISION_RE.search(stripped):
            counts["drop_revision_line"] += 1
            continue
        if CONT_PAGE_RE.fullmatch(stripped):
            counts["drop_cont_page_line"] += 1
            continue
        if SINGLE_X_RE.fullmatch(stripped):
            counts["drop_single_x_line"] += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()
    return cleaned, counts


def token_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, Counter[str]]:
    row = dict(row)
    counts: Counter[str] = Counter()
    if row.get("register") != "screenplay":
        return row, counts

    for field in ("ref_text", "target_text"):
        before = row.get(field, "")
        after, field_counts = clean_screenplay_text(before)
        counts.update(field_counts)
        row[field] = after

    if token_count(row.get("ref_text", "")) < 40 or token_count(row.get("target_text", "")) < 20:
        counts["drop_too_short_after_clean"] += 1
        return None, counts

    # Rows with dense timecode/table markup are usually storyboard/scrape
    # fragments rather than screenplay prose. Drop the worst offenders.
    if counts["drop_timecode_line"] >= 2:
        counts["drop_row_timecode_dense"] += 1
        return None, counts
    return row, counts


def ref_key(row: dict[str, Any]) -> str:
    return str(row.get("ref_doc_id") or row.get("ref_text", "")[:80])


def build_balanced_probes(
    splits: dict[str, list[dict[str, Any]]],
    registers: list[str],
    n_per_register: int,
) -> list[dict[str, Any]]:
    heldout: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for row in splits[split_name]:
            item = dict(row)
            item["_heldout_split"] = split_name
            heldout.append(item)

    by_reg_author_ref: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in heldout:
        by_reg_author_ref[row.get("register", "unknown")][row.get("author", "unknown")][ref_key(row)].append(row)

    probes: list[dict[str, Any]] = []
    for register in registers:
        authors = sorted(by_reg_author_ref[register], key=lambda author: (-sum(len(v) for v in by_reg_author_ref[register][author].values()), author))
        if len(authors) < 2:
            continue
        ref_cycle_by_author = {author: sorted(by_reg_author_ref[register][author]) for author in authors}
        for i in range(n_per_register):
            author = authors[i % len(authors)]
            own_refs = ref_cycle_by_author[author]
            own_ref = own_refs[(i // len(authors)) % len(own_refs)]
            own_rows = by_reg_author_ref[register][author][own_ref]
            own = own_rows[(i // (len(authors) * len(own_refs))) % len(own_rows)]

            swap_author = authors[(i + 1) % len(authors)]
            swap_refs = ref_cycle_by_author[swap_author]
            swap_ref = swap_refs[(i // len(authors)) % len(swap_refs)]
            swap_rows = by_reg_author_ref[register][swap_author][swap_ref]
            swap = swap_rows[(i // (len(authors) * len(swap_refs))) % len(swap_rows)]

            probes.append(
                {
                    "probe_id": f"core2v37_{register}_{i:02d}",
                    "author": own.get("author", "unknown"),
                    "register": register,
                    "reference_text": own["ref_text"],
                    "instruction": own["instruction"],
                    "expected_target": own["target_text"],
                    "heldout_split": own["_heldout_split"],
                    "ref_doc_id": own.get("ref_doc_id"),
                    "target_doc_id": own.get("target_doc_id"),
                    "swap_reference_text": swap["ref_text"],
                    "swap_reference_author": swap.get("author", "unknown"),
                    "swap_reference_register": swap.get("register", "unknown"),
                    "swap_heldout_split": swap["_heldout_split"],
                    "swap_ref_doc_id": swap.get("ref_doc_id"),
                }
            )
    return probes


def artifact_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = "\n".join(str(row.get(k, "")) for k in ("ref_text", "target_text", "instruction"))
        if IMAGE_DIM_RE.search(text):
            counts["image_dims"] += 1
        if FILE_SIZE_RE.search(text):
            counts["file_size"] += 1
        if TIMECODE_ARROW_RE.search(text):
            counts["timecode_arrow"] += 1
        if re.search(r"(?m)^\s*\d{1,4}\s*$", text):
            counts["bare_page_line"] += 1
        if re.search(r"(?m)^\s*\d{1,4}\s+\d{1,4}\s*$", text):
            counts["page_pair_line"] += 1
        if re.search(r"(?mi)^\s*(?:REVISED|REV\.)\b", text):
            counts["revision_line"] += 1
        if re.search(r"(?mi)^\s*(?:\d{1,4}(?:-[A-Z])?\s+)?(?:CONT\.?|CONTINUED)\s*$", text):
            counts["cont_page_line"] += 1
    return dict(counts)


def audit(splits: dict[str, list[dict[str, Any]]], registers: list[str], probes: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "author_disjoint": {}}
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split, rows in splits.items():
        by_register = Counter(row["register"] for row in rows)
        authors_by_register: dict[str, set[str]] = defaultdict(set)
        refs_by_register_author: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        split_keys[split] = set()
        for row in rows:
            reg = row["register"]
            author = row.get("author", "unknown")
            authors_by_register[reg].add(author)
            refs_by_register_author[reg][author].add(ref_key(row))
            split_keys[split].add((reg, author))
        out["splits"][split] = {
            "rows": len(rows),
            "by_register": {reg: by_register.get(reg, 0) for reg in registers},
            "authors_by_register": {reg: len(authors_by_register[reg]) for reg in registers},
            "artifact_counts": artifact_counts(rows),
            "unique_refs_by_register_author": {
                reg: {author: len(refs) for author, refs in sorted(refs_by_register_author[reg].items())}
                for reg in registers
            },
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        out["author_disjoint"][f"{left}_{right}"] = not bool(split_keys[left] & split_keys[right])
    out["balanced_probes"] = {
        "rows": len(probes),
        "by_register": dict(Counter(probe["register"] for probe in probes)),
        "authors_by_register": {
            reg: len({probe["author"] for probe in probes if probe["register"] == reg}) for reg in registers
        },
        "unique_ref_doc_ids_by_register": {
            reg: len({probe.get("ref_doc_id") for probe in probes if probe["register"] == reg}) for reg in registers
        },
        "duplicate_probe_refs": len(probes) - len({(probe["register"], probe["author"], probe.get("ref_doc_id")) for probe in probes}),
    }
    out["pass"] = all(out["author_disjoint"].values())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs_v3_5_artifact_clean_core3")
    parser.add_argument("--output-dir", default="data/pairs_v3_7_core2_repaired")
    parser.add_argument("--probes-per-register", type=int, default=10)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    registers = ["poetry", "screenplay"]
    splits: dict[str, list[dict[str, Any]]] = {}
    repair_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in read_jsonl(input_dir / f"{split}.jsonl"):
            if row.get("register") not in registers:
                continue
            repaired, row_counts = repair_row(row)
            counts.update(row_counts)
            if repaired is not None:
                rows.append(repaired)
        splits[split] = rows
        repair_counts[split] = dict(counts)
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    probes = build_balanced_probes(splits, registers, n_per_register=args.probes_per_register)
    write_jsonl(output_dir / "probes_balanced_n20.jsonl", probes)

    core_paths = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "probes_balanced_n20.jsonl")]
    manifest = {
        "corpus_version": "v3.7_core2_repaired",
        "derived_from": str(input_dir),
        "registers": registers,
        "repair_counts": repair_counts,
        "audit": audit(splits, registers, probes),
        "sha256_core": sha(core_paths),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
