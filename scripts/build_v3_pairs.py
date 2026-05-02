#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from text_ip_adapter.data.ingest_essays import ingest_essays
from text_ip_adapter.data.ingest_poetry import ingest_poetry
from text_ip_adapter.data.ingest_screenplays import ingest_screenplays
from text_ip_adapter.data.ingest_speeches import ingest_speeches
from text_ip_adapter.data.instructions import extract_theme, make_instruction
from text_ip_adapter.data.pairing import make_pairs, split_by_author

REGISTER_INGESTORS = {
    "poetry": ("gutenberg_poetry", ingest_poetry),
    "essay": ("gutenberg_essays", ingest_essays),
    "screenplay": ("imsdb", ingest_screenplays),
    "speech": ("speeches", ingest_speeches),
}
DEFAULT_CAPS = {
    "poetry": 100,
    "essay": 100,
    "screenplay": 30,
    "speech": 50,
}
GENERIC_STYLE_INSTRUCTION = "Write a piece in the style of the reference passage."
CONTENT_STYLE_INSTRUCTION = "Use the reference passage for style, and write about: {theme}."
CONTENT_STYLE_INSTRUCTION_NO_THEME = "Use the reference passage for style. Write a new passage on the same broad subject."

SUSPICIOUS_TARGET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in [
        r"^\s*contents\b",
        r"table of contents",
        r"^\s*index\b",
        r"^\s*chapter\s+[ivxlcdm0-9]+\.?\s*$",
        r"this book is a comprehensive reference work",
        r"^\s*the first volume of",
        r"^\s*the story of .* is the story of",
        r"^\s*the film is a fictionalized account",
        r"^\s*the first .* occurred in \d{3,4}",
        r"\bTo the (Senate|House of Representatives)\b",
        r"\bI have the honor to transmit herewith\b",
        r"\bthe bill is referred to the Committee\b",
        r"\bthe shilling is (a coin|divided|worth)\b",
        r"\bthe letter is a written message\b",
        r"\bwrite a \d+ word essay\b",
        r"\bthe reference passage\b",
    ]
]
POETRY_APPARATUS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in [
        r"\bLINENOTES\b",
        r"^\s*PAGE\s+\d+\b",
        r"\bFirst published in \d{4}\b",
        r"^\s*\[[^\]]*published by",
        r"\bPublished in _",
        r"\bwhich was plainly suggested by Homer\b",
        r"\b_Cf_\.",
        r"\(PAGE\s+\d+\.?\)",
        r"=\w+=",
        r"\bPrice,\s*\$",
        r"\bTo be followed by\b",
        r"\bA New Illustrated Edition\b",
        r"\bTHE WORLD AND ITS PEOPLE\b",
        r"\bBOOK\s+[IVXLCDM]+\b",
        r"\bAUTHOR OF \"",
        r"\bEDITED BY\b",
        r"\bSILVER,\s*BURDETT\s*&\s*COMPANY\b",
        r"\bwas born in\b",
        r"\battended a school\b",
        r"\bVoice of the Page\b",
        r"^\s*Entrance\s+[LR]\.",
        r"\bstone ledge\b.*\bparapet\b",
        r"\bMadame Bonaparte\b",
    ]
]
POETRY_CLEAN_LINE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(?:NOTE ON .+|NOTES\.?|NOTE\s+[IVXLCDM0-9]+\.?)\s*$",
        r"^\s*PAGE\s+[IVXLCDM0-9]+(?:\s*\(\d+\))?\.\-\-.*$",
        r"^\s*\[Picture:[^\]]+\]\s*$",
        r"^\s*\(_As sung by_[^)]+\)\s*$",
        r"^\s*\((?:Full\s+)?Chorus\)\b.*$",
        r"^\s*\((?:Bugle|Cornet):[^)]*\)\s*$",
        r"^\s*CONTINUED\.\s*$",
    ]
]
SCREENPLAY_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in [
        r"^\s*\d{1,4}\.\s*$",
        r"^\s*(?:\(?CONTINUED\)?(?:[:.]| TO NEXT PAGE)?|CONT'D\.?)\s*$",
    ]
]
SCREENPLAY_CLEAN_LINE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*\d{1,4}\.?\s*$",
        r"^\s*(?:\(?CONTINUED\)?(?:[:.]| TO NEXT PAGE)?|CONT'D\.?)\s*$",
    ]
]
SPEECH_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in [
        r"^\s*(?:By the President of the United States of America|A Proclamation|To the Congress(?: of the United States)?)\b",
    ]
]


def _source_dataset(record: dict[str, Any]) -> str:
    return str(record.get("source_dataset") or record.get("source") or "unknown")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["source_dataset"] = _source_dataset(record)
    return out


def _load_records(data_root: Path, registers: list[str], max_speeches: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for register in registers:
        cache_subdir, ingestor = REGISTER_INGESTORS[register]
        cache_dir = data_root / "raw" / cache_subdir
        if register == "speech":
            reg_records = ingestor(cache_dir, max_speeches=max_speeches, request_sleep=0)
        else:
            reg_records = ingestor(cache_dir)
        records.extend(_normalize_record(record) for record in reg_records)
        print(f"[v3] register={register} records={len(reg_records)} cache={cache_dir}", flush=True)
    return records


def _source_by_doc_id(records: list[dict[str, Any]]) -> dict[str, str]:
    return {record["doc_id"]: _source_dataset(record) for record in records}


def _parse_caps(raw_caps: list[str]) -> dict[str, int]:
    caps = dict(DEFAULT_CAPS)
    for item in raw_caps:
        if "=" not in item:
            raise ValueError(f"cap must be register=int, got {item!r}")
        reg, value = item.split("=", 1)
        caps[reg.strip()] = int(value)
    return caps


def _apply_instructions(pairs: list[dict[str, Any]], instruction_mode: str) -> None:
    for pair in pairs:
        pair["source_dataset"] = pair.get("source_dataset") or "unknown"
        if instruction_mode == "generic":
            pair["instruction_rule_based"] = make_instruction(pair["target_text"], register=pair.get("register"))
            pair["instruction"] = GENERIC_STYLE_INSTRUCTION
        elif instruction_mode == "content":
            instruction = make_instruction(pair["target_text"], register=pair.get("register"))
            pair["instruction_rule_based"] = instruction
            pair["instruction"] = instruction
        elif instruction_mode == "content_style":
            instruction = make_instruction(pair["target_text"], register=pair.get("register"))
            pair["instruction_rule_based"] = instruction
            theme = extract_theme(pair["target_text"])
            pair["instruction"] = CONTENT_STYLE_INSTRUCTION.format(theme=theme)
        elif instruction_mode == "content_style_no_theme":
            instruction = make_instruction(pair["target_text"], register=pair.get("register"))
            pair["instruction_rule_based"] = instruction
            pair["instruction"] = CONTENT_STYLE_INSTRUCTION_NO_THEME
        elif instruction_mode == "rule":
            instruction = make_instruction(pair["target_text"], register=pair.get("register"))
            pair["instruction"] = instruction
            pair["instruction_rule_based"] = instruction
        else:
            raise ValueError(f"unknown instruction mode: {instruction_mode}")


def _clean_boilerplate_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"(?i)^((view\s+)?transcript[\s:.-]*)+", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*transcript\s*$", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*view\s+transcript\s*$", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*download\s+transcript\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _clean_register_artifact_lines(text: str, register: str) -> str:
    patterns: list[re.Pattern[str]]
    if register == "poetry":
        patterns = POETRY_CLEAN_LINE_PATTERNS
    elif register == "screenplay":
        patterns = SCREENPLAY_CLEAN_LINE_PATTERNS
    else:
        patterns = []
    if not patterns:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in patterns):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _clean_pair_boilerplate(pair: dict[str, Any]) -> None:
    register = str(pair.get("register", "unknown"))
    for key in ("ref_text", "target_text"):
        if key in pair:
            cleaned = _clean_boilerplate_text(str(pair[key]))
            pair[key] = _clean_register_artifact_lines(cleaned, register)


def _repeated_ngram_fraction(text: str, n: int = 5) -> float:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count for gram, count in counts.items() if count > 1)
    return repeated / max(1, len(grams))


def _index_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    index_lines = 0
    for line in lines[:80]:
        if re.match(r"^[A-Z][A-Za-z' .\-]{2,50},\s*\d{1,3}([,\-–\s]\d{1,3})*\.?$", line):
            index_lines += 1
    return index_lines / min(len(lines), 80)


def _suspicious_reasons(pair: dict[str, Any], repeated_threshold: float) -> list[str]:
    reasons: list[str] = []
    register = pair.get("register", "unknown")
    for field_name in ("target_text", "ref_text"):
        text = str(pair.get(field_name) or "")
        stripped = text.strip()
        if field_name == "target_text" and len(stripped) < 100:
            reasons.append("target_lt_100")
        for pattern in SUSPICIOUS_TARGET_PATTERNS:
            if pattern.search(stripped):
                reasons.append(f"{field_name}:regex:{pattern.pattern}")
                break
        if register == "poetry":
            for pattern in POETRY_APPARATUS_PATTERNS:
                if pattern.search(stripped):
                    reasons.append(f"{field_name}:poetry_apparatus:{pattern.pattern}")
                    break
            if _index_line_fraction(stripped) >= 0.25:
                reasons.append(f"{field_name}:poetry_index_lines")
        if register == "screenplay":
            for pattern in SCREENPLAY_ARTIFACT_PATTERNS:
                if pattern.search(stripped):
                    reasons.append(f"{field_name}:screenplay_artifact:{pattern.pattern}")
                    break
        if register == "speech":
            for pattern in SPEECH_ARTIFACT_PATTERNS:
                if pattern.search(stripped):
                    reasons.append(f"{field_name}:speech_artifact:{pattern.pattern}")
                    break
        if _repeated_ngram_fraction(stripped) > repeated_threshold:
            reasons.append(f"{field_name}:repeated_5gram")
    return reasons


def _filter_pairs(pairs: list[dict[str, Any]], repeated_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    removed_by_reason: Counter[str] = Counter()
    removed_by_register: Counter[str] = Counter()
    for pair in pairs:
        reasons = _suspicious_reasons(pair, repeated_threshold)
        if reasons:
            for reason in reasons:
                removed_by_reason[reason] += 1
            removed_by_register[pair.get("register", "unknown")] += 1
            continue
        kept.append(pair)
    return kept, {
        "input_pairs": len(pairs),
        "kept_pairs": len(kept),
        "removed_pairs": len(pairs) - len(kept),
        "removed_by_reason": dict(removed_by_reason.most_common()),
        "removed_by_register": dict(sorted(removed_by_register.items())),
    }


def _load_blocklist(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {"target_doc_ids": set(), "ref_doc_ids": set(), "pair_doc_ids": set()}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "target_doc_ids": set(raw.get("target_doc_ids", [])),
        "ref_doc_ids": set(raw.get("ref_doc_ids", [])),
        "pair_doc_ids": set(raw.get("pair_doc_ids", [])),
    }


def _apply_blocklist(pairs: list[dict[str, Any]], blocklist: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    removed_by_reason: Counter[str] = Counter()
    removed_by_register: Counter[str] = Counter()
    for pair in pairs:
        pair_doc_id = f"{pair.get('ref_doc_id')}|{pair.get('target_doc_id')}"
        reasons: list[str] = []
        if pair.get("target_doc_id") in blocklist["target_doc_ids"]:
            reasons.append("target_doc_id")
        if pair.get("ref_doc_id") in blocklist["ref_doc_ids"]:
            reasons.append("ref_doc_id")
        if pair_doc_id in blocklist["pair_doc_ids"]:
            reasons.append("pair_doc_id")
        if reasons:
            for reason in reasons:
                removed_by_reason[reason] += 1
            removed_by_register[pair.get("register", "unknown")] += 1
            continue
        kept.append(pair)
    return kept, {
        "input_pairs": len(pairs),
        "kept_pairs": len(kept),
        "removed_pairs": len(pairs) - len(kept),
        "removed_by_reason": dict(removed_by_reason.most_common()),
        "removed_by_register": dict(sorted(removed_by_register.items())),
    }


def _clean_pairs(pairs: list[dict[str, Any]]) -> None:
    for pair in pairs:
        _clean_pair_boilerplate(pair)


def _sort_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda p: (p["register"], p["author"], p["ref_doc_id"], p["target_doc_id"]))


def _split_pairs_by_register(
    pairs: list[dict[str, Any]],
    registers: list[str],
    min_heldout: int,
    min_heldout_authors: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split each register by author while satisfying held-out pair floors.

    The generic splitter is author-disjoint but can assign a tiny author to a
    held-out split. That silently creates invalid val/test coverage for scarce
    registers such as Gutenberg essays. Here we assign enough high-capacity
    authors to val and test first, then put the remainder in train.
    """
    del seed  # deterministic capacity split; kept in signature for CLI stability.
    by_reg_author: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        by_reg_author[pair.get("register", "unknown")][pair["author"]].append(pair)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for reg in registers:
        author_rows = by_reg_author.get(reg, {})
        ordered = sorted(author_rows.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(ordered) < 3:
            for _, rows in ordered:
                splits["train"].extend(rows)
            continue

        remaining = list(ordered)
        for split_name in ("test", "val"):
            chosen = _choose_heldout_author_combo(remaining, min_heldout, min_heldout_authors)
            chosen_authors = {author for author, _rows in chosen}
            remaining = [(author, rows) for author, rows in remaining if author not in chosen_authors]
            for _, rows in chosen:
                splits[split_name].extend(rows)

        for _, rows in remaining:
            splits["train"].extend(rows)

    return {name: _sort_pairs(rows) for name, rows in splits.items()}


def _choose_heldout_author_combo(
    author_rows: list[tuple[str, list[dict[str, Any]]]],
    min_pairs: int,
    min_authors: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Pick the smallest author combination that clears held-out gates.

    v3's first capacity-aware split chose the largest authors for val/test,
    which is safe for held-out coverage but can hollow out scarce registers
    such as essay. For v3.1 we reserve high-capacity authors for train when
    smaller held-out combinations can still satisfy the gates.
    """
    if not author_rows:
        return []
    ordered = sorted(author_rows, key=lambda item: (len(item[1]), item[0]))
    max_k = min(len(ordered), max(min_authors + 4, min_authors))
    best: tuple[int, int, tuple[tuple[str, list[dict[str, Any]]], ...]] | None = None
    for k in range(min_authors, max_k + 1):
        for combo in itertools.combinations(ordered, k):
            total = sum(len(rows) for _author, rows in combo)
            if total < min_pairs:
                continue
            candidate = (total, k, combo)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            break
    if best is not None:
        return list(best[2])

    chosen: list[tuple[str, list[dict[str, Any]]]] = []
    total = 0
    for author, rows in ordered:
        chosen.append((author, rows))
        total += len(rows)
        if total >= min_pairs and len(chosen) >= min_authors:
            break
    return chosen


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_register = Counter(row.get("register", "unknown") for row in rows)
    authors_by_register: dict[str, set[str]] = defaultdict(set)
    sources_by_register: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        reg = row.get("register", "unknown")
        authors_by_register[reg].add(row.get("author", "unknown"))
        sources_by_register[reg][row.get("source_dataset", "unknown")] += 1
    return {
        "rows": len(rows),
        "by_register": dict(sorted(by_register.items())),
        "authors_by_register": {reg: len(authors) for reg, authors in sorted(authors_by_register.items())},
        "sources_by_register": {reg: dict(counter) for reg, counter in sorted(sources_by_register.items())},
    }


def _audit(
    splits: dict[str, list[dict[str, Any]]],
    registers: list[str],
    min_heldout: int,
    min_heldout_authors: int,
    min_train_pairs_by_register: dict[str, int],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    failures: list[str] = []

    for split_name in ("train", "val", "test"):
        by_reg = Counter(row.get("register", "unknown") for row in splits[split_name])
        authors_by_reg: dict[str, set[str]] = defaultdict(set)
        for row in splits[split_name]:
            authors_by_reg[row.get("register", "unknown")].add(row.get("author", "unknown"))
        for reg in registers:
            threshold = min_train_pairs_by_register.get(reg, 1) if split_name == "train" else min_heldout
            ok = by_reg.get(reg, 0) >= threshold
            key = f"{split_name}_{reg}_gte_{threshold}"
            gates[key] = ok
            if not ok:
                failures.append(f"{key}: got {by_reg.get(reg, 0)}")
            if split_name in {"val", "test"}:
                author_count = len(authors_by_reg.get(reg, set()))
                key = f"{split_name}_{reg}_authors_gte_{min_heldout_authors}"
                gates[key] = author_count >= min_heldout_authors
                if author_count < min_heldout_authors:
                    failures.append(f"{key}: got {author_count}")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_keys = {(row.get("register"), row.get("author")) for row in splits[left]}
        right_keys = {(row.get("register"), row.get("author")) for row in splits[right]}
        overlap = sorted(left_keys & right_keys)
        key = f"author_disjoint_{left}_{right}"
        gates[key] = not overlap
        if overlap:
            failures.append(f"{key}: {overlap[:5]}")

    source_missing = sum(1 for rows in splits.values() for row in rows if not row.get("source_dataset"))
    gates["source_dataset_present"] = source_missing == 0
    if source_missing:
        failures.append(f"source_dataset_present: missing {source_missing}")

    gates["pass"] = not failures
    return {"gates": gates, "failures": failures}


def _build_balanced_probes(splits: dict[str, list[dict[str, Any]]], registers: list[str], n_per_register: int = 5) -> list[dict[str, Any]]:
    heldout = []
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
            probes.append({
                "probe_id": f"v31_{register}_{i:02d}",
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
            })
    return probes


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v3 text style-transfer pairs with hard data gates.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pairs_v3"))
    parser.add_argument("--register", action="append", choices=sorted(REGISTER_INGESTORS), dest="registers")
    parser.add_argument("--cap", action="append", default=[], help="Override per-author cap, e.g. speech=50")
    parser.add_argument("--max-speeches", type=int, default=1000)
    parser.add_argument("--instruction-mode", choices=("generic", "rule", "content", "content_style", "content_style_no_theme"), default="generic")
    parser.add_argument("--min-heldout-per-register", type=int, default=50)
    parser.add_argument("--min-heldout-authors-per-register", type=int, default=1)
    parser.add_argument("--speech-train-min", type=int, default=1500)
    parser.add_argument("--min-train-pairs", action="append", default=[], help="Per-register train floor, e.g. poetry=1500")
    parser.add_argument("--filter-suspicious-targets", action="store_true")
    parser.add_argument("--clean-boilerplate", action="store_true")
    parser.add_argument("--audit-blocklist", type=Path, default=None)
    parser.add_argument("--repeated-5gram-threshold", type=float, default=0.20)
    parser.add_argument("--write-balanced-probes", action="store_true")
    parser.add_argument("--probes-per-register", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registers = args.registers or ["poetry", "essay", "screenplay", "speech"]
    caps = _parse_caps(args.cap)

    records = _load_records(args.data_root, registers, args.max_speeches)
    pairs = make_pairs(records, max_pairs_by_register=caps, seed=args.seed)
    sources = _source_by_doc_id(records)
    for pair in pairs:
        pair["source_dataset"] = sources.get(pair["ref_doc_id"], "unknown")
    if args.clean_boilerplate:
        _clean_pairs(pairs)
    blocklist = _load_blocklist(args.audit_blocklist)
    blocklist_report = {
        "input_pairs": len(pairs),
        "kept_pairs": len(pairs),
        "removed_pairs": 0,
        "removed_by_reason": {},
        "removed_by_register": {},
    }
    if args.audit_blocklist:
        pairs, blocklist_report = _apply_blocklist(pairs, blocklist)
    filter_report = {"input_pairs": len(pairs), "kept_pairs": len(pairs), "removed_pairs": 0, "removed_by_reason": {}, "removed_by_register": {}}
    if args.filter_suspicious_targets:
        pairs, filter_report = _filter_pairs(pairs, args.repeated_5gram_threshold)
    _apply_instructions(pairs, args.instruction_mode)

    splits = _split_pairs_by_register(
        pairs,
        registers,
        args.min_heldout_per_register,
        args.min_heldout_authors_per_register,
        seed=args.seed,
    )
    min_train_pairs = {"speech": args.speech_train_min}
    for item in args.min_train_pairs:
        if "=" not in item:
            raise ValueError(f"min-train-pairs must be register=int, got {item!r}")
        reg, value = item.split("=", 1)
        min_train_pairs[reg.strip()] = int(value)
    audit = _audit(splits, registers, args.min_heldout_per_register, args.min_heldout_authors_per_register, min_train_pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        _write_jsonl(args.output_dir / f"{split_name}.jsonl", rows)
    probes: list[dict[str, Any]] = []
    if args.write_balanced_probes:
        probes = _build_balanced_probes(splits, registers, n_per_register=args.probes_per_register)
        _write_jsonl(args.output_dir / "probes_balanced_n20.jsonl", probes)

    manifest = {
        "corpus_version": "v3.1" if args.filter_suspicious_targets else "v3",
        "registers": registers,
        "caps": caps,
        "instruction_mode": args.instruction_mode,
        "clean_boilerplate": args.clean_boilerplate,
        "audit_blocklist": str(args.audit_blocklist) if args.audit_blocklist else None,
        "blocklist_report": blocklist_report,
        "filter_suspicious_targets": args.filter_suspicious_targets,
        "filter_report": filter_report,
        "records": _counts(records),
        "pairs": _counts(pairs),
        "splits": {split_name: _counts(rows) for split_name, rows in splits.items()},
        "balanced_probes": _counts(probes) if probes else {"rows": 0},
        "audit": audit,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(manifest["splits"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if audit["gates"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
