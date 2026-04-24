from __future__ import annotations

import random
from collections import defaultdict

# Per-register pair fanout caps. Tuned to produce the target pair budget.
DEFAULT_MAX_PAIRS_BY_REGISTER: dict[str, int] = {
    "poetry": 20,
    "prose_fiction": 6,
    "speech": 15,
    "essay": 12,
    "screenplay": 6,
    "reddit": 4,
}


def _register_of(record: dict) -> str:
    return record.get("register", "unknown")


def _author_key_of(record: dict) -> str:
    return record.get("author", "unknown")


def make_pairs(
    records: list[dict],
    max_pairs_per_author: int | None = None,
    max_pairs_by_register: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Emit (ref, target) pairs within each (register, author) bucket.

    - Pairs are never cross-register (and never cross-author within a register).
    - Fanout cap is per (register, author); uses max_pairs_by_register if given,
      else falls back to max_pairs_per_author (legacy behavior), else to the
      DEFAULT_MAX_PAIRS_BY_REGISTER table.
    """
    rng = random.Random(seed)

    # Group: (register, author) -> docs
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(_register_of(r), _author_key_of(r))].append(r)

    pairs: list[dict] = []
    for (register, author), docs in buckets.items():
        if len(docs) < 2:
            continue
        # Resolve cap.
        if max_pairs_per_author is not None:
            cap = max_pairs_per_author
        elif max_pairs_by_register and register in max_pairs_by_register:
            cap = max_pairs_by_register[register]
        else:
            cap = DEFAULT_MAX_PAIRS_BY_REGISTER.get(register, 6)

        rng.shuffle(docs)
        count = 0
        used: set[tuple[str, str]] = set()
        for i in range(len(docs)):
            for j in range(len(docs)):
                if i == j:
                    continue
                key = (docs[i]["doc_id"], docs[j]["doc_id"])
                if key in used:
                    continue
                used.add(key)
                pairs.append({
                    "register": register,
                    "author": author,
                    "ref_doc_id": docs[i]["doc_id"],
                    "target_doc_id": docs[j]["doc_id"],
                    "ref_text": docs[i]["text"],
                    "target_text": docs[j]["text"],
                })
                count += 1
                if count >= cap:
                    break
            if count >= cap:
                break
    return pairs


def split_by_author(pairs: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Split 80/10/10 by (register, author_key).

    Within each register, pick ~80%/10%/10% of authors for train/val/test so
    every register is represented in every split and no (register, author)
    appears in more than one split. Settled Decision: author-split mandatory.
    """
    rng = random.Random(seed)

    # Gather authors per register.
    authors_by_reg: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        reg = p.get("register", "unknown")
        a = p["author"]
        if a not in seen[reg]:
            seen[reg].add(a)
            authors_by_reg[reg].append(a)

    train_set: set[tuple[str, str]] = set()
    val_set: set[tuple[str, str]] = set()
    test_set: set[tuple[str, str]] = set()

    for reg, authors in authors_by_reg.items():
        ordered = sorted(authors)  # deterministic baseline
        rng.shuffle(ordered)
        n = len(ordered)
        if n < 3:
            # Too few authors — put all in train; val/test will be empty for
            # this register. Caller can decide to tolerate this or raise.
            for a in ordered:
                train_set.add((reg, a))
            continue
        n_val = max(1, n // 10)
        n_test = max(1, n // 10)
        test_authors = ordered[:n_test]
        val_authors = ordered[n_test : n_test + n_val]
        train_authors = ordered[n_test + n_val :]
        for a in test_authors:
            test_set.add((reg, a))
        for a in val_authors:
            val_set.add((reg, a))
        for a in train_authors:
            train_set.add((reg, a))

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        key = (p.get("register", "unknown"), p["author"])
        if key in test_set:
            out["test"].append(p)
        elif key in val_set:
            out["val"].append(p)
        elif key in train_set:
            out["train"].append(p)

    # Invariants.
    assert not (train_set & test_set)
    assert not (train_set & val_set)
    assert not (val_set & test_set)
    return out
