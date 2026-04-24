from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from text_ip_adapter.data.ingest_all import ingest_all
from text_ip_adapter.data.instructions import make_instruction
from text_ip_adapter.data.pairing import make_pairs, split_by_author


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data"
    out_dir = data_root / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    parallel = os.environ.get("INGEST_PARALLEL", "0") == "1"
    workers = int(os.environ.get("INGEST_WORKERS", "3"))

    print(f"[fetch] dispatching multi-register ingest from {data_root} (parallel={parallel})")
    records = ingest_all(data_root, parallel=parallel, workers=workers)
    print(
        f"[fetch] got {len(records)} records from "
        f"{len({(r.get('register'), r.get('author')) for r in records})} (register,author) pairs"
    )
    by_reg = Counter(r.get("register", "?") for r in records)
    for reg, n in sorted(by_reg.items()):
        print(f"[fetch]   register={reg} records={n}")

    pairs = make_pairs(records)
    print(f"[fetch] generated {len(pairs)} pairs")
    for p in pairs:
        p["instruction"] = make_instruction(p["target_text"], register=p.get("register"))

    splits = split_by_author(pairs)
    for name, items in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        reg_break = Counter(p.get("register", "?") for p in items)
        print(f"[fetch] wrote {len(items)} -> {path} breakdown={dict(reg_break)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
