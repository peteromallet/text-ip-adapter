from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .sources import REGISTRY, cache_dir_for


def ingest_all(
    data_root: Path,
    registers: list[str] | None = None,
    parallel: bool = False,
    workers: int = 3,
) -> list[dict]:
    """Run every register's ingestor, merge, and return combined records.

    Each register caches into its own data/raw/<subdir>/.

    Args:
        data_root: repo-root-relative data directory (e.g. Path("data")).
        registers: subset of register names to run. None = all registered.
        parallel: if True, run ingestors in parallel threads.
        workers: thread count when parallel=True.

    Env:
        SKIP_REGISTERS: comma-separated list of register names to skip.
        ONLY_REGISTERS: comma-separated list of register names to include
            (overrides `registers` arg).
    """
    selected = registers or list(REGISTRY.keys())
    only_env = os.environ.get("ONLY_REGISTERS")
    if only_env:
        selected = [r.strip() for r in only_env.split(",") if r.strip()]
    skip_env = os.environ.get("SKIP_REGISTERS")
    if skip_env:
        skip = {r.strip() for r in skip_env.split(",") if r.strip()}
        selected = [r for r in selected if r not in skip]

    def run_one(register: str) -> tuple[str, list[dict]]:
        fn = REGISTRY[register]
        cache = cache_dir_for(register, data_root)
        print(f"[ingest-all] start register={register} cache={cache}")
        try:
            recs = fn(cache)
        except Exception as exc:
            print(f"[ingest-all] register={register} FAILED: {exc}")
            return register, []
        print(f"[ingest-all] register={register} records={len(recs)}")
        return register, recs

    merged: list[dict] = []
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one, r): r for r in selected}
            for fut in as_completed(futs):
                _, recs = fut.result()
                merged.extend(recs)
    else:
        for r in selected:
            _, recs = run_one(r)
            merged.extend(recs)

    # Final report by register.
    from collections import Counter
    by_reg = Counter(r.get("register", "?") for r in merged)
    print(f"[ingest-all] merged records: total={len(merged)}; breakdown={dict(by_reg)}")
    return merged
