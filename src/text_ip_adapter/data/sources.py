from __future__ import annotations

# Registry of all data sources by register. Each register maps to its
# ingestor function. Individual source lists (e.g. GUTENBERG_POETS) moved
# into their register-specific ingest_*.py modules.

from pathlib import Path
from typing import Callable

from .ingest_books import ingest_books
from .ingest_essays import ingest_essays
from .ingest_poetry import ingest_poetry
from .ingest_reddit import ingest_reddit
from .ingest_screenplays import ingest_screenplays
from .ingest_speeches import ingest_speeches

# Signature: ingestor(cache_dir: Path) -> list[dict]
Ingestor = Callable[[Path], list[dict]]

REGISTRY: dict[str, Ingestor] = {
    "poetry": ingest_poetry,
    "prose_fiction": ingest_books,
    "speech": ingest_speeches,
    "essay": ingest_essays,
    "screenplay": ingest_screenplays,
    "reddit": ingest_reddit,
}

# Cache subdir per register.
CACHE_SUBDIRS: dict[str, str] = {
    "poetry": "gutenberg_poetry",
    "prose_fiction": "pg19",
    "speech": "speeches",
    "essay": "gutenberg_essays",
    "screenplay": "imsdb",
    "reddit": "tldr17",
}


def cache_dir_for(register: str, data_root: Path) -> Path:
    subdir = CACHE_SUBDIRS.get(register, register)
    return data_root / "raw" / subdir


# Backward-compat shim: expose the legacy Gutenberg poet manifest without
# re-importing from the renamed module everywhere callers live.
from .ingest_poetry import GUTENBERG_POETS, gutenberg_url  # noqa: E402,F401
