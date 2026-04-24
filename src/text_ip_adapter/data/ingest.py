from __future__ import annotations

# Legacy shim: this module used to contain the single-register Gutenberg
# poetry ingestor. It's now in ingest_poetry.py. The multi-register
# dispatcher is in ingest_all.py. This file keeps existing imports working.

from .ingest_poetry import (  # noqa: F401
    GUTENBERG_POETS,
    doc_id,
    fetch_gutenberg_text,
    gutenberg_url,
    split_into_docs,
    strip_gutenberg,
)
from .ingest_poetry import ingest_all as _legacy_ingest_all
from .ingest_poetry import ingest_poetry

# Legacy entrypoint name.
ingest_all = _legacy_ingest_all

__all__ = [
    "GUTENBERG_POETS",
    "doc_id",
    "fetch_gutenberg_text",
    "gutenberg_url",
    "ingest_all",
    "ingest_poetry",
    "split_into_docs",
    "strip_gutenberg",
]
