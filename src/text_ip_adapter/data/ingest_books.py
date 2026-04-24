from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

REGISTER = "prose_fiction"
SOURCE_NAME = "pg19"
LICENSE = "public_domain"

# Chapter markers. Ordered most-specific first.
_CHAPTER_PATTERNS = [
    re.compile(r"^\s*CHAPTER\s+[IVXLCDM]+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*Chapter\s+[IVXLCDM]+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*CHAPTER\s+\d+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*Chapter\s+\d+\.?.*$", re.MULTILINE),
    re.compile(
        r"^\s*CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
        r"ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|"
        r"NINETEEN|TWENTY)\b.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]
_BLANK_SPLIT = re.compile(r"\n\s*\n\s*\n+")


def _author_key_from_title(short_book_title: str, book_id: str | int) -> str:
    """Derive a stable 'author' key.

    Style is book-level when the author isn't cleanly available in pg19.
    We use a slug of the short_book_title plus book_id suffix for uniqueness.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (short_book_title or "").strip().lower())
    slug = slug.strip("_") or "untitled"
    slug = slug[:60]
    return f"{slug}_{book_id}"


def split_book_into_sections(text: str) -> list[str]:
    """Try chapter-marker splits, fall back to 3+ blank-line paragraphs."""
    # Try each chapter pattern; pick the one producing the most splits.
    best_splits: list[str] = []
    for pat in _CHAPTER_PATTERNS:
        matches = list(pat.finditer(text))
        if len(matches) >= 3:
            parts: list[str] = []
            prev = 0
            for m in matches:
                if m.start() > prev:
                    parts.append(text[prev:m.start()])
                prev = m.start()
            parts.append(text[prev:])
            if len(parts) > len(best_splits):
                best_splits = parts
    parts = best_splits if best_splits else _BLANK_SPLIT.split(text)

    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"[ \t]+", " ", p)
        p = re.sub(r"\n{3,}", "\n\n", p)
        if 400 <= len(p) <= 3000:
            out.append(p)
    return out


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def ingest_books(cache_dir: Path, max_books: int | None = None) -> list[dict]:
    """Stream pg19 from HuggingFace, producing section-level records per book.

    Env:
        PG19_MAX_BOOKS: integer cap (default 500; set lower for tests).
        PG19_MIN_SECTIONS: minimum sections per book to include (default 3).
    """
    if max_books is None:
        env_max = os.environ.get("PG19_MAX_BOOKS")
        max_books = int(env_max) if env_max else 500
    min_sections = int(os.environ.get("PG19_MIN_SECTIONS", "3"))

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        print(f"[ingest-books] datasets library unavailable: {exc}")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "pg19_stream.txt"

    records: list[dict] = []
    try:
        ds = load_dataset("deepmind/pg19", split="train", streaming=True, trust_remote_code=True)
    except Exception as exc:
        print(f"[ingest-books] failed to load pg19: {exc}")
        return []

    seen_books = 0
    for row in ds:
        if seen_books >= max_books:
            break
        seen_books += 1

        title = str(row.get("short_book_title", "")) or f"book_{row.get('book_id', seen_books)}"
        book_id = row.get("book_id", seen_books)
        text = row.get("text") or ""
        if not text or len(text) < 2000:
            continue

        author_key = _author_key_from_title(title, book_id)
        sections = split_book_into_sections(text)
        if len(sections) < min_sections:
            continue

        for sec in sections:
            records.append({
                "doc_id": doc_id(author_key, sec),
                "author": author_key,
                "text": sec,
                "source": SOURCE_NAME,
                "register": REGISTER,
                "license": LICENSE,
                "title": title,
                "book_id": book_id,
            })

    # Leave a small cache marker so we can tell the stream ran.
    try:
        marker.write_text(f"streamed_books={seen_books}\nrecords={len(records)}\n")
    except Exception:
        pass

    return records
