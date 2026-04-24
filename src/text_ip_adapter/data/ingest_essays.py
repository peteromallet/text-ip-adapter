from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .ingest_poetry import fetch_gutenberg_text, strip_gutenberg

REGISTER = "essay"
SOURCE_NAME = "project_gutenberg"
LICENSE = "public_domain"

# Curated public-domain essayists on Gutenberg. Each book_id is a
# well-known essay collection; filter_mode adjusts chunk-size heuristics.
GUTENBERG_ESSAYISTS: list[dict] = [
    {"author": "ralph_waldo_emerson", "book_id": 16643},
    {"author": "henry_david_thoreau", "book_id": 205},  # Walden
    {"author": "michel_de_montaigne", "book_id": 3600},  # Essays (translated)
    {"author": "francis_bacon", "book_id": 575},
    {"author": "john_stuart_mill", "book_id": 34901},  # On Liberty
    {"author": "g_k_chesterton", "book_id": 470},
    {"author": "jonathan_swift", "book_id": 1080},  # A Modest Proposal + others
    {"author": "joseph_addison", "book_id": 12030},
    {"author": "virginia_woolf_room", "book_id": 60268},  # A Room of One's Own
    {"author": "william_hazlitt", "book_id": 18618},
    {"author": "charles_lamb", "book_id": 10125},
    {"author": "thomas_carlyle", "book_id": 20585},
    {"author": "john_ruskin", "book_id": 37423},
    {"author": "matthew_arnold_essays", "book_id": 12628},
    {"author": "john_locke_essay", "book_id": 10615},
    {"author": "david_hume_essays", "book_id": 36120},
    {"author": "william_james", "book_id": 11984},  # The Will to Believe
    {"author": "ambrose_bierce", "book_id": 13541},
    {"author": "h_l_mencken", "book_id": 39087},
    {"author": "george_bernard_shaw_prefaces", "book_id": 3506},
]

# Essay-boundary markers.
_ESSAY_HEADERS = [
    re.compile(r"^\s*ESSAY\s+[IVXLCDM]+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*[IVXLCDM]+\.\s+[A-Z][A-Z \-']{3,80}$", re.MULTILINE),
    re.compile(r"^\s*CHAPTER\s+[IVXLCDM]+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*Chapter\s+[IVXLCDM]+\.?.*$", re.MULTILINE),
    re.compile(r"^\s*[A-Z][A-Z \-']{5,80}\n", re.MULTILINE),
]


def split_into_essays(body: str) -> list[str]:
    """Try essay headers first, fall back to generous blank-line splits."""
    best: list[str] = []
    for pat in _ESSAY_HEADERS:
        matches = list(pat.finditer(body))
        if len(matches) >= 4:
            parts: list[str] = []
            prev = 0
            for m in matches:
                if m.start() > prev:
                    parts.append(body[prev:m.start()])
                prev = m.start()
            parts.append(body[prev:])
            if len(parts) > len(best):
                best = parts
    parts = best if best else re.split(r"\n\s*\n\s*\n+", body)

    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\[[^\]]{0,20}\]", " ", p)
        p = re.sub(r"[ \t]+", " ", p)
        p = re.sub(r"\n[ \t]+", "\n", p)
        p = re.sub(r"\n{3,}", "\n\n", p)
        if 500 <= len(p) <= 4000:
            out.append(p)
    return out


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def ingest_essays(cache_dir: Path, max_authors: int | None = None) -> list[dict]:
    """Fetch Gutenberg essay collections and split into essay-level records.

    Env:
        ESSAYS_MAX_AUTHORS: override max_authors for fast testing.
    """
    env_limit = os.environ.get("ESSAYS_MAX_AUTHORS")
    if env_limit:
        try:
            max_authors = int(env_limit)
        except ValueError:
            pass

    entries = GUTENBERG_ESSAYISTS if max_authors is None else GUTENBERG_ESSAYISTS[: max_authors]
    records: list[dict] = []
    for entry in entries:
        try:
            raw = fetch_gutenberg_text(entry["book_id"], cache_dir)
        except Exception as exc:
            print(f"[ingest-essays] skip {entry['author']} ({entry['book_id']}): {exc}")
            continue
        body = strip_gutenberg(raw)
        for text in split_into_essays(body):
            records.append({
                "doc_id": doc_id(entry["author"], text),
                "author": entry["author"],
                "text": text,
                "source": SOURCE_NAME,
                "register": REGISTER,
                "license": LICENSE,
            })

    from collections import Counter
    counts = Counter(r["author"] for r in records)
    qualified = {a for a, c in counts.items() if c >= 3}
    if len(qualified) < 3:
        print(f"[ingest-essays] WARNING: only {len(qualified)} qualified essayists")
    return [r for r in records if r["author"] in qualified]
