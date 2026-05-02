from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from pathlib import Path

REGISTER = "poetry"
SOURCE_NAME = "project_gutenberg"
LICENSE = "public_domain"

# Expanded public-domain Gutenberg poet list. Stable book IDs, verified to
# have enough filtered docs (>=3) per author with default split_into_docs().
# NOTE: `langston_hughes` dropped (book 60902 bad, and Hughes isn't PD).
#       `john_keats` 2422 -> 8209 (his poems, verified >=3 docs).
GUTENBERG_POETS: list[dict] = [
    {"author": "emily_dickinson", "book_id": 12242},
    {"author": "walt_whitman", "book_id": 1322},
    {"author": "john_keats", "book_id": 8209},
    {"author": "william_wordsworth", "book_id": 8774},
    {"author": "robert_frost", "book_id": 59824},
    {"author": "edgar_allan_poe", "book_id": 10031},
    {"author": "william_blake", "book_id": 1934},
    {"author": "robert_browning", "book_id": 16376},
    {"author": "percy_shelley", "book_id": 4800},
    {"author": "alfred_tennyson", "book_id": 8601},
    {"author": "christina_rossetti", "book_id": 19188},
    {"author": "oscar_wilde", "book_id": 1057},
    {"author": "rudyard_kipling", "book_id": 2819},
    # Removed bogus historical/non-poetry IDs found during Experiment 012:
    # lord_byron:16536, dante_rossetti:19023, matthew_arnold:36208,
    # william_carlos_williams:65110, ee_cummings:61653,
    # elizabeth_bishop_early:67524, carl_sandburg:1022,
    # edwin_arlington_robinson:13415, john_masefield:25873,
    # algernon_swinburne:35245, stephen_crane:37158,
    # francis_thompson:29676, amy_lowell:32100, rupert_brooke:3722,
    # sara_teasdale:444, siegfried_sassoon:2729, paul_laurence_dunbar:12423.
    {"author": "samuel_coleridge", "book_id": 29090},
    {"author": "thomas_hardy", "book_id": 3167},
    {"author": "elizabeth_browning", "book_id": 2002},
    {"author": "william_butler_yeats", "book_id": 49608},
    {"author": "dante_rossetti", "book_id": 3692},
    {"author": "matthew_arnold", "book_id": 54985},
    {"author": "henry_longfellow", "book_id": 1365},
    # Expansion set:
    {"author": "edna_millay", "book_id": 109},
    {"author": "edna_millay", "book_id": 1247},
    {"author": "gerard_manley_hopkins", "book_id": 22403},
    {"author": "sara_teasdale", "book_id": 400},
    {"author": "ezra_pound", "book_id": 5186},
    {"author": "t_s_eliot", "book_id": 1459},
    {"author": "rupert_brooke", "book_id": 262},
    {"author": "wilfred_owen", "book_id": 1034},
    {"author": "amy_lowell", "book_id": 841},
    {"author": "amy_lowell", "book_id": 1020},
    {"author": "edwin_arlington_robinson", "book_id": 1035},
    {"author": "edwin_arlington_robinson", "book_id": 1040},
    {"author": "john_masefield", "book_id": 40717},
    {"author": "algernon_swinburne", "book_id": 35402},
    {"author": "stephen_crane", "book_id": 40786},
    {"author": "stephen_crane", "book_id": 9870},
]


def gutenberg_url(book_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


_HEADER_RE = re.compile(
    r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def fetch_gutenberg_text(book_id: int, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pg{book_id}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    url = gutenberg_url(book_id)
    req = urllib.request.Request(url, headers={"User-Agent": "text-ip-adapter/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    cache_path.write_text(raw, encoding="utf-8")
    return raw


def strip_gutenberg(raw: str) -> str:
    m_start = _HEADER_RE.search(raw)
    m_end = _FOOTER_RE.search(raw)
    if m_start:
        raw = raw[m_start.end():]
    if m_end:
        raw = raw[: m_end.start() if not m_start else m_end.start() - m_start.end()]
    return raw


def split_into_docs(body: str) -> list[str]:
    parts = re.split(r"\n\s*\n\s*\n+", body)
    docs: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\[[^\]]{0,20}\]", " ", p)
        p = re.sub(r"[ \t]+", " ", p)
        p = re.sub(r"\n[ \t]+", "\n", p)
        if len(p) < 200 or len(p) > 3000:
            continue
        if p.lower().startswith(("contents", "index", "preface", "table of contents")):
            continue
        docs.append(p)
    return docs


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def ingest_poetry(cache_dir: Path, max_authors: int | None = None) -> list[dict]:
    """Fetch Gutenberg poetry sources and return normalized records.

    Env:
        POETRY_MAX_AUTHORS: override max_authors for fast testing.
    """
    env_limit = os.environ.get("POETRY_MAX_AUTHORS")
    if env_limit:
        try:
            max_authors = int(env_limit)
        except ValueError:
            pass

    entries = GUTENBERG_POETS if max_authors is None else GUTENBERG_POETS[: max_authors]
    records: list[dict] = []
    for entry in entries:
        try:
            raw = fetch_gutenberg_text(entry["book_id"], cache_dir)
        except Exception as exc:
            print(f"[ingest-poetry] skip {entry['author']} ({entry['book_id']}): {exc}")
            continue
        body = strip_gutenberg(raw)
        for text in split_into_docs(body):
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
    if len(qualified) < 5:
        print(f"[ingest-poetry] WARNING: only {len(qualified)} qualified authors")
    return [r for r in records if r["author"] in qualified]


# Legacy entrypoint — kept for backward compatibility.
def ingest_all(cache_dir: Path) -> list[dict]:
    return ingest_poetry(cache_dir)
