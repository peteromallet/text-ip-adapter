from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

REGISTER = "reddit"
SOURCE_NAME = "webis_tldr_17"
LICENSE = "fair_use_review"  # Reddit self-posts; operator review gate.

_BOT_SUFFIXES = ("_bot", "-bot", "bot")
_BOT_EXACT = {
    "automoderator",
    "autotldr",
    "converter_bot",
    "remindmebot",
    "good_bot",
    "bad_bot",
    "[deleted]",
    "[removed]",
    "none",
}


def _is_bot(author: str) -> bool:
    if not author:
        return True
    a = author.strip().lower()
    if a in _BOT_EXACT:
        return True
    for suf in _BOT_SUFFIXES:
        if a.endswith(suf):
            return True
    return False


def _clean_post(text: str) -> str:
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def ingest_reddit(
    cache_dir: Path,
    max_rows: int | None = None,
    top_users: int | None = None,
    min_post_chars: int = 500,
    min_posts_per_user: int = 5,
) -> list[dict]:
    """Stream webis/tldr-17 from HuggingFace, filter, group by author.

    Env:
        REDDIT_MAX_ROWS: cap on streamed rows (default 500_000).
        REDDIT_TOP_USERS: cap on distinct users kept (default 1000).
        REDDIT_MIN_POST_CHARS: minimum content length (default 500).
        REDDIT_MIN_POSTS_PER_USER: minimum posts/user (default 5).
    """
    if max_rows is None:
        env_max = os.environ.get("REDDIT_MAX_ROWS")
        max_rows = int(env_max) if env_max else 500_000
    if top_users is None:
        env_top = os.environ.get("REDDIT_TOP_USERS")
        top_users = int(env_top) if env_top else 1000
    env_min_chars = os.environ.get("REDDIT_MIN_POST_CHARS")
    if env_min_chars:
        try:
            min_post_chars = int(env_min_chars)
        except ValueError:
            pass
    env_min_posts = os.environ.get("REDDIT_MIN_POSTS_PER_USER")
    if env_min_posts:
        try:
            min_posts_per_user = int(env_min_posts)
        except ValueError:
            pass

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        print(f"[ingest-reddit] datasets library unavailable: {exc}")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "reddit_stream.txt"

    try:
        ds = load_dataset("webis/tldr-17", split="train", streaming=True, trust_remote_code=True)
    except Exception as exc:
        print(f"[ingest-reddit] failed to load tldr-17: {exc}")
        return []

    # First pass: stream, filter, group.
    by_author: dict[str, list[dict]] = defaultdict(list)
    seen = 0
    for row in ds:
        if seen >= max_rows:
            break
        seen += 1
        author = str(row.get("author") or "").strip()
        if _is_bot(author):
            continue
        content = row.get("content") or row.get("body") or ""
        if not isinstance(content, str):
            continue
        content = _clean_post(content)
        if len(content) < min_post_chars:
            continue
        subreddit = row.get("subreddit") or ""
        by_author[author].append({
            "text": content,
            "subreddit": subreddit,
        })

    # Keep only users with enough posts.
    qualified = {
        a: posts for a, posts in by_author.items()
        if len(posts) >= min_posts_per_user
    }

    # Top-N by post count.
    ordered = sorted(qualified.items(), key=lambda kv: len(kv[1]), reverse=True)
    ordered = ordered[: top_users]

    records: list[dict] = []
    for author, posts in ordered:
        for p in posts:
            records.append({
                "doc_id": doc_id(author, p["text"]),
                "author": author,
                "text": p["text"],
                "source": SOURCE_NAME,
                "register": REGISTER,
                "license": LICENSE,
                "subreddit": p["subreddit"],
            })

    try:
        marker.write_text(
            f"streamed_rows={seen}\nusers_kept={len(ordered)}\nrecords={len(records)}\n"
        )
    except Exception:
        pass

    return records
