from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REGISTER = "screenplay"
SOURCE_NAME = "imsdb"
LICENSE = "fair_use_review"  # Screenplay text — operator review gate.

IMSDB_BASE = "https://imsdb.com"
IMSDB_INDEX = f"{IMSDB_BASE}/all-scripts.html"
USER_AGENT = "text-ip-adapter/0.1 (research; contact peter@omalley.io)"

# Scene sluglines like "INT. KITCHEN - DAY" / "EXT. DESERT - NIGHT" / "INT/EXT".
_SLUGLINE_RE = re.compile(
    r"^\s*(?:INT|EXT|INT\.?/EXT|INT/EXT|I/E|E/I)[\.\s][^\n]{3,120}$",
    re.MULTILINE | re.IGNORECASE,
)


def _http_get(url: str, timeout: int = 30) -> str:
    # Percent-encode the path (IMSDb URLs contain spaces, commas, ampersands, etc).
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(parsed.path, safe="/")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, parsed.query, parsed.fragment))
    req = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    # IMSDb pages are commonly latin-1 / cp1252.
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def fetch_script_index(cache_dir: Path) -> list[tuple[str, str]]:
    """Return list of (movie_title, script_url) from IMSDb 'all scripts' page."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "all_scripts.html"
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            html = _http_get(IMSDB_INDEX, timeout=60)
            cache_path.write_text(html, encoding="utf-8")
        except Exception as exc:
            print(f"[ingest-screenplays] index fetch failed: {exc}")
            return []

    # <a href="/Movie Scripts/Foo Script.html" title="Foo Script">Foo</a>
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(
        r'<a\s+href="(/Movie[^"]+Script\.html)"[^>]*>([^<]+)</a>',
        html,
        re.IGNORECASE,
    ):
        href, label = m.group(1), m.group(2)
        url = IMSDB_BASE + href
        pairs.append((label.strip(), url))
    # Deduplicate.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for t, u in pairs:
        if u not in seen:
            seen.add(u)
            unique.append((t, u))
    return unique


def _script_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _script_html_to_url(script_page_html: str) -> str | None:
    """Given the IMSDb intermediate script page, find the /scripts/Foo.html link.

    IMSDb's intermediate page ("Movie Scripts/Foo Script.html") contains a
    'Read "Foo" Script' link pointing at /scripts/Foo.html where the <pre>
    block lives.
    """
    m = re.search(
        r'href="(/scripts/[^"]+\.html)"',
        script_page_html,
        re.IGNORECASE,
    )
    if m:
        return IMSDB_BASE + m.group(1)
    return None


def _extract_script_text(html: str) -> str | None:
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    body = m.group(1)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    # HTML entities.
    body = body.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return body


def split_into_scenes(script: str) -> list[str]:
    matches = list(_SLUGLINE_RE.finditer(script))
    if len(matches) < 3:
        return []
    scenes: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(script)
        chunk = script[start:end].strip()
        chunk = re.sub(r"[ \t]+", " ", chunk)
        chunk = re.sub(r"\n{3,}", "\n\n", chunk)
        if 300 <= len(chunk) <= 2500:
            scenes.append(chunk)
    return scenes


def _title_slug(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title.strip().lower())
    return slug.strip("_")[:80] or "untitled"


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def fetch_script(url: str, cache_dir: Path, request_sleep: float = 0.5) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _script_cache_key(url)
    inter_path = cache_dir / f"{key}_inter.html"
    final_path = cache_dir / f"{key}_final.html"

    if final_path.exists():
        return final_path.read_text(encoding="utf-8", errors="ignore")

    # Step 1: intermediate script landing page.
    if inter_path.exists():
        inter_html = inter_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            inter_html = _http_get(url, timeout=30)
            inter_path.write_text(inter_html, encoding="utf-8")
            time.sleep(request_sleep)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"[ingest-screenplays] inter fetch failed {url}: {exc}")
            return None
        except Exception as exc:
            print(f"[ingest-screenplays] inter fetch error {url}: {exc}")
            return None

    final_url = _script_html_to_url(inter_html)
    if not final_url:
        return None

    try:
        final_html = _http_get(final_url, timeout=30)
        final_path.write_text(final_html, encoding="utf-8")
        time.sleep(request_sleep)
        return final_html
    except Exception as exc:
        print(f"[ingest-screenplays] final fetch error {final_url}: {exc}")
        return None


def ingest_screenplays(
    cache_dir: Path,
    max_scripts: int | None = None,
    request_sleep: float = 0.5,
) -> list[dict]:
    """Scrape IMSDb screenplays and split into scenes.

    Env:
        SCREENPLAYS_MAX: integer cap (default 300; set lower for tests).
        SCREENPLAYS_SLEEP: seconds between requests (default 0.5).
    """
    if max_scripts is None:
        env_max = os.environ.get("SCREENPLAYS_MAX")
        max_scripts = int(env_max) if env_max else 300
    env_sleep = os.environ.get("SCREENPLAYS_SLEEP")
    if env_sleep:
        try:
            request_sleep = float(env_sleep)
        except ValueError:
            pass

    index = fetch_script_index(cache_dir)
    if not index:
        print("[ingest-screenplays] no scripts indexed")
        return []

    records: list[dict] = []
    fetched = 0
    for title, url in index:
        if fetched >= max_scripts:
            break
        html = fetch_script(url, cache_dir, request_sleep=request_sleep)
        fetched += 1
        if not html:
            continue
        script_text = _extract_script_text(html)
        if not script_text:
            continue
        scenes = split_into_scenes(script_text)
        if len(scenes) < 3:
            continue
        author_key = _title_slug(title)
        for scene in scenes:
            records.append({
                "doc_id": doc_id(author_key, scene),
                "author": author_key,
                "text": scene,
                "source": SOURCE_NAME,
                "register": REGISTER,
                "license": LICENSE,
                "title": title,
                "url": url,
            })

    return records
