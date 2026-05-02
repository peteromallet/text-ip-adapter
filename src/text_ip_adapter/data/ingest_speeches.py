from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REGISTER = "speech"
SOURCE_NAME = "miller_center"
LICENSE = "fair_use_review"  # Miller Center transcripts; operator review gate.

SITEMAP_URL = "https://millercenter.org/sitemap.xml"
SPEECH_PATH_PREFIX = "/the-presidency/presidential-speeches/"
USER_AGENT = "text-ip-adapter/0.1 (research; contact peter@omalley.io)"

_STAGE_DIRECTIONS = re.compile(r"\[[^\]]{1,60}\]")
_MULTI_BLANK = re.compile(r"\n\s*\n+")
_DATE_SLUG_RE = re.compile(
    r"/the-presidency/presidential-speeches/"
    r"([a-z]+)-(\d{1,2})-(\d{4})-",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_PRESIDENT_TERMS: tuple[tuple[date, date, str], ...] = (
    (date(1789, 4, 30), date(1797, 3, 4), "washington"),
    (date(1797, 3, 4), date(1801, 3, 4), "adams"),
    (date(1801, 3, 4), date(1809, 3, 4), "jefferson"),
    (date(1809, 3, 4), date(1817, 3, 4), "madison"),
    (date(1817, 3, 4), date(1825, 3, 4), "monroe"),
    (date(1825, 3, 4), date(1829, 3, 4), "jqadams"),
    (date(1829, 3, 4), date(1837, 3, 4), "jackson"),
    (date(1837, 3, 4), date(1841, 3, 4), "vanburen"),
    (date(1841, 3, 4), date(1841, 4, 4), "harrison"),
    (date(1841, 4, 4), date(1845, 3, 4), "tyler"),
    (date(1845, 3, 4), date(1849, 3, 4), "polk"),
    (date(1849, 3, 4), date(1850, 7, 9), "taylor"),
    (date(1850, 7, 9), date(1853, 3, 4), "fillmore"),
    (date(1853, 3, 4), date(1857, 3, 4), "pierce"),
    (date(1857, 3, 4), date(1861, 3, 4), "buchanan"),
    (date(1861, 3, 4), date(1865, 4, 15), "lincoln"),
    (date(1865, 4, 15), date(1869, 3, 4), "andrew_johnson"),
    (date(1869, 3, 4), date(1877, 3, 4), "grant"),
    (date(1877, 3, 4), date(1881, 3, 4), "hayes"),
    (date(1881, 3, 4), date(1881, 9, 19), "garfield"),
    (date(1881, 9, 19), date(1885, 3, 4), "arthur"),
    (date(1885, 3, 4), date(1889, 3, 4), "cleveland"),
    (date(1889, 3, 4), date(1893, 3, 4), "bharrison"),
    (date(1893, 3, 4), date(1897, 3, 4), "cleveland"),
    (date(1897, 3, 4), date(1901, 9, 14), "mckinley"),
    (date(1901, 9, 14), date(1909, 3, 4), "theodore_roosevelt"),
    (date(1909, 3, 4), date(1913, 3, 4), "taft"),
    (date(1913, 3, 4), date(1921, 3, 4), "wilson"),
    (date(1921, 3, 4), date(1923, 8, 2), "harding"),
    (date(1923, 8, 2), date(1929, 3, 4), "coolidge"),
    (date(1929, 3, 4), date(1933, 3, 4), "hoover"),
    (date(1933, 3, 4), date(1945, 4, 12), "fdroosevelt"),
    (date(1945, 4, 12), date(1953, 1, 20), "truman"),
    (date(1953, 1, 20), date(1961, 1, 20), "eisenhower"),
    (date(1961, 1, 20), date(1963, 11, 22), "kennedy"),
    (date(1963, 11, 22), date(1969, 1, 20), "lbjohnson"),
    (date(1969, 1, 20), date(1974, 8, 9), "nixon"),
    (date(1974, 8, 9), date(1977, 1, 20), "ford"),
    (date(1977, 1, 20), date(1981, 1, 20), "carter"),
    (date(1981, 1, 20), date(1989, 1, 20), "reagan"),
    (date(1989, 1, 20), date(1993, 1, 20), "bush"),
    (date(1993, 1, 20), date(2001, 1, 20), "clinton"),
    (date(2001, 1, 20), date(2009, 1, 20), "gwbush"),
    (date(2009, 1, 20), date(2017, 1, 20), "obama"),
    (date(2017, 1, 20), date(2021, 1, 20), "trump"),
    (date(2021, 1, 20), date(2025, 1, 20), "biden"),
    (date(2025, 1, 20), date(2029, 1, 20), "trump"),
)


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _speech_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def fetch_speech_urls(cache_dir: Path, sitemap_url: str = SITEMAP_URL) -> list[str]:
    """Parse Miller Center sitemap, return speech page URLs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sitemap.xml"
    if cache_path.exists():
        xml = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            xml = _http_get(sitemap_url, timeout=60)
            cache_path.write_text(xml, encoding="utf-8")
        except Exception as exc:
            print(f"[ingest-speeches] sitemap fetch failed: {exc}")
            return []

    # Extract <loc>...</loc> entries. Handle sitemap-index format too.
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml, flags=re.IGNORECASE)
    speech_urls: list[str] = []
    sub_sitemaps: list[str] = []
    for url in locs:
        if SPEECH_PATH_PREFIX in url:
            speech_urls.append(url)
        elif url.endswith(".xml"):
            sub_sitemaps.append(url)

    # Follow one level of sub-sitemaps if the top-level was an index.
    for sub in sub_sitemaps:
        try:
            sub_xml = _http_get(sub, timeout=60)
        except Exception:
            continue
        for url in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", sub_xml, flags=re.IGNORECASE):
            if SPEECH_PATH_PREFIX in url:
                speech_urls.append(url)

    # Deduplicate, keep listing order roughly stable.
    seen: set[str] = set()
    unique: list[str] = []
    for u in speech_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _president_slug_from_url(url: str) -> str | None:
    """Infer president from Miller Center speech URL date.

    Current Miller speech pages often expose no page-local president metadata.
    A global nav menu lists Washington first, so anchor-based fallback silently
    collapses every page to `washington`. The URL date is stable and sufficient
    for presidential speeches.
    """
    match = _DATE_SLUG_RE.search(url)
    if not match:
        return None
    month_name, day_raw, year_raw = match.groups()
    month = _MONTHS.get(month_name.lower())
    if month is None:
        return None
    try:
        speech_date = date(int(year_raw), month, int(day_raw))
    except ValueError:
        return None
    for start, end, slug in _PRESIDENT_TERMS:
        if start <= speech_date < end:
            return slug
    return None


def _extract_president_and_transcript(html: str) -> tuple[str | None, str | None]:
    """Extract (president_slug, transcript_text) from a speech HTML page.

    Does not require lxml or bs4 — uses regex on the HTML. Bs4 is preferred
    if available for robustness.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")

        pres: str | None = None
        # Miller Center embeds president in a body class like
        # "body--president-barack-obama" or in a header link.
        body = soup.find("body")
        if body and body.get("class"):
            for cls in body.get("class"):
                m = re.match(r"body--president-([a-z0-9\-]+)", cls)
                if m:
                    pres = m.group(1).replace("-", "_")
                    break
        transcript: str | None = None
        # Miller Center typically has a <div class="transcript"> or field-item.
        candidates = soup.select(
            ".transcript, .view-transcript, .field-name-field-docs-transcript, "
            ".transcript-inner, article .field-items, article .field-item"
        )
        for node in candidates:
            t = node.get_text("\n", strip=True)
            if t and len(t) > 400:
                transcript = t
                break
        if transcript is None:
            # Fall back to <article> text.
            art = soup.find("article")
            if art:
                transcript = art.get_text("\n", strip=True)

        return pres, transcript
    except ImportError:
        # Regex fallback.
        pres_match = re.search(r"body--president-([a-z0-9\-]+)", html)
        pres = pres_match.group(1).replace("-", "_") if pres_match else None
        # Grab anything between the first <article ...> and </article>.
        art = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL | re.IGNORECASE)
        transcript = None
        if art:
            raw = art.group(1)
            raw = re.sub(r"<script.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            raw = re.sub(r"<style.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            raw = re.sub(r"<[^>]+>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            if len(raw) > 400:
                transcript = raw
        return pres, transcript


def _clean_transcript(text: str) -> str:
    text = _STAGE_DIRECTIONS.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def fetch_speech_page(
    url: str,
    cache_dir: Path,
    request_sleep: float = 1.0,
) -> str | None:
    """Fetch a speech page, caching HTML by URL hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_speech_cache_key(url)}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    try:
        html = _http_get(url, timeout=30)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[ingest-speeches] fetch failed {url}: {exc}")
        return None
    except Exception as exc:
        print(f"[ingest-speeches] fetch error {url}: {exc}")
        return None
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(request_sleep)
    return html


def doc_id(author: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{author}_{h}"


def ingest_speeches(
    cache_dir: Path,
    max_speeches: int | None = None,
    request_sleep: float = 1.0,
) -> list[dict]:
    """Scrape Miller Center presidential speeches.

    Env:
        SPEECHES_MAX: integer cap (default 200; set lower for tests).
        SPEECHES_SLEEP: seconds between requests (default 1.0).
    """
    if max_speeches is None:
        env_max = os.environ.get("SPEECHES_MAX")
        max_speeches = int(env_max) if env_max else 200
    env_sleep = os.environ.get("SPEECHES_SLEEP")
    if env_sleep:
        try:
            request_sleep = float(env_sleep)
        except ValueError:
            pass

    urls = fetch_speech_urls(cache_dir)
    if not urls:
        print("[ingest-speeches] no speech URLs discovered")
        return []

    records: list[dict] = []
    fetched = 0
    for url in urls:
        if fetched >= max_speeches:
            break
        html = fetch_speech_page(url, cache_dir, request_sleep=request_sleep)
        fetched += 1
        if not html:
            continue
        html_pres, transcript = _extract_president_and_transcript(html)
        pres = _president_slug_from_url(url) or html_pres
        if not pres or not transcript:
            continue
        transcript = _clean_transcript(transcript)
        if not (500 <= len(transcript) <= 4000):
            # If it's too long, trim to the first 4000 chars on a sentence boundary.
            if len(transcript) > 4000:
                cut = transcript[:4000]
                last_period = cut.rfind(". ")
                if last_period > 2000:
                    transcript = cut[: last_period + 1]
                else:
                    transcript = cut
            else:
                continue

        records.append({
            "doc_id": doc_id(pres, transcript),
            "author": pres,
            "text": transcript,
            "source": SOURCE_NAME,
            "register": REGISTER,
            "license": LICENSE,
            "url": url,
        })

    from collections import Counter
    counts = Counter(r["author"] for r in records)
    qualified = {a for a, c in counts.items() if c >= 2}
    return [r for r in records if r["author"] in qualified]
