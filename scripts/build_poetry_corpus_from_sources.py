#!/usr/bin/env python3
"""Build a source-native clean poetry corpus candidate from source manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z']+")
GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
BAD_BLOCK_RE = re.compile(
    r"\b(project gutenberg|transcriber'?s note|table of contents|contents|index|"
    r"preface|introduction|publisher|advertisement|bibliography|copyright|"
    r"all rights reserved|distributed proofreaders|produced by|encoded by)\b",
    re.I,
)
APPARATUS_RE = re.compile(r"\b(variant from line|notes?\s+to|footnotes?|glossary)\b", re.I)
HTML_URL_RE = re.compile(r"<[^>]+>|https?://|www\.", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def norm_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown_author"


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def verse_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = WORD_RE.findall(text)
    if len(words) < 24 or len(lines) < 4:
        return False
    line_words = [line_word_count(line) for line in lines]
    if median(line_words) > 18:
        return False
    if sum(1 for count in line_words if count > 32) >= 3:
        return False
    return True


def strip_gutenberg(raw: str) -> str:
    start = GUTENBERG_START_RE.search(raw)
    end = GUTENBERG_END_RE.search(raw)
    if start:
        raw = raw[start.end() :]
    if end:
        raw = raw[: end.start() if not start else max(0, end.start() - (start.end() if start else 0))]
    return raw


def cache_path_for(cache_dir: Path, source: dict, suffix: str = ".txt") -> Path:
    source_name = str(source.get("source_name") or "unknown")
    source_id = str(source.get("source_id") or sha256_text(str(source.get("source_url") or ""))[:12])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
    return cache_dir / source_name / f"{safe_id}{suffix}"


def fetch_url(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": "text-ip-adapter/0.1"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    cache_path.write_text(raw, encoding="utf-8")
    return raw


def internet_archive_text_url(identifier: str) -> str | None:
    meta_url = f"https://archive.org/metadata/{urllib.parse.quote(identifier)}"
    req = urllib.request.Request(meta_url, headers={"User-Agent": "text-ip-adapter/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.load(resp)
    files = meta.get("files", [])
    candidates: list[str] = []
    for file in files:
        name = str(file.get("name") or "")
        fmt = str(file.get("format") or "").lower()
        lower = name.lower()
        if lower.endswith("_djvu.txt") or fmt in {"djvu txt", "text"} or lower.endswith(".txt"):
            if not any(skip in lower for skip in ("_meta", "_files", "_reviews")):
                candidates.append(name)
    if not candidates:
        return None
    candidates.sort(key=lambda name: (0 if name.lower().endswith("_djvu.txt") else 1, len(name)))
    return f"https://archive.org/download/{urllib.parse.quote(identifier)}/{urllib.parse.quote(candidates[0])}"


def fetch_source_text(source: dict, cache_dir: Path) -> tuple[str | None, dict[str, object]]:
    source_name = str(source.get("source_name") or "")
    url = str(source.get("source_url") or "")
    try:
        if source_name == "internet_archive":
            identifier = str(source.get("ia_identifier") or "").strip()
            if not identifier:
                return None, {"fetch_status": "missing_ia_identifier"}
            text_url = internet_archive_text_url(identifier)
            if not text_url:
                return None, {"fetch_status": "missing_ia_text_file"}
            raw = fetch_url(text_url, cache_path_for(cache_dir, source))
            return raw, {"fetch_status": "ok", "text_url": text_url, "raw_sha256": sha256_text(raw)}
        raw = fetch_url(url, cache_path_for(cache_dir, source))
        if source_name == "project_gutenberg":
            raw = strip_gutenberg(raw)
        return raw, {"fetch_status": "ok", "text_url": url, "raw_sha256": sha256_text(raw)}
    except Exception as exc:  # noqa: BLE001
        return None, {"fetch_status": "error", "error": repr(exc)}


def strip_heading(block: str) -> tuple[str | None, str | None]:
    lines = [line.rstrip() for line in block.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    title = None
    if lines:
        first = lines[0].strip()
        if 0 < len(first) <= 80 and line_word_count(first) <= 10 and (
            first.isupper() or re.match(r"^[IVXLCDM\d]+\.?$", first) or first.istitle()
        ):
            title = first
            lines = lines[1:]
    text = "\n".join(lines).strip()
    return (text or None), title


def split_blocks(raw: str) -> list[tuple[str, str | None]]:
    raw = norm_text(raw)
    raw = re.sub(r"\[[^\]]{1,40}\]", " ", raw)
    blocks = re.split(r"\n\s*\n\s*\n+", raw)
    out: list[tuple[str, str | None]] = []
    for block in blocks:
        block = norm_text(block)
        if not block:
            continue
        if len(block) > 5000:
            # Long anthology sections are usually several poems or prose. Split
            # further by double blank lines and let verse filters decide.
            subblocks = re.split(r"\n\s*\n+", block)
        else:
            subblocks = [block]
        for subblock in subblocks:
            subblock = norm_text(subblock)
            if not subblock:
                continue
            text, title = strip_heading(subblock)
            if not text:
                continue
            out.append((text, title))
    return out


def reject_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if BAD_BLOCK_RE.search(text):
        reasons.append("source_artifact")
    if APPARATUS_RE.search(text):
        reasons.append("textual_apparatus")
    if HTML_URL_RE.search(text):
        reasons.append("html_or_url")
    if not verse_like(text):
        reasons.append("not_verse_like")
    if word_count(text) > 900:
        reasons.append("too_long")
    return reasons


def build_records_for_source(source: dict, raw: str, fetch_info: dict[str, object], *, cleaning_version: str) -> tuple[list[dict], list[dict]]:
    author_name = str(source.get("author_name") or "").strip()
    author_id = str(source.get("author_id") or "").strip() or slug(author_name)
    source_id = str(source.get("source_id"))
    records: list[dict] = []
    rejects: list[dict] = []
    seen_in_source: set[str] = set()
    for chunk_index, (text, title_guess) in enumerate(split_blocks(raw)):
        clean_hash = sha256_text(text)
        reasons = reject_reasons(text)
        if clean_hash in seen_in_source:
            reasons.append("duplicate_within_source")
        seen_in_source.add(clean_hash)
        if reasons:
            rejects.append(
                {
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "author_id": author_id,
                    "title_guess": title_guess,
                    "reasons": reasons,
                    "text_preview": text[:500],
                }
            )
            continue
        corpus_id = f"{author_id}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
        records.append(
            {
                "corpus_id": corpus_id,
                "author_id": author_id,
                "author_name": author_name,
                "title": title_guess,
                "text": text,
                "source_id": source_id,
                "source_name": source.get("source_name"),
                "source_url": source.get("source_url"),
                "source_work_title": source.get("source_work_title"),
                "source_publication_year": source.get("source_publication_year"),
                "license": "public_domain_us" if str(source.get("license", "")).startswith("public_domain") else source.get("license"),
                "public_domain_basis": source.get("public_domain_basis"),
                "raw_sha256": fetch_info.get("raw_sha256"),
                "clean_sha256": clean_hash,
                "cleaning_version": cleaning_version,
                "split_hint": None,
                "flags": [],
                "source_chunk_index": chunk_index,
            }
        )
    return records, rejects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", default="data/poetry_corpus_v0_candidate")
    parser.add_argument("--cache-dir", default="data/raw/poetry_corpus_sources")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--cleaning-version", default="poetry_corpus_v0_source_native_001")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output = repo_root / args.output
    cache_dir = repo_root / args.cache_dir
    sources: list[dict] = []
    for manifest in args.manifest:
        sources.extend(read_jsonl(Path(manifest)))
    if args.max_sources is not None:
        sources = sources[: args.max_sources]

    all_records: list[dict] = []
    all_rejects: list[dict] = []
    source_results: list[dict] = []
    seen_global: set[str] = set()
    duplicate_global = 0
    for source in sources:
        raw, fetch_info = fetch_source_text(source, cache_dir)
        if raw is None:
            source_results.append({"source_id": source.get("source_id"), **fetch_info, "records": 0, "rejects": 0})
            continue
        records, rejects = build_records_for_source(source, raw, fetch_info, cleaning_version=args.cleaning_version)
        unique_records: list[dict] = []
        for record in records:
            clean_hash = str(record["clean_sha256"])
            if clean_hash in seen_global:
                duplicate_global += 1
                all_rejects.append(
                    {
                        "source_id": record["source_id"],
                        "chunk_index": record["source_chunk_index"],
                        "author_id": record["author_id"],
                        "title_guess": record["title"],
                        "reasons": ["duplicate_global_clean_text"],
                        "text_preview": record["text"][:500],
                    }
                )
                continue
            seen_global.add(clean_hash)
            unique_records.append(record)
        all_records.extend(unique_records)
        all_rejects.extend(rejects)
        source_results.append(
            {
                "source_id": source.get("source_id"),
                **fetch_info,
                "records": len(unique_records),
                "rejects": len(rejects),
                "author_name": source.get("author_name"),
                "source_work_title": source.get("source_work_title"),
            }
        )

    all_records.sort(key=lambda row: (str(row["author_id"]), str(row["source_id"]), int(row["source_chunk_index"])))
    write_jsonl(output / "corpus.jsonl", all_records)
    write_jsonl(output / "rejected_chunks.jsonl", all_rejects)
    write_jsonl(output / "source_results.jsonl", source_results)
    author_counts = Counter(row["author_id"] for row in all_records)
    manifest = {
        "description": "Source-native poetry corpus candidate; must pass audit before pair derivation.",
        "manifests": args.manifest,
        "output": str(output.relative_to(repo_root)),
        "cache_dir": str(cache_dir.relative_to(repo_root)),
        "source_count": len(sources),
        "sources_fetched": sum(1 for row in source_results if row.get("fetch_status") == "ok"),
        "sources_failed": sum(1 for row in source_results if row.get("fetch_status") != "ok"),
        "corpus_rows": len(all_records),
        "rejected_chunks": len(all_rejects),
        "duplicate_global_rejects": duplicate_global,
        "authors": dict(sorted(author_counts.items())),
        "cleaning_version": args.cleaning_version,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
