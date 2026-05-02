#!/usr/bin/env python3
"""Audit style-transfer pairs with a cheap LLM judge.

For each JSONL pair, asks a conservative auditor to choose:
- keep: pair is usable as training/eval data;
- delete: target/ref contains boilerplate, metadata, list/dictionary/reference
  prose, severe repetition, malformed screenplay, or otherwise bad signal;
- edit: only simple boilerplate removal is needed.

Writes decisions and cleaned split files. The decision log is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are auditing training pairs for a style-transfer model.

Each pair has a reference passage and a target passage from the same register/author. The model learns style from the reference and learns to write the target. Your job is data quality, not taste.

Return ONLY valid JSON with this schema:
{
  "action": "keep" | "delete" | "edit",
  "reason": "short reason",
  "confidence": 0.0-1.0,
  "edited_ref_text": null | "text",
  "edited_target_text": null | "text"
}

Decision rules:
- KEEP if both reference and target are plausible style exemplars for the register.
- DELETE if either passage is mostly metadata, table of contents, index, glossary, dictionary/reference entry, publication notes, headers only, a transcript marker, public-record transmission boilerplate, prompt text, severe repetition, or malformed/empty text.
- DELETE if the target teaches generic summaries like "This book is..." or "The passage is about..." rather than writing in the register.
- DELETE screenplay rows that lack scene/dialogue structure or are mostly prose summaries.
- DELETE speech rows that are formal document transmission records rather than public remarks/speeches.
- EDIT only when the row is otherwise good and the fix is simple boilerplate removal, such as deleting leading "Transcript", title clutter, or repeated header lines. Do not rewrite content or style.
- If unsure, choose DELETE for train data quality; choose KEEP only when the row is clearly usable.
"""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _row_id(split: str, idx: int, row: dict[str, Any]) -> str:
    return "|".join(
        [
            split,
            str(idx),
            str(row.get("register", "")),
            str(row.get("author", "")),
            str(row.get("ref_doc_id", "")),
            str(row.get("target_doc_id", "")),
        ]
    )


def _snippet(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _build_prompt(row: dict[str, Any], text_limit: int) -> str:
    return (
        f"REGISTER: {row.get('register', 'unknown')}\n"
        f"AUTHOR: {row.get('author', 'unknown')}\n"
        f"SOURCE: {row.get('source_dataset', 'unknown')}\n"
        f"INSTRUCTION: {row.get('instruction', '')}\n\n"
        f"REFERENCE:\n{_snippet(str(row.get('ref_text', '')), text_limit)}\n\n"
        f"TARGET:\n{_snippet(str(row.get('target_text', '')), text_limit)}"
    )


def _parse_decision(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    decision = json.loads(raw)
    action = str(decision.get("action", "")).lower()
    if action not in {"keep", "delete", "edit"}:
        raise ValueError(f"bad action: {action!r}")
    return {
        "action": action,
        "reason": str(decision.get("reason", ""))[:500],
        "confidence": float(decision.get("confidence", 0.0)),
        "edited_ref_text": decision.get("edited_ref_text"),
        "edited_target_text": decision.get("edited_target_text"),
    }


def _call_claude(client: Any, row: dict[str, Any], model: str, text_limit: int) -> dict[str, Any]:
    resp = client.messages.create(
        model=model,
        max_tokens=350,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(row, text_limit)}],
    )
    raw = resp.content[0].text if resp.content else ""
    return _parse_decision(raw)


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            existing[str(rec["row_id"])] = rec
    return existing


def _append_decision(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _apply_decision(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any] | None:
    action = decision.get("action")
    if action == "delete":
        return None
    out = dict(row)
    if action == "edit":
        edited_ref = decision.get("edited_ref_text")
        edited_target = decision.get("edited_target_text")
        if isinstance(edited_ref, str) and edited_ref.strip():
            out["ref_text"] = edited_ref.strip()
        if isinstance(edited_target, str) and edited_target.strip():
            out["target_text"] = edited_target.strip()
    out["audit_action"] = action
    out["audit_reason"] = decision.get("reason", "")
    return out


def _process_split(
    client: Any,
    split: str,
    in_path: Path,
    out_dir: Path,
    decisions_dir: Path,
    model: str,
    workers: int,
    max_rows: int,
    text_limit: int,
) -> dict[str, Any]:
    rows = _jsonl(in_path)
    if max_rows:
        rows = rows[:max_rows]

    decision_path = decisions_dir / f"{split}.decisions.jsonl"
    existing = _load_existing(decision_path)
    pending = [(idx, row) for idx, row in enumerate(rows) if _row_id(split, idx, row) not in existing]

    print(f"[{split}] rows={len(rows)} existing={len(existing)} pending={len(pending)} model={model}")
    t0 = time.time()

    def task(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        rid = _row_id(split, idx, row)
        try:
            decision = _call_claude(client, row, model, text_limit)
        except Exception as exc:  # noqa: BLE001
            decision = {
                "action": "delete",
                "reason": f"auditor_error:{type(exc).__name__}:{str(exc)[:160]}",
                "confidence": 0.0,
                "edited_ref_text": None,
                "edited_target_text": None,
            }
        return {"row_id": rid, "split": split, "idx": idx, **decision}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, idx, row): (idx, row) for idx, row in pending}
        for fut in as_completed(futures):
            record = fut.result()
            _append_decision(decision_path, record)
            existing[record["row_id"]] = record
            done += 1
            if done % 25 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(f"  [{split}] {done}/{len(pending)} ({done / elapsed:.2f}/s)")

    kept: list[dict[str, Any]] = []
    counts = {"keep": 0, "delete": 0, "edit": 0}
    by_register: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    for idx, row in enumerate(rows):
        decision = existing[_row_id(split, idx, row)]
        action = decision["action"]
        register = str(row.get("register", "unknown"))
        counts[action] = counts.get(action, 0) + 1
        by_register.setdefault(register, {"keep": 0, "delete": 0, "edit": 0})
        by_register[register][action] = by_register[register].get(action, 0) + 1
        reason = str(decision.get("reason", ""))[:120] or "none"
        reasons[reason] = reasons.get(reason, 0) + 1
        applied = _apply_decision(row, decision)
        if applied is not None:
            kept.append(applied)

    out_path = out_dir / f"{split}.jsonl"
    _write_jsonl(out_path, kept)
    print(f"[{split}] wrote {out_path} kept={len(kept)} counts={counts}")
    return {
        "input": len(rows),
        "output": len(kept),
        "actions": counts,
        "actions_by_register": by_register,
        "top_reasons": dict(sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:25]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pair JSONL with Claude Haiku keep/delete/edit decisions.")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--decisions-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="Per-split cap for smoke tests; 0 means all rows.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Alias for --max-rows.")
    parser.add_argument("--text-limit", type=int, default=1400)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        return 2

    client = anthropic.Anthropic(api_key=api_key)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    decisions_dir = Path(args.decisions_dir)
    summary: dict[str, Any] = {"model": args.model, "splits": {}}
    for split in args.splits:
        in_path = in_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] missing {in_path}")
            continue
        max_rows = args.max_rows or args.max_pairs
        summary["splits"][split] = _process_split(
            client,
            split,
            in_path,
            out_dir,
            decisions_dir,
            args.model,
            args.workers,
            max_rows,
            args.text_limit,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "audit_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
