from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _tok(tokenizer, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    return out["input_ids"], out["attention_mask"]


def _tok_instr_with_sep(tokenizer, instruction: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Matches the training format: instruction + "\n" separator, ready for target to be generated.
    instr_ids = tokenizer(instruction, max_length=max_len, truncation=True, add_special_tokens=True)["input_ids"]
    sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    ids = instr_ids + sep_ids
    t = torch.tensor([ids], dtype=torch.long)
    m = torch.ones_like(t)
    return t, m


def _tok_paired_completion_prompt(
    tokenizer,
    reference_text: str,
    max_ref_len: int = 384,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_ids = tokenizer("A piece of writing:\n\n", add_special_tokens=True)["input_ids"]
    ref_ids = tokenizer(
        reference_text,
        max_length=max_ref_len,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]
    suffix_ids = tokenizer("\n\nAnother piece by the same writer:\n\n", add_special_tokens=False)["input_ids"]
    ids = prefix_ids + ref_ids + suffix_ids
    t = torch.tensor([ids], dtype=torch.long)
    m = torch.ones_like(t)
    return t, m


def _decode_new(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    new_ids = full_ids[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def load_probes(path: str) -> list[dict]:
    probes: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    return probes


def build_default_probes(val_path: str, n: int = 20) -> list[dict]:
    # Build a diverse probe set:
    # - Stratify across registers (aim for balanced coverage).
    # - Prefer distinct authors across probes.
    # - Swap reference is SAME register but DIFFERENT author (isolates style-within-register signal).
    val_pairs: list[dict] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_pairs.append(json.loads(line))
    if not val_pairs:
        return []
    # Group by register.
    by_register: dict[str, list[dict]] = {}
    for p in val_pairs:
        by_register.setdefault(p.get("register", "unknown"), []).append(p)
    # Balanced round-robin across registers, preferring distinct authors within each.
    per_register_quota = max(1, n // max(len(by_register), 1))
    chosen: list[dict] = []
    seen_authors: set[str] = set()
    for register, pairs in by_register.items():
        count_for_register = 0
        # First pass: one per distinct author in this register.
        for p in pairs:
            if count_for_register >= per_register_quota or len(chosen) >= n:
                break
            if p.get("author") not in seen_authors:
                chosen.append(p)
                seen_authors.add(p.get("author", "unknown"))
                count_for_register += 1
        # Second pass: allow repeat authors if we didn't fill quota.
        for p in pairs:
            if count_for_register >= per_register_quota or len(chosen) >= n:
                break
            if p not in chosen:
                chosen.append(p)
                count_for_register += 1
    # Pad if we're short (register quota rounded down).
    for p in val_pairs:
        if len(chosen) >= n:
            break
        if p not in chosen:
            chosen.append(p)
    chosen = chosen[:n]
    # Build swap refs: same register, different author, prefer one we haven't already used as a swap.
    probes: list[dict] = []
    swap_usage: dict[str, int] = {}
    for i, p in enumerate(chosen):
        p_author = p.get("author", "unknown")
        p_register = p.get("register", "unknown")
        same_reg_diff_author = [q for q in val_pairs if q.get("register") == p_register and q.get("author") != p_author]
        if same_reg_diff_author:
            # Prefer least-used swap.
            same_reg_diff_author.sort(key=lambda q: swap_usage.get(q.get("author", ""), 0))
            swap = same_reg_diff_author[0]
            swap_usage[swap.get("author", "")] = swap_usage.get(swap.get("author", ""), 0) + 1
        else:
            # Fallback: any different-author pair.
            swap = next((q for q in val_pairs if q.get("author") != p_author), chosen[(i + 1) % len(chosen)])
        probes.append({
            "probe_id": f"probe_{i:02d}",
            "author": p_author,
            "register": p_register,
            "reference_text": p["ref_text"],
            "instruction": p["instruction"],
            "expected_target": p["target_text"],
            "swap_reference_text": swap["ref_text"],
            "swap_reference_author": swap.get("author", "unknown"),
            "swap_reference_register": swap.get("register", "unknown"),
        })
    return probes


@torch.no_grad()
def run_sample_probe(
    model: Any,
    tokenizer: Any,
    probes: list[dict],
    step: int,
    out_path: Path,
    max_new_tokens: int = 120,
    include_baseline_once: bool = False,
    baseline_done_flag: set[str] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    prompt_format: str = "instruction",
    prompt_reference_max: int = 384,
) -> list[dict]:
    # Generate adapter, adapter_swap, no_ref, prompted_baseline, and adapter_prompted.
    # Writes one JSONL line per (step, probe_id, variant).
    model.eval()
    base = getattr(model, "module", model)  # unwrap accelerate
    device = next(base.parameters()).device
    records: list[dict] = []
    baseline_done_flag = baseline_done_flag if baseline_done_flag is not None else set()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "use_cache": False,
    }
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)
    raw_gen_kwargs = dict(gen_kwargs)
    with open(out_path, "a", encoding="utf-8") as f:
        for probe in probes:
            ref_ids, ref_mask = _tok(tokenizer, probe["reference_text"], max_len=512)
            swap_ref_ids, swap_ref_mask = _tok(tokenizer, probe["swap_reference_text"], max_len=512)
            # Use training format so the generator knows where to start producing the target.
            if prompt_format == "paired_completion":
                instr_ids, instr_mask = _tok_paired_completion_prompt(
                    tokenizer,
                    probe["reference_text"],
                    max_ref_len=prompt_reference_max,
                )
            else:
                instr_ids, instr_mask = _tok_instr_with_sep(tokenizer, probe["instruction"], max_len=128)
            ref_ids, ref_mask = ref_ids.to(device), ref_mask.to(device)
            swap_ref_ids, swap_ref_mask = swap_ref_ids.to(device), swap_ref_mask.to(device)
            instr_ids, instr_mask = instr_ids.to(device), instr_mask.to(device)
            prompt_len = instr_ids.shape[1]

            # Variant 1: adapter with own reference.
            # NOTE: use_cache=False while prefix-injection KV-cache interaction is unfixed.
            gen = base.generate(
                reference_ids=ref_ids,
                reference_mask=ref_mask,
                input_ids=instr_ids,
                attention_mask=instr_mask,
                **gen_kwargs,
            )
            text = _decode_new(tokenizer, gen, prompt_len)
            rec = {"step": step, "probe_id": probe["probe_id"], "variant": "adapter", "author": probe["author"], "text": text}
            f.write(json.dumps(rec) + "\n"); records.append(rec)

            # Variant 2: adapter with SWAPPED reference.
            gen_sw = base.generate(
                reference_ids=swap_ref_ids,
                reference_mask=swap_ref_mask,
                input_ids=instr_ids,
                attention_mask=instr_mask,
                **gen_kwargs,
            )
            text_sw = _decode_new(tokenizer, gen_sw, prompt_len)
            rec = {"step": step, "probe_id": probe["probe_id"], "variant": "adapter_swap", "author": probe["swap_reference_author"], "text": text_sw}
            f.write(json.dumps(rec) + "\n"); records.append(rec)

            # Variant 3: no reference (random/uniform prefix = base.generate WITHOUT adapter hooks fired).
            # We pass reference_ids but then zero out the projector output. Cheapest: call base directly.
            raw_base = _raw_base(base)
            gen_nr = raw_base.generate(input_ids=instr_ids, attention_mask=instr_mask, **raw_gen_kwargs)
            text_nr = _decode_new(tokenizer, gen_nr, prompt_len)
            rec = {"step": step, "probe_id": probe["probe_id"], "variant": "no_ref", "author": "-", "text": text_nr}
            f.write(json.dumps(rec) + "\n"); records.append(rec)

            # Variant 4: prompted_baseline (once per probe).
            key = f"prompted::{probe['probe_id']}"
            if key not in baseline_done_flag:
                if prompt_format == "paired_completion":
                    prompted = (
                        f'A piece of writing:\n\n{probe["reference_text"]}'
                        "\n\nAnother piece by the same writer:\n\n"
                    )
                    p_ids, p_mask = _tok(tokenizer, prompted, max_len=prompt_reference_max + 96)
                else:
                    prompted = f'Write in the style of the following reference.\n\nReference:\n{probe["reference_text"]}\n\nInstruction:\n{probe["instruction"]}\n\nResponse:\n'
                    p_ids, p_mask = _tok(tokenizer, prompted, max_len=1024)
                p_ids, p_mask = p_ids.to(device), p_mask.to(device)
                gen_pb = raw_base.generate(input_ids=p_ids, attention_mask=p_mask, **raw_gen_kwargs)
                text_pb = _decode_new(tokenizer, gen_pb, p_ids.shape[1])
                rec = {"step": "baseline_once", "probe_id": probe["probe_id"], "variant": "prompted_baseline", "author": probe["author"], "text": text_pb}
                f.write(json.dumps(rec) + "\n"); records.append(rec)

                gen_ap = base.generate(
                    reference_ids=ref_ids,
                    reference_mask=ref_mask,
                    input_ids=p_ids,
                    attention_mask=p_mask,
                    **gen_kwargs,
                )
                text_ap = _decode_new(tokenizer, gen_ap, p_ids.shape[1])
                rec = {"step": step, "probe_id": probe["probe_id"], "variant": "adapter_prompted", "author": probe["author"], "text": text_ap}
                f.write(json.dumps(rec) + "\n"); records.append(rec)
                baseline_done_flag.add(key)

            f.flush()
    model.train()
    return records


def _raw_base(adapter_model: Any) -> Any:
    # Return the raw HF base model (temporarily clear prefix so injection hooks are no-ops).
    from ..model.injection import set_prefix_kv

    if hasattr(adapter_model, "_state"):
        set_prefix_kv(adapter_model._state, None)
    return adapter_model.base
