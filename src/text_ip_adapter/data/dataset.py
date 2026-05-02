from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    # Reads JSONL pairs and tokenizes on-the-fly with the Gemma-3 tokenizer.
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        reference_max: int = 512,
        instruction_max: int = 128,
        target_max: int = 256,
    ) -> None:
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.reference_max = reference_max
        self.instruction_max = instruction_max
        self.target_max = target_max
        self.records: list[dict] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.records.append(json.loads(line))
        self._style_docs_by_register_author: dict[tuple[str, str], list[tuple[str, str]]] = {}
        pools: dict[tuple[str, str], dict[str, str]] = {}
        for r in self.records:
            key = (r.get("register", ""), r.get("author", ""))
            pools.setdefault(key, {})
            for text_key, doc_key in (("ref_text", "ref_doc_id"), ("target_text", "target_doc_id")):
                text = r.get(text_key, "")
                doc_id = r.get(doc_key) or f"{text_key}:{len(pools[key])}"
                if text:
                    pools[key][doc_id] = text
        self._style_docs_by_register_author = {
            key: sorted(docs.items()) for key, docs in pools.items() if docs
        }
        self._authors_by_register: dict[str, list[str]] = {}
        for register, author in self._style_docs_by_register_author:
            self._authors_by_register.setdefault(register, []).append(author)
        self._authors_by_register = {
            register: sorted(set(authors)) for register, authors in self._authors_by_register.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        register = r.get("register", "")
        author = r.get("author", "")
        positive_ref = self._pick_positive_ref(r, idx)
        negative_ref = self._pick_negative_ref(register, author, idx)
        return {
            "ref_text": r["ref_text"],
            "target_text": r["target_text"],
            "instruction": r.get("instruction", "Write a short poem."),
            "author": author,
            "register": register,
            "positive_ref_text": positive_ref,
            "negative_ref_text": negative_ref,
        }

    def _pick_positive_ref(self, r: dict, idx: int) -> str:
        key = (r.get("register", ""), r.get("author", ""))
        docs = self._style_docs_by_register_author.get(key, [])
        if not docs:
            return r.get("ref_text", "")
        avoid = {r.get("ref_doc_id"), r.get("target_doc_id")}
        candidates = [item for item in docs if item[0] not in avoid] or docs
        return candidates[idx % len(candidates)][1]

    def _pick_negative_ref(self, register: str, author: str, idx: int) -> str:
        authors = [a for a in self._authors_by_register.get(register, []) if a != author]
        if not authors:
            return ""
        neg_author = authors[idx % len(authors)]
        docs = self._style_docs_by_register_author.get((register, neg_author), [])
        if not docs:
            return ""
        return docs[(idx // max(1, len(authors))) % len(docs)][1]


def make_collator(
    tokenizer: Any,
    reference_max: int,
    instruction_max: int,
    target_max: int,
    include_style_triplets: bool = False,
    prompt_format: str = "instruction",
    prompt_reference_max: int = 384,
):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        # Reference: BOS + text (for backbone encode).
        ref_enc = tokenizer(
            [b["ref_text"] for b in batch],
            max_length=reference_max,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        # For main sequence: instruction then target; mask instruction span in labels.
        input_ids_list = []
        labels_list = []
        for b in batch:
            if prompt_format == "paired_completion":
                instr_ids = _paired_completion_prompt_ids(
                    tokenizer,
                    b["ref_text"],
                    reference_max=prompt_reference_max,
                )
                sep_ids = []
            else:
                instr_ids = tokenizer(
                    b["instruction"],
                    max_length=instruction_max,
                    truncation=True,
                    add_special_tokens=True,
                )["input_ids"]
                # Add a separator newline token so instruction and target are distinguishable.
                sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
            tgt_ids = tokenizer(b["target_text"], max_length=target_max, truncation=True, add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                tgt_ids = tgt_ids + [eos_id]
            full = instr_ids + sep_ids + tgt_ids
            labels = [-100] * (len(instr_ids) + len(sep_ids)) + list(tgt_ids)
            input_ids_list.append(full)
            labels_list.append(labels)
        # Right-pad to batch max.
        max_len = max(len(x) for x in input_ids_list)
        ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        lab_t = torch.full((len(batch), max_len), -100, dtype=torch.long)
        mask_t = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lab_t[i, : len(labs)] = torch.tensor(labs, dtype=torch.long)
            mask_t[i, : len(ids)] = 1
        out = {
            "reference_ids": ref_enc["input_ids"],
            "reference_mask": ref_enc["attention_mask"],
            "input_ids": ids_t,
            "attention_mask": mask_t,
            "labels": lab_t,
        }
        if include_style_triplets:
            pos_enc = tokenizer(
                [b["positive_ref_text"] for b in batch],
                max_length=reference_max,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            neg_enc = tokenizer(
                [b["negative_ref_text"] for b in batch],
                max_length=reference_max,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            out.update(
                {
                    "positive_reference_ids": pos_enc["input_ids"],
                    "positive_reference_mask": pos_enc["attention_mask"],
                    "negative_reference_ids": neg_enc["input_ids"],
                    "negative_reference_mask": neg_enc["attention_mask"],
                }
            )
        return out

    return collate


def _paired_completion_prompt_ids(tokenizer: Any, reference_text: str, reference_max: int) -> list[int]:
    prefix = "A piece of writing:\n\n"
    suffix = "\n\nAnother piece by the same writer:\n\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    ref_ids = tokenizer(
        reference_text,
        max_length=reference_max,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    return prefix_ids + ref_ids + suffix_ids
