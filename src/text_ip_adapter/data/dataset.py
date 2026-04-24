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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "ref_text": r["ref_text"],
            "target_text": r["target_text"],
            "instruction": r.get("instruction", "Write a short poem."),
            "author": r.get("author", ""),
        }


def make_collator(tokenizer: Any, reference_max: int, instruction_max: int, target_max: int):
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
            instr_ids = tokenizer(b["instruction"], max_length=instruction_max, truncation=True, add_special_tokens=True)["input_ids"]
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
        return {
            "reference_ids": ref_enc["input_ids"],
            "reference_mask": ref_enc["attention_mask"],
            "input_ids": ids_t,
            "attention_mask": mask_t,
            "labels": lab_t,
        }

    return collate
