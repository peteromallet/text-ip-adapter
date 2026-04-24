from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text_ip_adapter import load_experiment_config
from text_ip_adapter.model.adapter_model import AdapterModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Reference-conditioned inference.")
    parser.add_argument("--config", default="configs/stage1_gemma.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to trainable-params .pt file")
    parser.add_argument("--reference", required=True, help="File path or inline string for the reference text")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    model, tokenizer = AdapterModel.from_config(cfg)
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_trainable_state_dict(sd)
    model.eval()

    ref_text = args.reference
    p = Path(args.reference)
    if p.exists() and p.is_file():
        ref_text = p.read_text(encoding="utf-8")

    ref_enc = tokenizer(ref_text, max_length=cfg.data.reference_max, truncation=True, return_tensors="pt")
    instr_enc = tokenizer(args.instruction + "\n", return_tensors="pt")

    out_ids = model.generate(
        reference_ids=ref_enc["input_ids"],
        reference_mask=ref_enc["attention_mask"],
        input_ids=instr_enc["input_ids"],
        attention_mask=instr_enc["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.9,
    )
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
