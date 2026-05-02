#!/usr/bin/env python3
"""Load a trained adapter checkpoint, run probes, write samples.jsonl.

Used for post-hoc re-evaluation of already-trained experiments with a larger or
different probe set, without re-training. Loads the saved trainable-only state
dict and a fresh base model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from text_ip_adapter.config import ExperimentConfig
from text_ip_adapter.eval.samples import build_default_probes, load_probes, run_sample_probe
from text_ip_adapter.model.adapter_model import AdapterModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to final.pt (trainable state dict)")
    parser.add_argument("--config", required=True, help="path to the experiment's config.yaml")
    parser.add_argument("--val-path", required=True, help="path to val.jsonl for probe building")
    parser.add_argument("--n-probes", type=int, default=20)
    parser.add_argument("--output", required=True, help="path to write samples.jsonl")
    parser.add_argument("--probe-path", default=None, help="optional: fixed probe-set path (bypass build_default_probes)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--step-tag", type=int, default=2000, help="integer step tag written into samples.jsonl")
    parser.add_argument("--do-sample", action="store_true", help="enable sampled decoding for all probe variants")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    args = parser.parse_args()

    # Load config.
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f)
    cfg = ExperimentConfig.model_validate(cfg_data)

    # Build or load probes.
    if args.probe_path and Path(args.probe_path).exists():
        probes = load_probes(args.probe_path)
        print(f"[eval] loaded {len(probes)} probes from {args.probe_path}")
    else:
        probes = build_default_probes(args.val_path, n=args.n_probes)
        print(f"[eval] built {len(probes)} probes from {args.val_path}")
        if args.probe_path:
            Path(args.probe_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.probe_path, "w", encoding="utf-8") as f:
                for p in probes:
                    f.write(json.dumps(p) + "\n")

    # Load model + checkpoint.
    print(f"[eval] loading base model {cfg.model.base_model_id}")
    model, tokenizer = AdapterModel.from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_trainable_state_dict(sd)
    model.eval()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[eval] loaded trainable checkpoint ({trainable_count:,} params)")

    # Run probes.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    print(f"[eval] running {len(probes)} probes, writing to {out_path}")
    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    print(f"[eval] generation kwargs: {generation_kwargs}")
    run_sample_probe(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        step=args.step_tag,
        out_path=out_path,
        max_new_tokens=args.max_new_tokens,
        baseline_done_flag=set(),
        generation_kwargs=generation_kwargs,
        prompt_format=cfg.data.prompt_format,
        prompt_reference_max=cfg.data.prompt_reference_max,
    )
    n = sum(1 for _ in open(out_path))
    print(f"[eval] wrote {n} sample rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
