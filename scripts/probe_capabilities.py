#!/usr/bin/env python3
"""Capability tests for a trained adapter checkpoint.

Two probes:
- alpha_blend: z_mix = alpha * z_own + (1 - alpha) * z_swap, alpha in {0, 0.25, 0.5, 0.75, 1.0}
- strength_dial: z_scaled = lambda * z_own, lambda in {0, 0.5, 1.0, 2.0}

For each, runs generation across all parameter values, computes pairwise text
similarity, and writes results to JSONL + analysis.json.

This is the load-bearing test of the architecture's claim: capabilities prompting
literally cannot replicate. If alpha-blend produces smooth interpolation, the
latent space is composable. If strength-dial produces a smooth scaling, the
adapter has a continuous control knob.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from text_ip_adapter.config import ExperimentConfig
from text_ip_adapter.model.adapter_model import AdapterModel
from text_ip_adapter.model.injection import set_prefix_kv


def load_probes(path: str, n_max: int) -> list[dict]:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    return rows[:n_max]


def encode_ref(model, tokenizer, device, text: str, max_len: int = 512) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        z = model._encode_reference(ids["input_ids"], ids["attention_mask"])
    return z


@torch.no_grad()
def gen_with_z(model, tokenizer, device, z: torch.Tensor, instruction: str,
               max_new_tokens: int = 80, max_instr: int = 128) -> str:
    instr_ids = tokenizer(instruction, max_length=max_instr, truncation=True, add_special_tokens=True)["input_ids"]
    sep = tokenizer("\n", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([instr_ids + sep], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    prefix_kv = model.projector(z)
    set_prefix_kv(model._state, prefix_kv)
    try:
        out = model.base.generate(input_ids=ids, attention_mask=mask,
                                   max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
    finally:
        set_prefix_kv(model._state, None)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def ngrams(s: str, n: int = 3) -> set:
    toks = re.findall(r"[A-Za-z']+", s.lower())
    return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-probes", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    cfg = ExperimentConfig.model_validate(yaml.safe_load(open(args.config)))
    probes = load_probes(args.probes, args.n_probes)
    print(f"[cap-probe] loaded {len(probes)} probes")

    print(f"[cap-probe] loading base + adapter")
    model, tokenizer = AdapterModel.from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_trainable_state_dict(sd)
    model.eval()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    lambdas = [0.0, 0.5, 1.0, 2.0]

    alpha_records = []
    lambda_records = []

    for i, probe in enumerate(probes):
        pid = probe.get("probe_id", f"p{i:02d}")
        instr = probe.get("instruction", "")
        ref_text = probe.get("reference_text", "")
        swap_text = probe.get("swap_reference_text", "")

        z_own = encode_ref(model, tokenizer, device, ref_text)
        z_swap = encode_ref(model, tokenizer, device, swap_text)

        # ALPHA-BLEND PROBE: smooth interpolation between own and swap.
        gens_alpha = {}
        for alpha in alphas:
            z_mix = alpha * z_own + (1.0 - alpha) * z_swap
            gen = gen_with_z(model, tokenizer, device, z_mix, instr, max_new_tokens=args.max_new_tokens)
            gens_alpha[alpha] = gen

        # Pairwise jaccards across alpha steps.
        jaccards = {}
        for a1 in alphas:
            for a2 in alphas:
                if a1 < a2:
                    jaccards[f"{a1}_{a2}"] = jaccard(ngrams(gens_alpha[a1]), ngrams(gens_alpha[a2]))

        # Smoothness check: jaccard(α=0, α=0.5) should be similar to jaccard(α=0.5, α=1.0)
        # And jaccard(α=0, α=1.0) should be the smallest (most distant).
        alpha_records.append({
            "probe_id": pid,
            "generations": {str(a): g for a, g in gens_alpha.items()},
            "jaccards": jaccards,
            "j_endpoints": jaccards["0.0_1.0"],  # similarity of α=0 and α=1 outputs
            "j_mid_to_left": jaccards["0.0_0.5"],
            "j_mid_to_right": jaccards["0.5_1.0"],
            "smoothness_balance": abs(jaccards["0.0_0.5"] - jaccards["0.5_1.0"]),  # lower = more symmetric interpolation
        })
        print(f"[cap-probe] {pid} alpha: endpoints J={jaccards['0.0_1.0']:.3f}, mid-balance={alpha_records[-1]['smoothness_balance']:.3f}")

        # STRENGTH-DIAL PROBE: scale z magnitude.
        gens_lambda = {}
        for lam in lambdas:
            z_scaled = lam * z_own
            gen = gen_with_z(model, tokenizer, device, z_scaled, instr, max_new_tokens=args.max_new_tokens)
            gens_lambda[lam] = gen

        # Distance to lambda=1.0 baseline as lambda varies.
        baseline = ngrams(gens_lambda[1.0])
        dists = {str(lam): jaccard(ngrams(gens_lambda[lam]), baseline) for lam in lambdas}
        # Length stability — does lambda=2 produce coherent text or noise?
        lengths = {str(lam): len(gens_lambda[lam]) for lam in lambdas}
        lambda_records.append({
            "probe_id": pid,
            "generations": {str(l): g for l, g in gens_lambda.items()},
            "j_to_baseline": dists,
            "lengths": lengths,
            # Coherence: lambda=2 should still produce reasonable length
            "length_at_lam2_vs_lam1": lengths["2.0"] / max(lengths["1.0"], 1),
        })
        print(f"[cap-probe] {pid} lambda: J_lam0={dists['0.0']:.3f} J_lam0.5={dists['0.5']:.3f} J_lam2={dists['2.0']:.3f}")

    # Aggregate analysis.
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    n = len(alpha_records)
    summary = {
        "n_probes": n,
        "alpha_blend": {
            "mean_endpoint_jaccard": mean([r["j_endpoints"] for r in alpha_records]),
            "mean_mid_to_left": mean([r["j_mid_to_left"] for r in alpha_records]),
            "mean_mid_to_right": mean([r["j_mid_to_right"] for r in alpha_records]),
            "mean_smoothness_balance": mean([r["smoothness_balance"] for r in alpha_records]),
            # The key signal: monotonicity — does jaccard(α=0, α=k) decrease as k grows?
            "monotonic_count": sum(
                1 for r in alpha_records
                if r["jaccards"]["0.0_0.25"] >= r["jaccards"]["0.0_0.5"] >= r["jaccards"]["0.0_0.75"] >= r["jaccards"]["0.0_1.0"]
            ),
        },
        "strength_dial": {
            "mean_j_lam0": mean([float(r["j_to_baseline"]["0.0"]) for r in lambda_records]),
            "mean_j_lam05": mean([float(r["j_to_baseline"]["0.5"]) for r in lambda_records]),
            "mean_j_lam2": mean([float(r["j_to_baseline"]["2.0"]) for r in lambda_records]),
            "mean_length_at_lam2_vs_lam1": mean([r["length_at_lam2_vs_lam1"] for r in lambda_records]),
            # Monotonic distance: jaccard(λ=0, λ=1) < jaccard(λ=0.5, λ=1) < jaccard(λ=2, λ=1)?
            # No — actually we expect jaccard(λ=k, λ=1) to be largest near k=1 and decrease symmetrically.
            "mean_length_at_lam0": mean([r["lengths"]["0.0"] for r in lambda_records]),
            "mean_length_at_lam1": mean([r["lengths"]["1.0"] for r in lambda_records]),
            "mean_length_at_lam2": mean([r["lengths"]["2.0"] for r in lambda_records]),
        },
    }

    with open(out_dir / "alpha_blend_results.jsonl", "w") as f:
        for r in alpha_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "strength_dial_results.jsonl", "w") as f:
        for r in lambda_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "capabilities_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CAPABILITY TEST SUMMARY ===")
    print(f"n_probes: {n}")
    print("\nALPHA-BLEND (own vs swap interpolation):")
    print(f"  mean Jaccard(α=0, α=1):        {summary['alpha_blend']['mean_endpoint_jaccard']:.3f}  (lower = endpoints more distinct)")
    print(f"  mean Jaccard(α=0, α=0.5):      {summary['alpha_blend']['mean_mid_to_left']:.3f}")
    print(f"  mean Jaccard(α=0.5, α=1):      {summary['alpha_blend']['mean_mid_to_right']:.3f}")
    print(f"  mean smoothness balance:        {summary['alpha_blend']['mean_smoothness_balance']:.3f}  (lower = more symmetric interpolation)")
    print(f"  monotonic interpolations:       {summary['alpha_blend']['monotonic_count']}/{n}  (the strict-monotonic count for jaccard(0, k) decreasing in k)")
    print("\nSTRENGTH-DIAL (z magnitude vs λ=1.0 baseline):")
    print(f"  mean Jaccard(λ=0, λ=1):         {summary['strength_dial']['mean_j_lam0']:.3f}  (low = λ=0 differs from λ=1)")
    print(f"  mean Jaccard(λ=0.5, λ=1):       {summary['strength_dial']['mean_j_lam05']:.3f}  (should be intermediate)")
    print(f"  mean Jaccard(λ=2, λ=1):         {summary['strength_dial']['mean_j_lam2']:.3f}  (lower = λ=2 differs from λ=1)")
    print(f"  mean length λ=2 / λ=1:          {summary['strength_dial']['mean_length_at_lam2_vs_lam1']:.2f}  (~1.0 = coherent)")
    print(f"  mean lengths (λ=0, λ=1, λ=2):   {summary['strength_dial']['mean_length_at_lam0']:.0f} / {summary['strength_dial']['mean_length_at_lam1']:.0f} / {summary['strength_dial']['mean_length_at_lam2']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
