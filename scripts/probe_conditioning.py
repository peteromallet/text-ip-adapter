#!/usr/bin/env python3
"""Conditioning-pathway diagnostic for a trained adapter checkpoint.

For each diagnostic probe, generate with 5 reference variants and capture:
- encoder latent z (1 x num_queries x hidden_dim)
- projector prefix K/V per injected layer (magnitude summary + flat vector for cosine)
- generated text

Then compute pairwise cosines at each pathway layer across variants.
Used to localize where reference-specific signal dies.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from text_ip_adapter.config import ExperimentConfig
from text_ip_adapter.model.adapter_model import AdapterModel
from text_ip_adapter.model.injection import set_prefix_kv


OUT_OF_DOMAIN_REF = """def fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class Counter:
    def __init__(self):
        self._count = 0

    def increment(self, step: int = 1) -> int:
        self._count += step
        return self._count

    @property
    def value(self) -> int:
        return self._count"""


def flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).cpu()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten(); b = b.flatten()
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))


def load_probes(path: str, n_max: int) -> list[dict]:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    return rows[:n_max]


def encode_ref(model, tokenizer, device, text: str, max_len: int = 512) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        z = model._encode_reference(ids["input_ids"], ids["attention_mask"])
    return z  # (1, num_queries, hidden_dim), encoder dtype


def random_ref_tokens(tokenizer, length: int = 200, seed: int = 0) -> str:
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size
    # Sample token ids and decode — gives gibberish but valid tokens.
    ids = [rng.randint(1000, vocab_size - 1) for _ in range(length)]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def gen_with_z(model, tokenizer, device, z: torch.Tensor, instruction: str,
               max_new_tokens: int = 100, max_instr: int = 128) -> str:
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
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True), prefix_kv


def summarize_prefix_kv(prefix_kv: dict) -> dict:
    # For each injected layer, store shape info + a flattened tensor (CPU, float) for cosine.
    out = {}
    for li, (K, V) in prefix_kv.items():
        k_flat = flat(K); v_flat = flat(V)
        out[li] = {
            "K_norm": float(K.float().norm().item()),
            "V_norm": float(V.float().norm().item()),
            "K_mean_abs": float(K.float().abs().mean().item()),
            "V_mean_abs": float(V.float().abs().mean().item()),
            "_K_flat": k_flat,  # kept in-memory for cosine; stripped before JSON write
            "_V_flat": v_flat,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--probes", required=True, help="probes jsonl with reference_text / swap_reference_text / instruction")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-probes", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    cfg = ExperimentConfig.model_validate(yaml.safe_load(open(args.config)))
    probes = load_probes(args.probes, args.n_probes)
    print(f"[probe] loaded {len(probes)} probes")

    print(f"[probe] loading base model + adapter")
    model, tokenizer = AdapterModel.from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_trainable_state_dict(sd)
    model.eval()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["own", "swap", "zero", "random", "code"]
    # Persistent shapes.
    hidden_dim = model.encoder.hidden_size
    num_queries = model.encoder.num_queries

    # For the zero variant, we bypass the encoder and feed z=0.
    # For random/code, we encode the pathological reference.

    per_probe_records: list[dict] = []
    z_by_probe_variant: dict[tuple, torch.Tensor] = {}
    kv_by_probe_variant: dict[tuple, dict] = {}
    gen_by_probe_variant: dict[tuple, str] = {}

    for i, probe in enumerate(probes):
        pid = probe.get("probe_id", f"p{i:02d}")
        instr = probe.get("instruction", "Write a short piece.")
        ref_by_variant = {
            "own": probe.get("reference_text", ""),
            "swap": probe.get("swap_reference_text", ""),
            "random": random_ref_tokens(tokenizer, length=150, seed=i),
            "code": OUT_OF_DOMAIN_REF,
        }

        for v in variants:
            if v == "zero":
                # Build a zero z of the right shape/dtype/device, skip encoder.
                z_dtype = next(model.encoder.parameters()).dtype
                z = torch.zeros(1, num_queries, hidden_dim, dtype=z_dtype, device=device)
            else:
                ref_text = ref_by_variant[v]
                z = encode_ref(model, tokenizer, device, ref_text)

            text, prefix_kv = gen_with_z(model, tokenizer, device, z, instr, max_new_tokens=args.max_new_tokens)
            z_by_probe_variant[(pid, v)] = flat(z)
            kv_by_probe_variant[(pid, v)] = summarize_prefix_kv(prefix_kv)
            gen_by_probe_variant[(pid, v)] = text
            print(f"[probe] {pid} {v}: z_norm={z.float().norm().item():.2f} gen_len={len(text)}")

    # Pairwise cosines per probe across variants.
    cosine_table: list[dict] = []
    for probe in probes:
        pid = probe.get("probe_id")
        z_own = z_by_probe_variant[(pid, "own")]
        rec = {"probe_id": pid}
        for v in variants:
            if v == "own":
                rec[f"cos_z_{v}"] = 1.0
                continue
            z_v = z_by_probe_variant[(pid, v)]
            rec[f"cos_z_{v}"] = cos(z_own, z_v)
        # Projector K/V cosines: use first and last injected layers.
        layer_ids = sorted(kv_by_probe_variant[(pid, "own")].keys())
        for li_label, li in [("first", layer_ids[0]), ("last", layer_ids[-1])]:
            kv_own = kv_by_probe_variant[(pid, "own")][li]
            for v in variants:
                if v == "own":
                    rec[f"cos_K_{li_label}_{v}"] = 1.0
                    rec[f"cos_V_{li_label}_{v}"] = 1.0
                    continue
                kv_v = kv_by_probe_variant[(pid, v)][li]
                rec[f"cos_K_{li_label}_{v}"] = cos(kv_own["_K_flat"], kv_v["_K_flat"])
                rec[f"cos_V_{li_label}_{v}"] = cos(kv_own["_V_flat"], kv_v["_V_flat"])
        # Generation 3-gram Jaccard vs own.
        import re
        def ngrams(s: str) -> set:
            toks = re.findall(r"[A-Za-z']+", s.lower())
            return set(tuple(toks[i:i+3]) for i in range(len(toks)-2))
        own_ng = ngrams(gen_by_probe_variant[(pid, "own")])
        for v in variants:
            if v == "own":
                rec[f"gen_jaccard_{v}"] = 1.0; continue
            v_ng = ngrams(gen_by_probe_variant[(pid, v)])
            inter = len(own_ng & v_ng); union = len(own_ng | v_ng)
            rec[f"gen_jaccard_{v}"] = inter / union if union else 0.0
        cosine_table.append(rec)

    # Strip big tensors from the K/V summary before writing JSON.
    kv_for_json = {}
    for (pid, v), lays in kv_by_probe_variant.items():
        kv_for_json[f"{pid}::{v}"] = {li: {k: layer[k] for k in layer if not k.startswith("_")} for li, layer in lays.items()}

    # Write generations.
    with open(out_dir / "generations.jsonl", "w") as f:
        for (pid, v), text in gen_by_probe_variant.items():
            f.write(json.dumps({"probe_id": pid, "variant": v, "text": text}) + "\n")

    # Write K/V summary (per-layer norms).
    with open(out_dir / "prefix_kv_summary.json", "w") as f:
        json.dump(kv_for_json, f, indent=2)

    # Write cosine table.
    with open(out_dir / "cosines.json", "w") as f:
        json.dump({"per_probe": cosine_table, "variants": variants, "n_probes": len(probes)}, f, indent=2)

    # Aggregate summary.
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    agg = {"variants": variants}
    for v in variants:
        if v == "own":
            continue
        agg[f"mean_cos_z_{v}"] = mean([r[f"cos_z_{v}"] for r in cosine_table])
        agg[f"mean_cos_K_first_{v}"] = mean([r[f"cos_K_first_{v}"] for r in cosine_table])
        agg[f"mean_cos_K_last_{v}"] = mean([r[f"cos_K_last_{v}"] for r in cosine_table])
        agg[f"mean_cos_V_first_{v}"] = mean([r[f"cos_V_first_{v}"] for r in cosine_table])
        agg[f"mean_cos_V_last_{v}"] = mean([r[f"cos_V_last_{v}"] for r in cosine_table])
        agg[f"mean_gen_jaccard_{v}"] = mean([r[f"gen_jaccard_{v}"] for r in cosine_table])
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== PATHWAY DIAGNOSTIC SUMMARY (vs own reference) ===")
    print(f"n_probes: {len(probes)}")
    print(f"{'variant':<8} {'cos_z':>8} {'cos_K_1':>8} {'cos_K_N':>8} {'cos_V_1':>8} {'cos_V_N':>8} {'gen_J':>8}")
    for v in variants:
        if v == "own": continue
        print(f"{v:<8} {agg[f'mean_cos_z_{v}']:>8.3f} {agg[f'mean_cos_K_first_{v}']:>8.3f} {agg[f'mean_cos_K_last_{v}']:>8.3f} {agg[f'mean_cos_V_first_{v}']:>8.3f} {agg[f'mean_cos_V_last_{v}']:>8.3f} {agg[f'mean_gen_jaccard_{v}']:>8.3f}")
    print("\nInterpretation guide:")
    print("- cos_z ~ 1.0: encoder produces identical latents → collapsed encoder")
    print("- cos_z diverse but cos_K/V ~ 1.0: projector flattens signal")
    print("- cos_K/V diverse but gen_J ~ 1.0: injection doesn't reach output")
    print("- all layers diverse: signal propagates; issue is style-axis alignment (data or loss)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
