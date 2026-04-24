from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import PairDataset, make_collator
from ..eval.samples import build_default_probes, load_probes, run_sample_probe
from ..model.adapter_model import AdapterModel


def contrastive_kv_loss(prefix_kv: dict, clamp: bool = True) -> torch.Tensor:
    """Decorrelate projector K/V outputs across the batch.

    For each injected layer, compute the off-diagonal cosine similarity of
    per-sample K and V (flattened). Penalize high (positive) cosines; ideal
    is orthogonal or negative. Targets the experiment-003-diagnosed failure
    mode where the projector produces near-identical K/V regardless of the
    reference.
    """
    device = None
    total = torch.tensor(0.0)
    n_layers = 0
    for _layer_idx, (K, V) in prefix_kv.items():
        B = K.shape[0]
        if B < 2:
            continue
        if device is None:
            device = K.device
            total = total.to(device)
        k_flat = K.reshape(B, -1).float()
        v_flat = V.reshape(B, -1).float()
        k_norm = torch.nn.functional.normalize(k_flat, dim=-1)
        v_norm = torch.nn.functional.normalize(v_flat, dim=-1)
        k_cos = k_norm @ k_norm.t()
        v_cos = v_norm @ v_norm.t()
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        k_off = k_cos[mask]
        v_off = v_cos[mask]
        if clamp:
            k_off = k_off.clamp(min=0.0)
            v_off = v_off.clamp(min=0.0)
        total = total + 0.5 * (k_off.mean() + v_off.mean())
        n_layers += 1
    if n_layers == 0:
        return torch.tensor(0.0, device=device or "cpu")
    return total / n_layers


def _build_optimizer(model: AdapterModel, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    # Projector vs encoder+queries: distinct LRs.
    projector_params = list(model.projector.parameters())
    encoder_params = list(model.encoder.parameters())
    groups = [
        {"params": projector_params, "lr": cfg.training.lr_projector},
        {"params": encoder_params, "lr": cfg.training.lr_encoder},
    ]
    return torch.optim.AdamW(groups, weight_decay=0.01)


def _lr_at(step: int, base_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _try_wandb_init(project: str, config: dict) -> Any:
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb

        return wandb.init(project=project, config=config, reinit=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] disabled: {exc}")
        return None


def train(cfg: ExperimentConfig) -> dict:
    # Accelerate-based train loop with trainable-only checkpoints.
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=cfg.training.mixed_precision)
    torch.manual_seed(cfg.training.seed)

    model, tokenizer = AdapterModel.from_config(cfg)

    trainable = model.trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable_count:,} / total {total_count:,}")
    assert trainable_count > 0, "No trainable parameters found"

    train_ds = PairDataset(
        cfg.data.train_path,
        tokenizer,
        reference_max=cfg.data.reference_max,
        instruction_max=cfg.data.instruction_max,
        target_max=cfg.data.target_max,
    )
    collate = make_collator(tokenizer, cfg.data.reference_max, cfg.data.instruction_max, cfg.data.target_max)
    loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    optimizer = _build_optimizer(model, cfg)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    wandb_run = _try_wandb_init("text-ip-adapter", cfg.model_dump())
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    # Sample probe setup.
    samples_path = out_dir / "samples.jsonl"
    probe_path = Path(cfg.training.probe_path)
    if probe_path.exists():
        probes = load_probes(str(probe_path))
    else:
        probes = build_default_probes(cfg.data.val_path, n=4)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as pf:
            for p in probes:
                pf.write(json.dumps(p) + "\n")
    baseline_done: set[str] = set()
    # Step-0 baseline sample (pre-training).
    print(f"[train] running step-0 baseline samples ({len(probes)} probes)")
    run_sample_probe(
        model=model, tokenizer=tokenizer, probes=probes, step=0,
        out_path=samples_path, max_new_tokens=cfg.training.sample_max_new_tokens,
        baseline_done_flag=baseline_done,
    )

    step = 0
    t0 = time.time()
    model.train()
    done = False
    summary: dict = {"steps": 0, "trainable_params": trainable_count}
    while not done:
        for batch in loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            # Update LRs.
            for i, g in enumerate(optimizer.param_groups):
                base = cfg.training.lr_projector if i == 0 else cfg.training.lr_encoder
                g["lr"] = _lr_at(step, base, cfg.training.warmup, cfg.training.max_steps)
            with accelerator.accumulate(model):
                contrastive_enabled = cfg.training.contrastive_weight > 0.0
                forward_kwargs = dict(
                    reference_ids=batch["reference_ids"],
                    reference_mask=batch["reference_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                if contrastive_enabled:
                    out, prefix_kv = model(**forward_kwargs, return_prefix_kv=True)
                    loss_ntl = out.loss
                    loss_contrastive = contrastive_kv_loss(prefix_kv, clamp=cfg.training.contrastive_clamp)
                    loss = loss_ntl + cfg.training.contrastive_weight * loss_contrastive
                else:
                    out = model(**forward_kwargs)
                    loss = out.loss
                    loss_ntl = loss
                    loss_contrastive = torch.tensor(0.0, device=loss.device)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

            if step % cfg.training.log_every == 0:
                # Prefix norm is a sanity signal; recomputed cheaply from projector heads.
                with torch.no_grad():
                    prefix_norm = 0.0
                    for li in model.inject_layer_indices if hasattr(model, "inject_layer_indices") else []:
                        pass
                    # Use L2 norm of projector output weights as a cheap proxy.
                    proj_norm = sum(p.float().norm().item() for p in model.projector.parameters())
                grad_norm = 0.0
                for p in trainable:
                    if p.grad is not None:
                        grad_norm += p.grad.float().norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                rec = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_ntl": float(loss_ntl.detach().cpu()),
                    "loss_contrastive": float(loss_contrastive.detach().cpu()),
                    "contrastive_weight": cfg.training.contrastive_weight,
                    "lr_projector": optimizer.param_groups[0]["lr"],
                    "lr_encoder": optimizer.param_groups[1]["lr"],
                    "proj_norm": proj_norm,
                    "grad_norm": grad_norm,
                    "elapsed": time.time() - t0,
                }
                print("[train]", rec)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if wandb_run is not None:
                    wandb_run.log(rec, step=step)

            if step > 0 and step % cfg.training.save_every == 0:
                save_trainable(model, out_dir / f"step_{step}.pt")

            if step > 0 and cfg.training.sample_every > 0 and step % cfg.training.sample_every == 0:
                print(f"[train] running sample probe at step {step}")
                run_sample_probe(
                    model=model, tokenizer=tokenizer, probes=probes, step=step,
                    out_path=samples_path, max_new_tokens=cfg.training.sample_max_new_tokens,
                    baseline_done_flag=baseline_done,
                )

            step += 1
            if step >= cfg.training.max_steps:
                done = True
                break

    # Final probe for the last-step snapshot.
    print(f"[train] running final sample probe at step {step}")
    run_sample_probe(
        model=model, tokenizer=tokenizer, probes=probes, step=step,
        out_path=samples_path, max_new_tokens=cfg.training.sample_max_new_tokens,
        baseline_done_flag=baseline_done,
    )
    save_trainable(model, out_dir / "final.pt")
    log_f.close()
    summary["steps"] = step
    if wandb_run is not None:
        wandb_run.finish()
    return summary


def save_trainable(model: Any, path: Path) -> None:
    # Unwrap accelerate if present.
    base = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base.trainable_state_dict(), path)
    print(f"[train] saved trainable ckpt -> {path}")
