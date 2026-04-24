# text-ip-adapter

Reference-conditioned control of a frozen Gemma-3-4B base model. A small perceiver resampler encodes a reference text into `P = 16` learned latent vectors. A per-layer projector (zero-init) turns those into GQA-aware prefix K/V tensors that are injected into mid-to-upper decoder layers (`[N/2, N-2] = [17, 32]` inclusive for N=34) via PyTorch hooks on `self_attn`. Prefix K is never RoPE-rotated.

## Install

```bash
PYENV_VERSION=3.11.11 python -m pip install -e .[dev]
```

Python >= 3.10, `transformers>=4.50`, `accelerate>=0.34`, `torch>=2.3`.

## Environment

Copy `env.example` to `.env` and populate:

- `HF_TOKEN` — HuggingFace token with Gemma-3 license accepted. If you have `~/.cache/huggingface/token` the code will pick it up automatically.
- `WANDB_API_KEY` — optional; if absent training logs to `checkpoints/<run>/train_log.jsonl` only.
- `RUNPOD_LIFECYCLE_ENV` — path to `../runpod-lifecycle/.env` for the RunPod launcher.
- `RUNPOD_STORAGE_NAME` — network volume to attach (default `Peter`).

## Data

Fetch six registers (poetry, prose fiction, presidential speeches, essays, screenplays, reddit self-posts), build pairs, and write JSONL splits to `data/pairs/`:

```bash
PYENV_VERSION=3.11.11 python scripts/fetch_data.py
```

The dispatcher runs each register's ingestor (see `src/text_ip_adapter/data/ingest_*.py`), caches raw fetches under `data/raw/<register>/`, groups documents by `(register, author_key)`, emits `(reference, target)` pairs within each bucket (never cross-register), synthesizes a rule-based target-only instruction with a register-aware template, and splits 80/10/10 by `(register, author_key)` — authors in any register never appear in both train and test.

Per-register early-abort env vars for fast testing: `POETRY_MAX_AUTHORS`, `PG19_MAX_BOOKS`, `SPEECHES_MAX`, `ESSAYS_MAX_AUTHORS`, `SCREENPLAYS_MAX`, `REDDIT_MAX_ROWS`. Also `ONLY_REGISTERS=poetry,essay` / `SKIP_REGISTERS=reddit` at the dispatcher level.

## Smoke

Imports and shape tests should pass without any Gemma weight download:

```bash
PYENV_VERSION=3.11.11 python -m pytest tests/test_shapes.py tests/test_config.py tests/test_data.py -x
```

A tiny smoke training run (requires the 8GB base download on first use):

```bash
PYENV_VERSION=3.11.11 python scripts/smoke_train.py
```

## Train locally

```bash
PYENV_VERSION=3.11.11 python scripts/train.py --config configs/stage1_gemma.yaml
```

Training loop uses `accelerate` with bf16 mixed precision. Only encoder + projector are trainable; base is frozen. Param groups: projector `lr=1e-4`, encoder+queries `lr=5e-5`. Gradient clip 1.0. Logs every 10 steps, checkpoints every 500 steps (trainable-only, per Settled Decision).

## Inference

```bash
PYENV_VERSION=3.11.11 python scripts/infer.py \
  --config configs/stage1_gemma.yaml \
  --checkpoint checkpoints/stage1_gemma/final.pt \
  --reference path/to/reference.txt \
  --instruction "Write a short poem about storms and silence." \
  --max-new-tokens 128
```

`--reference` accepts either a path or an inline string.

## Launch on RunPod

```bash
PYENV_VERSION=3.11.11 python scripts/train_runpod.py --detach
```

This uses the sibling `runpod-lifecycle` package to launch a pod, sync the project tree and `data/pairs/` separately, install the package on the pod, and run `scripts/train.py`. See `src/text_ip_adapter/infra/` for the unchanged transport layer.

## Repo map

- `src/text_ip_adapter/config.py` — pydantic config, prefix-contract validator
- `src/text_ip_adapter/model/` — encoder, projector, injection hooks, AdapterModel glue
- `src/text_ip_adapter/data/` — sources, ingest, pairing, instruction synthesis, dataset/collator
- `src/text_ip_adapter/train/loop.py` — accelerate training loop
- `src/text_ip_adapter/infra/` — RunPod launcher and SSH sync (left unchanged)
- `configs/stage1_gemma.yaml` — main Stage 1 config
- `configs/smoke.yaml` — tiny smoke config (batch_size=1, max_steps=10)
- `scripts/fetch_data.py`, `scripts/train.py`, `scripts/smoke_train.py`, `scripts/infer.py`, `scripts/train_runpod.py`

## Stage 1 scope

This is the minimal end-to-end runnable path. Phase 2 evaluation harness, reference-swap probe, LoRA, style mixing, MinHash dedup, and LLM-as-judge are explicit future work (see `docs/megaplan.md`).
