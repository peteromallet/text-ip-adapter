# Implementation Plan: IP-Adapter for Text (Reference-Conditioned LLM Control)

## Overview

### Goal and novelty

Build `text-ip-adapter` as a reference-conditioned control system for a frozen base LLM, starting with Gemma in the ~2B to 4B range. The core idea is to encode a reference text into a small set of dense latent vectors, project those vectors into per-layer prefix K/V states, and inject them into mid-to-upper transformer layers so the model follows target instructions while adopting stylistic properties from the reference. The novelty is not generic prefix conditioning by itself; it is the data recipe and framing: matched `(reference, target)` pairs drawn from the same source element family, such as two poems by the same poet, two scenes from one script, two articles by one journalist, or two speeches by the same speaker. That pairing yields style-shared but content-distinct supervision, which makes reference conditioning learnable without asking the base model to memorize full-text exemplars.

### Repository shape at planning time

This repository starts as a planning shell with an empty `docs/` directory and no research code checked in yet. The working plan assumes research code will be introduced under `src/text_ip_adapter/`, operational scripts under `scripts/`, experiment configs under `configs/`, and project-level metadata in `pyproject.toml`, `README.md`, and `env.example`. Data products are split by lifecycle: raw and processed source material remain local artifacts, while paired training inputs are materialized as a separately synchronized dataset under `data/pairs/`. Evaluation reports and checkpoints stay outside the remote sync contract except when explicitly downloaded as run outputs.

### Upstream prerequisite summary

This project depends on the sibling package at `../runpod-lifecycle/` for ephemeral GPU pod orchestration, remote command execution, and storage integration. The research code in this repository will treat `runpod-lifecycle` as an external dependency rather than a home for model or data logic. The only planned upstream contribution is Step 0 in the sibling package: expose a public `Pod.open_ssh_client()` surface that yields a `paramiko`-compatible SSH client for local `scp` transport. If the current private helper returns a wrapper type, `open_ssh_client()` unwraps to the underlying `paramiko.SSHClient` so downstream `scp.SCPClient(ssh.get_transport())` works. All other experimentation, data, model, training, evaluation, and infra glue remain in this repository.

### Constraints

- `GQA compatibility` is non-negotiable because Gemma uses grouped-query attention; injected prefix K/V tensors must be shaped against `num_kv_heads`, not `num_attention_heads`, and the injector must remain correct across Gemma variants.
- `Content leak is failure mode #1.` The system is only useful if reference swaps change style without copying reference content. The full plan therefore treats leak prevention as a three-layer firewall spanning pair construction, synthetic instruction generation, and evaluation.
- `Licensing posture is public-domain first.` Public-domain and clearly redistributable sources are the default path. Modern or ambiguous sources may enter only behind a per-source review gate, with explicit manifest metadata and manual approval before use in training.
- `Compute is constrained.` The default path assumes single-pod, single-GPU runs on ephemeral RunPod hardware, with Stage 1 sized for one `4090` or `A40` class GPU and optional later stages only if earlier gates justify the spend.
- `RoPE interaction must stay explicit.` Prefix keys are prepended as learned latent memory and are not rotated with content positions; the hook implementation and tests must keep that invariant obvious.

### Encoder ↔ injector contract

The encoder and injector are coupled by a fixed prefix-token contract: `num_queries == num_prefix_tokens == P`, with default `P = 16`. The reference encoder always emits exactly `P` latent vectors per example, and the projector always maps those `P` vectors into the prefix K/V tensors consumed by the injector. This resolves the planning ambiguity in `FLAG-005`: there is no separate learned query count and runtime prefix length. One config value defines both, and the config layer will reject mismatches rather than trying to broadcast or truncate.

### Remote-workspace contract

The remote pod workspace is intentionally narrow. The required-on-pod project payload is:

- `pyproject.toml`
- `README.md`
- `env.example`
- `src/text_ip_adapter/`
- `scripts/`
- `configs/`
- `data/pairs/` synced separately behind its own hash gate

The excluded paths are:

- `.venv/`
- `__pycache__/`
- `wandb/`
- `checkpoints/`
- `data/raw/`
- `data/processed/`
- `.git/`
- `eval_reports/`

This contract exists to keep pod sync fast, deterministic, and auditable. Code and configs move as one atomic project-root sync, while `data/pairs/` is treated as a larger dataset artifact with separate change detection and transfer rules.

### Scope guardrail

Research code for this project lives in `src/text_ip_adapter/` and nowhere inside `../runpod-lifecycle/`. The sibling package is a dependency with one narrowly scoped upstream change in Step 0 to expose the SSH client surface needed for transport helpers. No data pipeline, model architecture, training loop, evaluation logic, or experiment orchestration code is to be added to `runpod-lifecycle`; those concerns belong to this repository.

## Phase 0: Upstream prerequisite (sibling package)

### Step 0: Add `Pod.open_ssh_client()` to `runpod-lifecycle`

Add a public `Pod.open_ssh_client()` method in `../runpod-lifecycle/src/runpod_lifecycle/pod.py` as the only planned upstream code change needed by this project. The method should wrap the existing `_build_ssh_client()` auth ladder instead of inventing a second SSH path: first prefer an inline private key already available to the pod client, then a configured private-key path, and only then fall back to password auth. The public contract for downstream consumers is a `paramiko`-compatible SSH client suitable for file transfer helpers, not the current private helper shape.

Document the new public surface in `../runpod-lifecycle/README.md`, including the intended usage boundary: `text-ip-adapter` may call `Pod.open_ssh_client()` for explicit transport operations, while ordinary remote command execution can keep using the existing higher-level pod APIs. If the current private helper returns a wrapper type, `open_ssh_client()` unwraps to the underlying `paramiko.SSHClient` so `scp.SCPClient(ssh.get_transport())` works downstream.

Extend coverage in `../runpod-lifecycle/tests/test_ssh_details.py` to lock down the public behavior rather than just the private helper internals. The tests should cover the auth ladder order of operations, ensure the public method returns a `paramiko`-compatible client even when the private implementation uses an internal wrapper, and confirm that the local auth escape hatch is only active when `TEXT_IP_ADAPTER_SSH_LOCAL_AUTH=1` is set. Because this is a public API addition in the sibling dependency, plan for a patch version bump in `../runpod-lifecycle/pyproject.toml` after the method, docs, and tests land.

## Phase 1: Foundation — Repo Scaffolding, Dependencies, Config

### Step 1: Initialize packaging and dependency surface in `pyproject.toml`

Define the project package in `pyproject.toml` with runtime dependencies listed explicitly rather than accreting them ad hoc during experimentation. The required runtime set is: `torch>=2.3`, `transformers>=4.44`, `accelerate>=0.33`, `sentencepiece`, `bitsandbytes`, `peft>=0.11`, `numpy`, `datasets>=2.20`, `pyarrow>=15`, `datasketch`, `ftfy`, `tqdm`, `pyyaml`, `pydantic>=2`, `regex`, `sentence-transformers>=3.0`, `scikit-learn>=1.4`, `scipy>=1.11`, `sacrebleu`, `rouge-score`, `wandb>=0.17`, `paramiko>=3.4`, `scp>=0.14`, optional `anthropic>=0.30`, and `runpod-lifecycle @ file:../runpod-lifecycle`. Keep developer extras separate in `pyproject.toml` under a `dev` extra containing at least `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, and `ipykernel`.

Add `env.example` alongside `pyproject.toml` and `README.md` so local runs and pod runs share one documented environment surface from the start. After Step 0 lands in the sibling package, include a one-line verification probe in the setup notes to confirm the imported dependency exposes the public SSH accessor before any infra work begins: `python -c "from runpod_lifecycle.pod import Pod; print(hasattr(Pod, 'open_ssh_client'))"`.

### Step 2: Centralize config models in `src/text_ip_adapter/config.py`

Create `src/text_ip_adapter/config.py` as the single source of truth for experiment configuration and stage YAML parsing. The config model set should cover `ModelConfig`, `AdapterConfig`, `DataConfig`, `TrainingConfig`, `EvalConfig`, `LoggingConfig`, `RunPodConfig`, and top-level `ExperimentConfig`, with stage entry files in `configs/stage1_frozen.yaml`, `configs/stage2_lora.yaml`, and `configs/stage3_gating.yaml`. The load-bearing invariant from the overview must be enforced here with a Pydantic root validator on the adapter settings, implemented in v2 style if needed, so `num_prefix_tokens == num_queries` is rejected at config-load time rather than discovered later in tensor-shape failures.

Back the config contract with `tests/test_config.py`. That test module should verify the root validator rejects mismatched `num_prefix_tokens` and `num_queries`, confirm the stage YAML files round-trip into `ExperimentConfig`, and lock down the default `P = 16` path so later stage-specific overrides do not silently drift.

### Step 3: Scaffold project layout, `README.md`, and `Makefile`

Establish the initial repository layout up front so every later phase has a stable home: `src/text_ip_adapter/`, `src/text_ip_adapter/data/`, `src/text_ip_adapter/model/`, `src/text_ip_adapter/train/`, `src/text_ip_adapter/eval/`, `src/text_ip_adapter/infra/`, `scripts/`, `configs/`, `tests/`, `tests/fixtures/`, `data/pairs/`, `data/raw/`, `data/processed/`, `checkpoints/`, and `eval_reports/`. Keep `README.md` focused on the architecture sketch, repo map, quickstart, and stage progression, while `Makefile` owns the common entrypoints so the repo can be driven consistently from local smoke runs through pod launches.

The first `Makefile` target set should be declared explicitly in `Makefile` and documented in `README.md`: `install`, `format`, `lint`, `typecheck`, `test`, `smoke-train`, `build-pairs`, `gen-instructions`, `train-local`, `eval-local`, `train-runpod`, `sync-pairs`, and `clean`. This keeps phase boundaries legible and gives later steps concrete command names to reference instead of inventing one-off shell snippets.

## Phase 2: Data Sourcing, Pairing, and Licensing

### Step 4: Define source manifest and license policy in `src/text_ip_adapter/data/sources.py`

Centralize source metadata in `src/text_ip_adapter/data/sources.py` and `data/sources.yaml`, with review guidance captured in `docs/licensing.md`. The source manifest should enumerate a small, explicit license taxonomy such as `public_domain`, `cc_by`, `cc_by_sa`, `fair_use_review`, and `reject`, because later pairing and training steps need machine-readable filtering rather than free-text notes. Seed the manifest with public-domain-first sources: Project Gutenberg poetry and prose, Chronicling America articles, and Miller Center or LibriVox speech/transcript material as the default ingest pool; keep Poetry Foundation, `imsdb`, and modern journalism entries classified as `fair_use_review` pending case-by-case approval.

Make the public-domain-first posture enforceable, not aspirational. Any non-public-domain entry in `data/sources.yaml` must require the operator to set `ALLOW_REVIEW_SOURCES=1` before ingestion scripts accept it, and `docs/licensing.md` should spell out that the default training path excludes review-gated material unless an explicit human decision has been recorded.

### Step 5: Normalize raw text into `data/processed/documents.parquet`

Implement ingestion in `src/text_ip_adapter/data/ingest.py` and drive it from `scripts/fetch_sources.py`, with normalized outputs written to `data/processed/documents.parquet`. The normalized schema should preserve the fields later stages need for grouping and filtering: `doc_id`, `source_id`, `license_class`, `author_id`, `author_name`, `work_id`, `title`, `url`, `published_at`, `content`, `token_count`, and `language`. Keep the content normalization simple and reproducible: HTML or transcript extraction first, `ftfy` cleanup second, whitespace cleanup third, and final token-length filtering to keep only documents between 200 and 8000 tokens.

This stage should aggressively reject malformed records rather than pass them downstream. The objective is a clean document table that later dedup and pairing passes can trust, not a best-effort archive of every scraped artifact.

### Step 6: Deduplicate documents and apply layer 1 of the leak firewall in `src/text_ip_adapter/data/dedup.py`

Run document-level near-duplicate detection in `src/text_ip_adapter/data/dedup.py`, with regression coverage in `tests/test_dedup.py`. The first pass should use MinHash-LSH at document level with Jaccard threshold `0.8` to collapse mirrored copies, syndicated reposts, and duplicate transcripts before pair generation. The second pass is the first layer of the three-layer content-leak firewall: paragraph-level overlap screening with Jaccard threshold `0.3`, so any candidate pair that shares too much literal paragraph content is rejected before training data is formed.

Call this out explicitly in `tests/test_dedup.py`: the phase is not just about cleanliness, it is the first barrier against style-transfer examples degenerating into reference-copy examples.

### Step 7: Build matched-style pairs and author-stratified splits in `src/text_ip_adapter/data/pairing.py`

Construct `(reference, target)` examples in `src/text_ip_adapter/data/pairing.py` and expose them via `scripts/build_pairs.py`, with split correctness asserted in `tests/test_pairing.py`. Pair only within the same author or speaker bucket, but across distinct source elements: poem-to-poem, article-to-article, scene-to-scene, speech-to-speech. Use author-stratified train/val/test splits, not pair-stratified splits, and make `tests/test_pairing.py` assert zero author overlap across those splits so the model is evaluated on unseen authors rather than held-out examples from known authors.

Cap pair fanout with `max_pairs_per_author=8` to avoid prolific authors dominating the dataset and target an initial scale near `50k` training pairs, `2k` validation pairs, and `2k` test pairs. This is the main structural bet behind the project’s novelty: matched-style examples with content variation, rather than arbitrary prompt-completion style imitation data.

### Step 8: Generate target-only instructions in `src/text_ip_adapter/data/instructions.py`

Implement instruction synthesis in `src/text_ip_adapter/data/instructions.py` and drive it from `scripts/gen_instructions.py`, with `scripts/inspect_pairs.py` reserved for manual spot checks. Instruction generation must look only at the target text, never the reference text, because that is layer 2 of the three-layer leak firewall. Use `INSTRUCTION_GEN_MODEL` to switch between `claude-sonnet-4-6` and `gemma-2-9b-it` so the plan supports both API-backed and local generation modes without changing pipeline code.

Add a 5-gram rejection pass before accepting a generated instruction as part of the dataset. If the instruction overlaps too directly with the reference or target text at the 5-gram level, reject and regenerate; this serves as layer 3 at data time, complementing the later evaluation-time leak metric in Phase 5. Keep the instructions semantically about content goals, constraints, or topics, not stylistic labels copied from the reference.

### Step 9: Materialize training examples in `src/text_ip_adapter/data/dataset.py`

Finalize the data interface in `src/text_ip_adapter/data/dataset.py`. The dataset and collator should emit tokenized `reference_input_ids`, `reference_attention_mask`, `prompt_input_ids`, `prompt_attention_mask`, `target_input_ids`, `target_attention_mask`, and a `loss_mask` that restricts optimization to the target span. Keep max lengths explicit in config-backed fields so reference and target truncation are deliberate, for example separate caps for reference text, prompt text, and target text rather than one global sequence budget.

The output of this step is the stable, training-ready pair artifact under `data/pairs/`, with enough metadata retained to support per-source-category evaluation later without reopening the raw-ingestion pipeline.

## Phase 3: Model Architecture

### Step 10: Build the reference encoder in `src/text_ip_adapter/model/encoder.py`

Implement the reference encoder in `src/text_ip_adapter/model/encoder.py` as a frozen Gemma text backbone followed by a compact 2-layer perceiver-resampler. The Gemma forward pass should consume the reference text only, keep all base weights frozen, and hand the final hidden states to a learned resampler with `num_queries` learned latent queries that compress the reference into `P` style vectors. This keeps the dense conditioning signal small enough to inject repeatedly across layers without turning the project into full-model finetuning.

Make encoder output caching optional at the file-system level so repeated evaluation sweeps can reuse reference latents when the reference set is fixed. The cache is an optimization, not part of the correctness contract, so the uncached path remains the default truth path.

### Step 11: Project latents into GQA-aware prefix tensors in `src/text_ip_adapter/model/projector.py`

Implement the adapter projector in `src/text_ip_adapter/model/projector.py`, with shape and regression coverage in `tests/test_injection_shapes.py`. The projector must map the encoder’s `P` latent vectors into per-layer prefix K/V tensors with grouped-query-attention awareness, using shape `(P, num_kv_heads, head_dim)` rather than anything derived from the full query-head count. `tests/test_injection_shapes.py` should assert this contract for both Gemma-2-2B and Gemma-2-9B so head-layout differences cannot silently break the hook path.

Use a shared MLP trunk feeding per-layer linear heads, and zero-initialize the projector output heads so the system starts as a no-op relative to the frozen base model. Keep the trainable budget below `5%` of the base parameter count, and default the injection layer range to `[N/2, N-2]` so the adapter influences style in mid-to-upper blocks without perturbing the entire stack.

### Step 12: Inject prefix K/V states in `src/text_ip_adapter/model/injection.py`

Implement the runtime hook logic in `src/text_ip_adapter/model/injection.py`, with correctness pinned by `tests/test_injection_shapes.py` and `tests/test_generation.py`. The hook should concatenate learned prefix K/V tensors with content K/V tensors along the sequence dimension, extend the attention mask accordingly, and ensure prefix positions remain unmasked for every generated token. This is where the GQA shape contract becomes operational, so the injection code must consume `num_kv_heads`-aligned prefix tensors directly rather than reshaping them on the fly.

Keep RoPE handling explicit: rotary position embedding is applied only to content keys, while prefix keys remain un-rotated because they represent learned memory slots rather than sequence-positioned tokens. `tests/test_injection_shapes.py` should assert that prefix K is not RoPE-rotated, and `tests/test_generation.py` should assert that the prefix survives into the KV cache on generation step `>= 2` rather than being dropped after the first decode call.

### Step 13: Wrap the full model in `src/text_ip_adapter/model/adapter_model.py`

Assemble the end-to-end interface in `src/text_ip_adapter/model/adapter_model.py`. This wrapper owns freezing the Gemma base model, wiring together `src/text_ip_adapter/model/encoder.py`, `src/text_ip_adapter/model/projector.py`, and `src/text_ip_adapter/model/injection.py`, and asserting `trainable_param_count` at startup so the run fails early if an unexpected parameter group becomes trainable. Expose both `forward(...)` and `generate(...)` APIs from the wrapper, because training, evaluation, and CLI inference all need one coherent surface rather than bespoke entrypoints for each phase.

The wrapper should make the architecture boundaries obvious: reference path in, `P` latent vectors out, per-layer prefix K/V projection, then injected generation with a frozen base. That clarity matters because later stages add LoRA and per-layer gating on top of this exact interface rather than redefining it.

## Phase 4: Training Loop

### Step 14: Implement the main trainer in `src/text_ip_adapter/train/loop.py`

Implement the core optimization loop in `src/text_ip_adapter/train/loop.py` and expose the runnable entrypoint in `scripts/train.py`. The training stack should stay simple: `accelerate` plus bare `torch`, next-token cross-entropy computed on the target span only, and no trainer abstraction that obscures the custom prefix-injection path. Use `AdamW` with learning rates split by component: `projector 1e-4`, `encoder-queries 5e-5`, `weight_decay=0.01`, `warmup_steps=500`, then cosine decay. Run the default Stage 1 configuration in `bf16` mixed precision with an effective batch size of `64` on one `4090`-class GPU.

Make observability part of the contract, not an optional extra. `wandb` is the primary live telemetry surface and should record at least `loss`, `grad_norm`, `learning_rate`, `prefix_norm`, `trainable_param_count`, per-category validation loss, and GPU utilization. The `trainable_param_count` assertion from `src/text_ip_adapter/model/adapter_model.py` should be surfaced at startup, and `prefix_norm` should be tracked explicitly because growth from near-zero is the fastest sanity signal that the adapter path is learning anything at all.

Checkpoint only the trainable parameter set, not the frozen Gemma weights, and write checkpoints every `2000` steps plus on validation improvement. That keeps artifact sizes compatible with the pod workflow and avoids paying storage or sync costs for unchanged base-model weights.

### Step 15: Add a smoke path in `scripts/smoke_train.py`

Create a cheap preflight training path in `scripts/smoke_train.py`, backed by `tests/fixtures/toy_pairs.jsonl`. The smoke dataset should contain `100` hand-constructed `(reference, instruction, target)` triples sized to exercise the full forward and generation path without depending on external data prep. The smoke run should execute `200` training steps on CPU or MPS in under `5` minutes and must run before any paid pod hour is started.

The smoke pass criteria are intentionally narrow and observable: training loss should drop over the run, and `prefix_norm` should grow from approximately zero. If either signal is absent, the pipeline is not ready for a real GPU run regardless of whether the process exits cleanly.

## Phase 5: Evaluation Harness

### Step 16: Define core metrics in `src/text_ip_adapter/eval/metrics.py`

Implement the evaluation metric layer in `src/text_ip_adapter/eval/metrics.py`. The style pillar should use `mpnet` embeddings plus a `LogisticRegression` classifier trained on author labels, combined with cosine similarity against the reference text so the evaluation captures both coarse style attribution and local reference alignment. The content-preservation pillar should pair an LLM-as-judge rubric with `sacrebleu` and `rouge` so instruction fidelity is measured both semantically and lexically instead of relying on one weak proxy.

Leak detection is the third evaluation pillar and also the evaluation-time closure of the content-leak firewall: compute longest-common-subsequence over reference length, explicit 5-gram overlap, and an LLM judge score for whether generated content copies or closely paraphrases the reference. The prompting baseline should be evaluated side by side with the adapter using the explicit template `Write in the style of: <ref>\n\nInstruction: ...`, because the adapter only earns its keep if it beats direct prompting on the combined style, content, and leak criteria.

### Step 17: Build the evaluation runner in `src/text_ip_adapter/eval/runner.py`

Implement orchestration in `src/text_ip_adapter/eval/runner.py` and expose it through `scripts/eval.py`. Each run should write structured outputs to `eval_reports/<run_name>/report.json` and a human-readable summary to `eval_reports/<run_name>/report.md`. The runner should include the continuous-control probe directly rather than treat it as an optional appendix: evaluate `alpha` mixtures at `0`, `0.25`, `0.5`, `0.75`, and `1.0`, then compute a Spearman correlation target of `ρ >= 0.7` between intended alpha progression and measured style shift.

This phase is where the project starts proving novelty instead of just producing text. The evaluation runner therefore needs to emit both raw scores and pass/fail gate summaries, because later stage decisions depend on thresholds rather than narrative interpretation.

### Step 18: Add the reference-swap probe in `src/text_ip_adapter/eval/ref_swap.py`

Implement the dedicated swap analysis in `src/text_ip_adapter/eval/ref_swap.py`, with a runnable wrapper in `scripts/eval_ref_swap.py`, and persist structured outputs to `eval_reports/<run_name>/ref_swap.json`. For each instruction, evaluate `10` different references while holding the instruction constant. The expected outcome is stable content with variation in style: content metrics should remain within `± 0.05`, while style variance should be at least `2x` the no-reference ablation.

Keep this probe distinct from the generic report because it is the clearest empirical test of the adapter’s purpose. If swapping the reference does not move the style metric while preserving instruction content, the whole method has failed regardless of average validation loss.

### Step 19: Ship the inference CLI in `scripts/infer.py`

Provide a user-facing inference surface in `scripts/infer.py`, with smoke coverage in `tests/test_infer_cli.py`. The CLI contract should include `--checkpoint`, `--reference`, `--instruction`, `--alpha-mix`, and `--max-tokens`, and it should support multi-reference mixing by allowing repeated `--reference` and `--alpha-mix` flags in one invocation. This keeps the public artifact aligned with the continuous-control and style-mixing claims made elsewhere in the plan.

`tests/test_infer_cli.py` should exercise the smoke checkpoint path and assert that the command returns non-empty output for a valid `(reference, instruction)` input. The point of this test is not output quality; it is proving that the saved trainable-params checkpoint can be loaded end to end through the public interface.

## Phase 6: Infra — RunPod, Artifact Sync, wandb

### Step 20: Implement SSH/SCP transport in `src/text_ip_adapter/infra/ssh_sync.py`

Implement the transport layer in `src/text_ip_adapter/infra/ssh_sync.py`, with behavioral coverage in `tests/test_ssh_sync.py`. The public helpers should all be `async def`, and the `Pod.open_ssh_client()` call should be bridged into the async world with `asyncio.to_thread(...)`. Keep the local-auth fallback wording precise: `_build_client_from_env` should read `RUNPOD_SSH_PRIVATE_KEY_PATH` and `RUNPOD_SSH_PRIVATE_KEY` first, both of which are already consumed by `RunPodConfig.from_env()`, and then `RUNPOD_SSH_PASSWORD`, which is read directly from environment and is not yet parsed by `RunPodConfig.from_env()`. That fallback path must remain gated behind `TEXT_IP_ADAPTER_SSH_LOCAL_AUTH=1`.

The transport lifecycle should be short-lived per call, matching the existing spirit of `pod.exec_ssh()`: connect, tar and `scp`, untar or copy, then disconnect. Use `paramiko.SSHClient` plus `scp.SCPClient(ssh.get_transport())` for `upload_path(...)` and `download_path(...)`, and do not imply any long-lived persistent session. The one load-bearing primitive is `sync_project_root(pod, repo_root, remote_root='/workspace')`, which atomically enforces the Remote-workspace contract by hashing the synced project payload and storing the gate at `/workspace/.project_sync_hash`. Keep `data/pairs/` out of that primitive and handle it with a separate hash-gated `sync_dataset(...)` path.

`tail_remote_log(...)` should be explicitly polled on a 5-second cadence, not described as streamed output, because `exec_ssh` is buffered and `wandb` is the primary live telemetry surface. `tests/test_ssh_sync.py` should cover round-trip upload and download, the Remote-workspace contract test that required paths are present and excluded paths absent, `tail_remote_log(...)` shutdown on `stop_event`, and the fallback auth ladder.

### Step 21: Verify storage prerequisites in `scripts/runpod_bootstrap.py`

Implement the storage bootstrap check in `scripts/runpod_bootstrap.py`. This script should be an async entrypoint that queries `get_network_volumes(config.api_key)` and verifies that the named storage target exists before training starts. It must not auto-create a network volume, because size, region, and cost are operator decisions rather than safe defaults.

This step is deliberately boring but necessary: fail fast if the expected storage target is absent, instead of discovering the problem after a pod is already running.

### Step 22: Orchestrate pod training in `src/text_ip_adapter/infra/runpod_runner.py`

Implement the launcher in `src/text_ip_adapter/infra/runpod_runner.py`, with a thin script wrapper in `scripts/train_runpod.py` and ordering coverage in `tests/test_runpod_runner.py`. The critical contract is ordering: `sync_project_root(...)` must run before `sync_dataset(...)`, which must run before any remote `pip install -e .[dev]`, which must run before `accelerate launch scripts/train.py --config configs/stage1_frozen.yaml`. Persist `pod.id` to `checkpoints/<run>/pod_id.txt`, poll logs through `tail_remote_log(...)`, wait on the remote process, download outputs, and always terminate the pod in `finally:`.

The launcher shape should be captured directly in the plan:

```python
import asyncio

from runpod_lifecycle import launch
from runpod_lifecycle.config import RunPodConfig

from text_ip_adapter.infra.runpod_runner import (
    download_path,
    sync_dataset,
    sync_project_root,
    tail_remote_log,
    wait_for_remote_pid_exit,
)


async def main() -> None:
    config = RunPodConfig.from_env(storage_name="text-ip-adapter")
    pod = await launch(config, hooks=None)
    await pod.wait_ready(timeout=900)
    await sync_project_root(pod, repo_root=".", remote_root="/workspace")
    await sync_dataset(pod, local_dir="data/pairs", remote_dir="/workspace/data/pairs")
    await pod.exec_ssh("cd /workspace && pip install -e .[dev]")
    await pod.exec_ssh(
        "cd /workspace && accelerate launch scripts/train.py --config configs/stage1_frozen.yaml"
    )
    # persist pod.id to checkpoints/<run>/pod_id.txt
    log_task = asyncio.create_task(tail_remote_log(pod, remote_path="/workspace/train.log"))
    try:
        await wait_for_remote_pid_exit(pod, remote_pid_path="/workspace/train.pid")
        await download_path(pod, remote_path="/workspace/checkpoints", local_path="checkpoints")
    finally:
        log_task.cancel()
        await pod.terminate()


if __name__ == "__main__":
    asyncio.run(main())
```

`tests/test_runpod_runner.py` should assert the ordering above and verify that termination still occurs on exception. The same test should also ensure that `sync_project_root(...)` happens before any remote install or train command is dispatched.

### Step 23: Add orphan cleanup in `scripts/runpod_sweep.py`

Implement the cleanup sweep in `scripts/runpod_sweep.py` as an async entrypoint. Load `known_pod_ids` from `checkpoints/*/pod_id.txt`, treating an empty set as valid, then call `find_orphans(api_key, known_pod_ids, name_prefix=..., older_than_seconds=8*3600)` with `known_pod_ids` passed positionally rather than by keyword. Require an explicit `--yes` flag before any termination action is allowed.

This script is a safety valve, not a convenience shortcut. The goal is to reduce lingering-cost mistakes while keeping destructive cleanup gated behind explicit intent.

## Phase 7: Staged Training Roadmap

### Step 24: Stage 1 — frozen base in `configs/stage1_frozen.yaml`

Use `configs/stage1_frozen.yaml` as the default Stage 1 configuration and record outcomes in `docs/stage1_results.md`. This stage keeps the Gemma base frozen, fixes `P = num_queries = 16`, injects over the default layer range `[N/2, N-2]`, and targets an initial run budget of one `4090` or `A40` class GPU for roughly `24` hours over `50k` pairs and `2` epochs. Stage 1 is the proof-of-concept stage; it should answer whether the dense reference-conditioning path works at all before any extra trainable capacity is introduced.

The Stage 1 gate is explicitly three-pronged and all three conditions must pass before moving on: the adapter must beat the direct prompting baseline on the combined style, content, and leak criteria; it must hit the continuous-control target of `Spearman ρ >= 0.7`; and the reference-swap probe must show style variance inside the expected bands while keeping content stable. If any one of those fails, Stage 2 does not start.

### Step 25: Stage 2 — upper-layer LoRA in `configs/stage2_lora.yaml`

Use `configs/stage2_lora.yaml` for the optional second stage and record results in `docs/stage2_results.md`. This stage initializes from the best Stage 1 checkpoint and adds `peft` LoRA adapters with rank `8` to `16` on `q/k/v/o` projections in the upper half of the transformer. The objective is not to rescue a broken Stage 1; it is to test whether a modest amount of extra capacity improves controllable style transfer once the prefix-conditioning path already works.

Stage 2 is strictly conditional on Stage 1 success. Its gate is also explicit: style quality must increase by at least `0.05`, content preservation must not drop by more than `0.03`, and leak metrics must not worsen. If those conditions are not met, the roadmap stops here rather than rolling forward by inertia.

### Step 26: Stage 3 — per-layer gating and style mixing in `configs/stage3_gating.yaml`

Use `configs/stage3_gating.yaml` for the final stage and write the outcome summary to `docs/stage3_results.md`. This stage adds per-layer sigmoid gates initialized near `1.0` and extends the adapter to multi-reference mixing through learned attention over reference latents. The point is to move from “reference-conditioned style transfer works” to “style can be blended and steered continuously across multiple references.”

Stage 3 only runs if Stage 2 hits its gate. The evaluation extension here is the multi-reference probe: the learned mixing path must preserve the Stage 1 and Stage 2 gains while enabling predictable composite-style control, rather than regressing into unstable interpolation.

## Execution Order

1. Complete `## Phase 0` first so `../runpod-lifecycle/src/runpod_lifecycle/pod.py` exposes the public `open_ssh_client()` contract before any downstream infra code depends on it.
2. Finish `## Phase 1` end-to-end next, since `pyproject.toml`, `src/text_ip_adapter/config.py`, `configs/stage1_frozen.yaml`, `configs/stage2_lora.yaml`, and `configs/stage3_gating.yaml` define the dependency, config, and scaffold surface that every later phase consumes.
3. Run `## Phase 2` before `## Phase 3` so `src/text_ip_adapter/data/dataset.py` and the paired-data artifacts exist before wiring the adapter in `src/text_ip_adapter/model/adapter_model.py`.
4. Interleave `## Phase 3` and `## Phase 4` only after the model interfaces stabilize, but require `### Step 15` in `scripts/smoke_train.py` to pass before spending any paid pod hour on `scripts/train_runpod.py`.
5. Finish `## Phase 5` before the full `## Phase 6` rollout so evaluation in `src/text_ip_adapter/eval/runner.py` and `scripts/eval.py` is ready before remote training runs produce checkpoints.
6. Inside `## Phase 6`, implement `### Step 20` in `src/text_ip_adapter/infra/ssh_sync.py` before `### Step 22` in `src/text_ip_adapter/infra/runpod_runner.py`, because the runner must call `sync_project_root(...)` and `sync_dataset(...)` from the finalized transport layer rather than inventing its own copy logic.
7. Execute `## Phase 7` strictly sequentially: Stage 1 in `configs/stage1_frozen.yaml`, then Stage 2 in `configs/stage2_lora.yaml` only if Stage 1 clears its gate, then Stage 3 in `configs/stage3_gating.yaml` only if Stage 2 clears its gate.

## Validation Order

1. Start with the unit tests that lock the paired-data contract: `tests/test_pairing.py` must assert zero author overlap across train, val, and test, and `tests/test_dedup.py` must verify the paragraph-level Jaccard firewall behavior.
2. Validate shape and generation invariants next: `tests/test_injection_shapes.py` must cover GQA-aware prefix shapes, unrotated prefix K, and the extended attention mask, while `tests/test_generation.py` must prove the prefix stays in cache on generation step `>= 2` and that generation differs with and without the prefix path enabled.
3. Validate configuration constraints in `tests/test_config.py`, specifically that the root validator rejects any mismatch between `num_prefix_tokens` and `num_queries`.
4. Validate the infra contract locally before any pod run: `tests/test_ssh_sync.py` must cover round-trip upload/download, the `sync_project_root(...)` required-vs-excluded contract, `tail_remote_log(...)` exiting on `stop_event`, and the fallback auth ladder.
5. Validate the runner ordering next: `tests/test_runpod_runner.py` must assert `sync_project_root(...)` runs before any remote `pip install -e .[dev]` or `accelerate launch scripts/train.py --config configs/stage1_frozen.yaml`, and must also verify `pod.terminate()` still runs on failure.
6. Validate inference usability with `tests/test_infer_cli.py`, which should run end-to-end against the smoke checkpoint produced by `scripts/infer.py` and assert that the CLI emits non-empty text.
7. Run the local smoke training path in `scripts/smoke_train.py` with `tests/fixtures/toy_pairs.jsonl` and require both signals from `### Step 15`: loss decreases and `prefix_norm` rises above zero.
8. Run a local-mini evaluation on a tiny split using `scripts/eval.py` and `src/text_ip_adapter/eval/runner.py` to catch metric, prompt-baseline, and leak-firewall regressions before touching remote infrastructure.
9. Run the first pod-backed training pass through `scripts/train_runpod.py` at roughly `10%` of Stage 1 steps and compare its metrics and artifacts against the local-mini run before scaling up.
10. Run the full Stage 1 evaluation suite against all four brief criteria: reference swaps change style, instruction content is preserved, reference content does not leak, and alpha blending produces continuous control. Only proceed if the `configs/stage1_frozen.yaml` run also clears the Stage 1 gate documented in `docs/stage1_results.md`.
11. Gate Stage 2 and Stage 3 on prior metrics only: `configs/stage2_lora.yaml` may run only after Stage 1 passes, and `configs/stage3_gating.yaml` may run only after Stage 2 passes its style/content/leak thresholds and preserves the earlier gains.

## Settled Decisions

- **The planning artifact lives at `docs/megaplan.md`.** **Rationale:** The document itself is the execution contract for this metaplan-mode run and is the file downstream planning flows can import.
- **`runpod-lifecycle` remains an external sibling dependency, and Step 0 is the only upstream contribution.** **Rationale:** Pod orchestration should be reused, but all research-specific logic stays in `src/text_ip_adapter/` so model iteration does not sprawl into the sibling package.
- **`Pod.open_ssh_client()` is the primary SSH accessor, with `TEXT_IP_ADAPTER_SSH_LOCAL_AUTH=1` as the gated fallback path.** **Rationale:** The transport layer needs one sanctioned entrypoint for authenticated SSH/SCP access, while the fallback remains explicit and opt-in.
- **Async infra entrypoints are consumed through `asyncio.run(main())`, and orphan cleanup calls `find_orphans(api_key, known_pod_ids, ...)` with positional `known_pod_ids`.** **Rationale:** This matches the intended RunPod integration surface and avoids inventing alternate lifecycle conventions.
- **The Remote-workspace contract is enforced by a single `sync_project_root(...)` primitive, with `data/pairs/` synchronized separately.** **Rationale:** One atomic project sync boundary keeps remote state reproducible, while dataset transfer remains independently hash-gated and easier to reason about.
- **`wandb` is the primary live telemetry channel, and `tail_remote_log(...)` is only a polled secondary surface.** **Rationale:** `exec_ssh` is buffered rather than streamed, so live observability must rely on metrics infrastructure first and log polling second.
- **Network-volume bootstrap is manual verification only and must not auto-create storage.** **Rationale:** Region, size, and cost choices are operational decisions that should remain explicit rather than hidden behind convenience automation.
- **`num_prefix_tokens == num_queries` is mandatory, with default `P = 16`, and the invariant is enforced by the Pydantic root validator.** **Rationale:** The encoder, projector, and injector all rely on one shared prefix cardinality, so mismatches should fail fast at config load time.
- **Gemma-2-2B is the default Stage 1 base model.** **Rationale:** It is the smallest base large enough to validate the reference-conditioning idea while keeping the initial training budget tractable.
- **Prefix K/V tensors are GQA-aware, prefix K remains un-rotated under RoPE, and projector outputs start from zero initialization.** **Rationale:** Those three choices make the adapter mathematically compatible with Gemma attention internals while minimizing destabilizing interference at initialization.
- **Author-stratified splits and the three-layer content-leak firewall are mandatory.** **Rationale:** Zero author overlap plus paragraph filtering, target-only instruction synthesis, and evaluation-time leak checks are the core safeguards against fake style transfer driven by memorized overlap.
- **The data posture is public-domain-first, with `ALLOW_REVIEW_SOURCES=1` gating any per-source fair-use review exceptions.** **Rationale:** Licensing risk should default to the conservative path, and any exception needs an explicit operator decision.
- **Synthetic instructions are generated from the target text only.** **Rationale:** Target-only synthesis preserves the reference-as-style-signal framing and prevents the instruction channel from leaking reference content into supervision.
- **Training uses `accelerate` plus bare `torch`, and checkpoints save trainable parameters only.** **Rationale:** The runtime stack stays minimal and transparent, while checkpoint size and portability stay aligned with a frozen-base adapter workflow.
- **Evaluation must beat the explicit prompting baseline.** **Rationale:** If the adapter does not outperform direct prompting with the reference text, the extra architecture and training complexity are not justified.
- **`scripts/infer.py` is a required public artifact.** **Rationale:** The project needs one concrete inference surface that loads saved checkpoints and exposes reference-conditioned generation outside the training loop.
- **The staged roadmap is strictly sequential: Stage 2 only follows a successful Stage 1, and Stage 3 only follows a successful Stage 2.** **Rationale:** Later-stage capacity and mixing work should refine a validated mechanism, not obscure whether the earlier stage actually solved the problem.
