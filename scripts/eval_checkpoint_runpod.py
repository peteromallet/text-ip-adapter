#!/usr/bin/env python3
"""Run checkpoint-only probe evaluation on a fresh RunPod and terminate it."""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from pathlib import Path

from runpod_lifecycle import launch

from text_ip_adapter.config import load_experiment_config
from text_ip_adapter.infra.runpod_runner import (
    _configured_dataset_dirs,
    _sync_hf_token,
    load_runpod_config,
    verify_storage,
)
from text_ip_adapter.infra.ssh_sync import download_path, sync_dataset, sync_project_root


VARIANTS = {
    "greedy_no_repeat": [
        "--repetition-penalty", "1.15",
        "--no-repeat-ngram-size", "3",
    ],
    "sampled_rep": [
        "--do-sample",
        "--temperature", "0.8",
        "--top-p", "0.9",
        "--repetition-penalty", "1.12",
        "--no-repeat-ngram-size", "3",
    ],
}


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run eval_from_checkpoint on RunPod without training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--local-output-dir", required=True)
    parser.add_argument("--remote-output-dir", default="/workspace/text-ip-adapter/eval_runs/checkpoint_eval")
    parser.add_argument("--variants", default="greedy_no_repeat,sampled_rep")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--step-tag", type=int, default=1500)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    experiment = load_experiment_config(repo_root / args.config)
    runpod_config = load_runpod_config(repo_root, storage_name=experiment.runpod.storage_name)
    storage_info = await verify_storage(runpod_config)

    pod = await launch(runpod_config, name="text-ip-adapter-eval")
    manifest: dict[str, object] = {
        "pod_id": pod.id,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "probe_path": args.probe_path,
        "val_path": args.val_path,
        "variants": args.variants.split(","),
        "storage": storage_info,
        "status": "launched",
    }
    local_output_dir = Path(args.local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    (local_output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        await pod.wait_ready(timeout=900)
        manifest["status"] = "ready"
        manifest["hf_token_synced"] = await _sync_hf_token(pod)
        await sync_project_root(pod, repo_root=str(repo_root), remote_root=experiment.runpod.remote_root)
        for local_dataset_dir in _configured_dataset_dirs(repo_root, experiment):
            rel_dir = local_dataset_dir.relative_to(repo_root)
            await sync_dataset(
                pod,
                local_dir=str(local_dataset_dir),
                remote_dir=f"{experiment.runpod.remote_root}/{rel_dir.as_posix()}",
            )

        remote_root = experiment.runpod.remote_root
        await pod.exec_ssh(f"cd {shlex.quote(remote_root)} && PYTHONPATH=src python -m pip install -e .", timeout=1800)
        await pod.exec_ssh(f"mkdir -p {shlex.quote(args.remote_output_dir)}", timeout=60)

        for variant in manifest["variants"]:
            if variant not in VARIANTS:
                raise ValueError(f"unknown variant {variant!r}; options: {sorted(VARIANTS)}")
            remote_variant_dir = f"{args.remote_output_dir.rstrip('/')}/{variant}"
            remote_samples = f"{remote_variant_dir}/samples.jsonl"
            await pod.exec_ssh(f"mkdir -p {shlex.quote(remote_variant_dir)}", timeout=60)
            cmd_parts = [
                "PYTHONPATH=src",
                "python", "scripts/eval_from_checkpoint.py",
                "--checkpoint", args.checkpoint,
                "--config", args.config,
                "--val-path", args.val_path,
                "--probe-path", args.probe_path,
                "--output", remote_samples,
                "--max-new-tokens", str(args.max_new_tokens),
                "--step-tag", str(args.step_tag),
                *VARIANTS[variant],
            ]
            cmd = " ".join(shlex.quote(part) for part in cmd_parts)
            exit_code, stdout, stderr = await pod.exec_ssh(
                f"cd {shlex.quote(remote_root)} && {cmd}",
                timeout=3600,
            )
            variant_dir = local_output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (variant_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            if exit_code != 0:
                raise RuntimeError(f"{variant} eval failed with exit {exit_code}: {stderr or stdout}")

        await download_path(pod, args.remote_output_dir, str(local_output_dir))
        manifest["status"] = "completed"
        return 0
    finally:
        try:
            await pod.terminate()
            manifest["terminated"] = True
        except Exception as exc:  # noqa: BLE001
            manifest["terminated"] = False
            manifest["terminate_error"] = str(exc)
        (local_output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
