#!/usr/bin/env python3
"""Eval the local 022 checkpoint on v3.9 eval-clean probes using RunPod /tmp."""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from runpod_lifecycle import launch

from text_ip_adapter.infra.runpod_runner import _sync_hf_token, load_runpod_config, verify_storage
from text_ip_adapter.infra.ssh_sync import download_path, sync_dataset, sync_project_root


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


async def upload_file(pod, local_path: Path, remote_path: str) -> None:
    ssh = await asyncio.to_thread(pod.open_ssh_client)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()
    finally:
        ssh.close()


async def run_checked(pod, cmd: str, *, timeout: int, label: str, out_dir: Path) -> None:
    exit_code, stdout, stderr = await pod.exec_ssh(cmd, timeout=timeout)
    (out_dir / f"{label}.stdout.log").write_text(stdout, encoding="utf-8")
    (out_dir / f"{label}.stderr.log").write_text(stderr, encoding="utf-8")
    if exit_code != 0:
        raise RuntimeError(f"{label} failed with exit {exit_code}: {stderr or stdout}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/pairs_v3_9_core2_evalclean")
    parser.add_argument("--config", default="configs/stage1_v3_8_core2_cleanheldout_restart.yaml")
    parser.add_argument(
        "--checkpoint",
        default="../prompt-adapters/experiments/2026-05-text-022-broader-clean-split-restart/results/runpod_tmp/train_artifacts/tmp/stage1_v3_8_core2_cleanheldout_restart/final.pt",
    )
    parser.add_argument("--local-output-dir", default="../prompt-adapters/experiments/2026-05-text-023-evalclean-probe-audit/results/runpod_v39_eval")
    parser.add_argument("--remote-project-root", default="/tmp/text-ip-adapter-v39-eval")
    parser.add_argument("--workspace-root", default="/workspace/text-ip-adapter")
    parser.add_argument("--remote-output-dir", default="/tmp/exp023_v39_eval")
    parser.add_argument("--remote-checkpoint", default="/tmp/exp022_final.pt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    local_out = (repo_root / args.local_output_dir).resolve()
    local_out.mkdir(parents=True, exist_ok=True)
    manifest_path = local_out / "launch_manifest.json"
    checkpoint = (repo_root / args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    runpod_config = load_runpod_config(repo_root, storage_name="Peter")
    storage_info = await verify_storage(runpod_config)
    pod = await launch(runpod_config, name="text-ip-adapter-v39-eval")
    manifest = {
        "experiment_id": "2026-05-text-023-evalclean-probe-audit",
        "status": "launched",
        "launched_at": now(),
        "pod_id": pod.id,
        "checkpoint": str(checkpoint),
        "dataset_dir": args.dataset_dir,
        "remote_project_root": args.remote_project_root,
        "remote_output_dir": args.remote_output_dir,
        "storage": storage_info,
    }
    write_json(manifest_path, manifest)

    try:
        await pod.wait_ready(timeout=900)
        manifest["status"] = "ready"
        manifest["ready_at"] = now()
        manifest["hf_token_synced"] = await _sync_hf_token(pod)
        write_json(manifest_path, manifest)

        manifest["project_sync"] = await sync_project_root(pod, str(repo_root), remote_root=args.remote_project_root)
        manifest["dataset_sync"] = await sync_dataset(
            pod,
            str(repo_root / args.dataset_dir),
            f"{args.workspace_root}/{args.dataset_dir}",
        )
        write_json(manifest_path, manifest)

        q_project = shlex.quote(args.remote_project_root)
        q_workspace = shlex.quote(args.workspace_root)
        q_output = shlex.quote(args.remote_output_dir)
        q_config = shlex.quote(args.config)
        q_ckpt = shlex.quote(args.remote_checkpoint)
        probe_path = f"{args.dataset_dir}/probes_balanced.jsonl"
        val_path = f"{args.dataset_dir}/val.jsonl"
        q_probe = shlex.quote(probe_path)
        q_val = shlex.quote(val_path)

        await upload_file(pod, checkpoint, args.remote_checkpoint)
        await run_checked(
            pod,
            f"rm -rf {q_project}/data {q_project}/checkpoints && "
            f"ln -s {q_workspace}/data {q_project}/data && "
            f"ln -s {q_workspace}/checkpoints {q_project}/checkpoints && "
            f"mkdir -p {q_output}",
            timeout=120,
            label="prep",
            out_dir=local_out,
        )
        await run_checked(pod, f"cd {q_project} && PYTHONPATH=src python -m pip install -e .", timeout=1800, label="install", out_dir=local_out)
        await run_checked(
            pod,
            f"cd {q_project} && PYTHONPATH=src python -u scripts/probe_conditioning.py "
            f"--checkpoint {q_ckpt} --config {q_config} --probes {q_probe} "
            f"--output-dir {q_output}/pathway --n-probes 32 --max-new-tokens 80",
            timeout=3600,
            label="pathway",
            out_dir=local_out,
        )
        await run_checked(
            pod,
            f"cd {q_project} && PYTHONPATH=src python -u scripts/eval_from_checkpoint.py "
            f"--checkpoint {q_ckpt} --config {q_config} --val-path {q_val} --probe-path {q_probe} "
            f"--output {q_output}/sampled_rep/samples.jsonl --max-new-tokens 120 --step-tag 1000 "
            "--do-sample --temperature 0.8 --top-p 0.9 --repetition-penalty 1.12 --no-repeat-ngram-size 3",
            timeout=3600,
            label="sampled_eval",
            out_dir=local_out,
        )
        await download_path(pod, args.remote_output_dir, str(local_out / "eval_artifacts"))
        manifest["status"] = "completed"
        manifest["completed_at"] = now()
        return 0
    finally:
        try:
            await pod.terminate()
            manifest["terminated"] = True
            manifest["terminated_at"] = now()
        except Exception as exc:  # noqa: BLE001
            manifest["terminated"] = False
            manifest["terminate_error"] = repr(exc)
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
