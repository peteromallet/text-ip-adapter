#!/usr/bin/env python3
"""Run experiment 024 from /tmp on RunPod.

This is a cheap continuation smoke: upload the local 022 final checkpoint,
train 500 more steps with the style-triplet objective on v3.9, evaluate on the
same clean probes, download artifacts, and terminate the pod.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from datetime import datetime, timezone
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


async def _upload_file(pod, local_path: Path, remote_path: str) -> None:
    ssh = await asyncio.to_thread(pod.open_ssh_client)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()
    finally:
        ssh.close()


async def _run_checked(pod, cmd: str, *, timeout: int, label: str, local_dir: Path) -> tuple[str, str]:
    exit_code, stdout, stderr = await pod.exec_ssh(cmd, timeout=timeout)
    (local_dir / f"{label}.stdout.log").write_text(stdout, encoding="utf-8")
    (local_dir / f"{label}.stderr.log").write_text(stderr, encoding="utf-8")
    if exit_code != 0:
        raise RuntimeError(f"{label} failed with exit {exit_code}: {stderr or stdout}")
    return stdout, stderr


async def _run_remote_job(
    pod,
    cmd: str,
    *,
    timeout: int,
    label: str,
    local_dir: Path,
    remote_log_dir: str,
    poll_seconds: int = 30,
) -> None:
    q_log_dir = shlex.quote(remote_log_dir)
    q_stdout = shlex.quote(f"{remote_log_dir}/{label}.stdout.log")
    q_stderr = shlex.quote(f"{remote_log_dir}/{label}.stderr.log")
    q_exit = shlex.quote(f"{remote_log_dir}/{label}.exit")
    q_done = shlex.quote(f"{remote_log_dir}/{label}.done")
    wrapped = (
        f"mkdir -p {q_log_dir} && rm -f {q_stdout} {q_stderr} {q_exit} {q_done} && "
        f"nohup bash -lc {shlex.quote(cmd + f' > {q_stdout} 2> {q_stderr}; rc=$?; echo $rc > {q_exit}; touch {q_done}; exit $rc')} "
        f"</dev/null >/tmp/{shlex.quote(label)}.launch.log 2>&1 & echo $!"
    )
    exit_code, stdout, stderr = await pod.exec_ssh(wrapped, timeout=60)
    if exit_code != 0 or not stdout.strip():
        raise RuntimeError(f"{label} could not start: {stderr or stdout}")
    pid = stdout.strip().splitlines()[-1]

    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        check_cmd = (
            f"if test -f {q_done}; then cat {q_exit}; "
            f"elif kill -0 {shlex.quote(pid)} 2>/dev/null; then echo RUNNING; "
            f"else echo MISSING; fi"
        )
        _ec, status_out, _status_err = await pod.exec_ssh(check_cmd, timeout=60)
        status = status_out.strip().splitlines()[-1] if status_out.strip() else "MISSING"
        if status not in ("RUNNING", "MISSING"):
            out_text = (await pod.exec_ssh(f"test -f {q_stdout} && cat {q_stdout} || true", timeout=120))[1]
            err_text = (await pod.exec_ssh(f"test -f {q_stderr} && cat {q_stderr} || true", timeout=120))[1]
            (local_dir / f"{label}.stdout.log").write_text(out_text, encoding="utf-8")
            (local_dir / f"{label}.stderr.log").write_text(err_text, encoding="utf-8")
            if status != "0":
                raise RuntimeError(f"{label} failed with exit {status}: {err_text or out_text}")
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"{label} exceeded timeout after remote pid {pid}")
        await asyncio.sleep(poll_seconds)


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run exp024 style-triplet smoke on a fresh RunPod.")
    parser.add_argument("--config", default="configs/stage1_v3_9_style_triplet_022continue.yaml")
    parser.add_argument(
        "--checkpoint",
        default="../prompt-adapters/experiments/2026-05-text-022-broader-clean-split-restart/results/runpod_tmp/train_artifacts/tmp/stage1_v3_8_core2_cleanheldout_restart/final.pt",
    )
    parser.add_argument("--local-output-dir", default="../prompt-adapters/experiments/2026-05-text-024-v4-objective-data-repair/results/runpod_style_triplet")
    parser.add_argument("--remote-project-root", default="/tmp/text-ip-adapter-024")
    parser.add_argument("--workspace-root", default="/workspace/text-ip-adapter")
    parser.add_argument("--remote-checkpoint", default="/tmp/exp022_final.pt")
    parser.add_argument("--remote-train-dir", default="/tmp/stage1_v3_9_style_triplet_022continue")
    parser.add_argument("--remote-eval-dir", default="/tmp/exp024_eval")
    parser.add_argument("--remote-orch-log-dir", default="/tmp/exp024_orch")
    parser.add_argument("--experiment-id", default="2026-05-text-024-v4-objective-data-repair")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument(
        "--checkpoint-eval-steps",
        default="",
        help="Comma-separated checkpoint step numbers to eval in addition to final, e.g. 1000,2000,3000.",
    )
    parser.add_argument("--skip-checkpoint-upload", action="store_true")
    parser.add_argument("--prune-step-checkpoints-before-download", action="store_true")
    parser.add_argument("--skip-train-artifacts-download", action="store_true")
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = (repo_root / args.checkpoint).resolve()
    if not args.skip_checkpoint_upload and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    local_output_dir = (repo_root / args.local_output_dir).resolve()
    local_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_output_dir / "launch_manifest.json"

    experiment = load_experiment_config(repo_root / args.config)
    runpod_config = load_runpod_config(repo_root, storage_name=experiment.runpod.storage_name)
    storage_info = await verify_storage(runpod_config)

    pod = await launch(runpod_config, name="text-ip-adapter-exp024-triplet")
    manifest: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "status": "launched",
        "launched_at": _now(),
        "pod_id": pod.id,
        "config": args.config,
        "checkpoint": str(checkpoint),
        "remote_project_root": args.remote_project_root,
        "workspace_root": args.workspace_root,
        "remote_train_dir": args.remote_train_dir,
        "remote_eval_dir": args.remote_eval_dir,
        "remote_orch_log_dir": args.remote_orch_log_dir,
        "max_steps": args.max_steps,
        "n_probes": args.n_probes,
        "storage": storage_info,
    }
    _write_json(manifest_path, manifest)

    try:
        await pod.wait_ready(timeout=900)
        manifest["status"] = "ready"
        manifest["ready_at"] = _now()
        manifest["hf_token_synced"] = await _sync_hf_token(pod)
        _write_json(manifest_path, manifest)

        manifest["project_sync"] = await sync_project_root(
            pod,
            repo_root=str(repo_root),
            remote_root=args.remote_project_root,
        )
        dataset_syncs = []
        for local_dataset_dir in _configured_dataset_dirs(repo_root, experiment):
            rel_dir = local_dataset_dir.relative_to(repo_root)
            dataset_syncs.append(
                {
                    "local_dir": str(local_dataset_dir),
                    "remote_dir": f"{args.workspace_root}/{rel_dir.as_posix()}",
                    "result": await sync_dataset(
                        pod,
                        local_dir=str(local_dataset_dir),
                        remote_dir=f"{args.workspace_root}/{rel_dir.as_posix()}",
                    ),
                }
            )
        manifest["dataset_syncs"] = dataset_syncs
        _write_json(manifest_path, manifest)

        if args.skip_checkpoint_upload:
            manifest["checkpoint_upload"] = "skipped"
        else:
            await _upload_file(pod, checkpoint, args.remote_checkpoint)
            manifest["checkpoint_uploaded_at"] = _now()
        _write_json(manifest_path, manifest)

        q_project = shlex.quote(args.remote_project_root)
        q_workspace = shlex.quote(args.workspace_root)
        q_config = shlex.quote(args.config)
        q_train = shlex.quote(args.remote_train_dir)
        q_eval = shlex.quote(args.remote_eval_dir)
        q_probe = shlex.quote(experiment.training.probe_path)
        q_val = shlex.quote(experiment.data.val_path)
        q_ckpt = shlex.quote(f"{args.remote_train_dir.rstrip('/')}/final.pt")

        prep_cmd = (
            f"rm -rf {q_project}/data {q_project}/checkpoints && "
            f"ln -s {q_workspace}/data {q_project}/data && "
            f"ln -s {q_workspace}/checkpoints {q_project}/checkpoints && "
            f"mkdir -p {q_train} {q_eval}"
        )
        await _run_checked(pod, prep_cmd, timeout=120, label="prep", local_dir=local_output_dir)

        install_cmd = f"cd {q_project} && PYTHONPATH=src python -m pip install -e ."
        await _run_checked(pod, install_cmd, timeout=1800, label="install", local_dir=local_output_dir)

        preflight = f"""cd {q_project} && PYTHONPATH=src python - <<'PY'
from pathlib import Path
from text_ip_adapter import load_experiment_config
from text_ip_adapter.data.dataset import PairDataset
cfg = load_experiment_config('{args.config}')
paths = [
    'scripts/train.py',
    'src/text_ip_adapter/data/dataset.py',
    'src/text_ip_adapter/train/loop.py',
    '{args.config}',
    '{experiment.data.train_path}',
    '{experiment.training.probe_path}',
    '{experiment.training.init_from}',
]
for p in paths:
    path = Path(p)
    print(p, path.exists(), path.stat().st_size if path.exists() else -1)
print('max_steps', cfg.training.max_steps)
print('min_lr_ratio', cfg.training.min_lr_ratio)
print('style_triplet_weight', cfg.training.style_triplet_weight)
print('style_triplet_margin', cfg.training.style_triplet_margin)
print('contrastive_weight', cfg.training.contrastive_weight)
print('train_rows', len(PairDataset(cfg.data.train_path, tokenizer=None)))
PY"""
        stdout, _stderr = await _run_checked(pod, preflight, timeout=120, label="preflight", local_dir=local_output_dir)
        manifest["preflight_stdout"] = stdout
        manifest["status"] = "preflight_ok"
        _write_json(manifest_path, manifest)

        train_cmd = (
            f"cd {q_project} && PYTHONPATH=src python -u scripts/train.py "
            f"--config {q_config} --max-steps {args.max_steps} --output-dir {q_train}"
        )
        manifest["status"] = "training"
        manifest["training_started_at"] = _now()
        _write_json(manifest_path, manifest)
        await _run_remote_job(
            pod,
            train_cmd,
            timeout=10800,
            label="train",
            local_dir=local_output_dir,
            remote_log_dir=args.remote_orch_log_dir,
        )
        manifest["status"] = "training_completed"
        manifest["training_completed_at"] = _now()
        _write_json(manifest_path, manifest)

        eval_specs: list[tuple[str, str, int]] = []
        for raw_step in [part.strip() for part in args.checkpoint_eval_steps.split(",") if part.strip()]:
            step = int(raw_step)
            eval_specs.append((f"step_{step}", f"{args.remote_train_dir.rstrip('/')}/step_{step}.pt", step))
        eval_specs.append(("final", f"{args.remote_train_dir.rstrip('/')}/final.pt", args.max_steps))

        for tag, ckpt_path, step_tag in eval_specs:
            q_tag = shlex.quote(tag)
            q_step_ckpt = shlex.quote(ckpt_path)
            exists_code, _exists_out, _exists_err = await pod.exec_ssh(f"test -f {q_step_ckpt}", timeout=60)
            if exists_code != 0:
                manifest.setdefault("skipped_checkpoint_evals", []).append({"tag": tag, "checkpoint": ckpt_path})
                _write_json(manifest_path, manifest)
                continue

            pathway_cmd = (
                f"cd {q_project} && PYTHONPATH=src python -u scripts/probe_conditioning.py "
                f"--checkpoint {q_step_ckpt} --config {q_config} --probes {q_probe} "
                f"--output-dir {q_eval}/{q_tag}/pathway --n-probes {args.n_probes} --max-new-tokens 80"
            )
            await _run_remote_job(
                pod,
                pathway_cmd,
                timeout=3600,
                label=f"pathway_{tag}",
                local_dir=local_output_dir,
                remote_log_dir=args.remote_orch_log_dir,
            )

            sampled_cmd = (
                f"cd {q_project} && PYTHONPATH=src python -u scripts/eval_from_checkpoint.py "
                f"--checkpoint {q_step_ckpt} --config {q_config} --val-path {q_val} --probe-path {q_probe} "
                f"--output {q_eval}/{q_tag}/sampled_rep/samples.jsonl --max-new-tokens 120 --step-tag {step_tag} "
                "--do-sample --temperature 0.8 --top-p 0.9 --repetition-penalty 1.12 --no-repeat-ngram-size 3"
            )
            await _run_remote_job(
                pod,
                sampled_cmd,
                timeout=3600,
                label=f"sampled_eval_{tag}",
                local_dir=local_output_dir,
                remote_log_dir=args.remote_orch_log_dir,
            )

            pairwise_cmd = (
                f"cd {q_project} && PYTHONPATH=src python -u scripts/pairwise_style_eval.py "
                f"--samples {q_eval}/{q_tag}/sampled_rep/samples.jsonl --probes {q_probe} "
                f"--output {q_eval}/{q_tag}/pairwise_style_eval.json"
            )
            await _run_remote_job(
                pod,
                pairwise_cmd,
                timeout=600,
                label=f"pairwise_eval_{tag}",
                local_dir=local_output_dir,
                remote_log_dir=args.remote_orch_log_dir,
            )

        await download_path(pod, args.remote_eval_dir, str(local_output_dir / "eval_artifacts"))
        if args.skip_train_artifacts_download:
            manifest["train_artifacts_download"] = "skipped"
        else:
            if args.prune_step_checkpoints_before_download:
                await _run_checked(
                    pod,
                    f"rm -f {q_train}/step_*.pt",
                    timeout=120,
                    label="prune_step_checkpoints",
                    local_dir=local_output_dir,
                )
            await download_path(pod, args.remote_train_dir, str(local_output_dir / "train_artifacts"))
        manifest["status"] = "completed"
        manifest["completed_at"] = _now()
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "failed"
        manifest["failed_at"] = _now()
        manifest["error"] = repr(exc)
        if args.keep_pod_on_failure:
            manifest["terminated"] = False
            _write_json(manifest_path, manifest)
            raise
        raise
    finally:
        if manifest.get("status") == "failed" and args.keep_pod_on_failure:
            _write_json(manifest_path, manifest)
            return
        try:
            await pod.terminate()
            manifest["terminated"] = True
            manifest["terminated_at"] = _now()
        except Exception as exc:  # noqa: BLE001
            manifest["terminated"] = False
            manifest["terminate_error"] = repr(exc)
        _write_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
