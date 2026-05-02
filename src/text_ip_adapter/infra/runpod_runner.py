from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from runpod_lifecycle import RunPodConfig, get_network_volumes, launch

from text_ip_adapter.config import load_experiment_config
from text_ip_adapter.infra.ssh_sync import sync_dataset, sync_project_root


def _configured_dataset_dirs(repo_path: Path, experiment) -> list[Path]:
    dirs: list[Path] = []
    for raw_path in (
        experiment.data.train_path,
        experiment.data.val_path,
        experiment.data.test_path,
        experiment.training.probe_path,
    ):
        if not raw_path:
            continue
        path = (repo_path / raw_path).parent.resolve()
        if path not in dirs:
            dirs.append(path)
    return dirs


def _default_env_file(repo_root: Path) -> Path:
    return repo_root.parent / "runpod-lifecycle" / ".env"


async def _sync_hf_token(pod) -> bool:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token_text = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token_path.exists() and not token_text:
        return False
    await pod.exec_ssh("mkdir -p ~/.cache/huggingface", timeout=30)
    ssh = await asyncio.to_thread(pod.open_ssh_client)
    try:
        sftp = ssh.open_sftp()
        try:
            if token_path.exists():
                sftp.put(str(token_path), "/root/.cache/huggingface/token")
            else:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
                    handle.write(token_text.strip())
                    handle.flush()
                    sftp.put(handle.name, "/root/.cache/huggingface/token")
        finally:
            sftp.close()
    finally:
        ssh.close()
    await pod.exec_ssh("chmod 600 ~/.cache/huggingface/token", timeout=30)
    return True


def load_runpod_config(repo_root: Path, storage_name: str | None = None) -> RunPodConfig:
    env_file = os.getenv("RUNPOD_LIFECYCLE_ENV")
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv(_default_env_file(repo_root))
    return RunPodConfig.from_env(storage_name=storage_name or os.getenv("RUNPOD_STORAGE_NAME"))


async def verify_storage(config: RunPodConfig) -> dict[str, object]:
    volumes = await asyncio.to_thread(get_network_volumes, config.api_key)
    if not volumes:
        raise RuntimeError(
            "Could not fetch any RunPod network volumes. The local environment likely cannot reach the RunPod API."
        )
    matches = [volume for volume in volumes if volume.get("name") == config.storage_name]
    if not matches:
        raise RuntimeError(f"Configured storage volume '{config.storage_name}' was not found")
    return {"storage_name": config.storage_name, "matches": matches}


async def launch_training_run(
    repo_root: str,
    config_path: str,
    *,
    detach: bool = True,
    max_steps: int | None = None,
) -> dict[str, object]:
    repo_path = Path(repo_root).resolve()
    experiment = load_experiment_config(repo_path / config_path)
    runpod_config = load_runpod_config(repo_path, storage_name=experiment.runpod.storage_name)
    storage_info = await verify_storage(runpod_config)

    pod = await launch(runpod_config, name="text-ip-adapter-train")
    await pod.wait_ready(timeout=900)
    hf_token_synced = await _sync_hf_token(pod)

    await sync_project_root(pod, repo_root=str(repo_path), remote_root=experiment.runpod.remote_root)
    for local_dataset_dir in _configured_dataset_dirs(repo_path, experiment):
        rel_dir = local_dataset_dir.relative_to(repo_path)
        await sync_dataset(
            pod,
            local_dir=str(local_dataset_dir),
            remote_dir=f"{experiment.runpod.remote_root}/{rel_dir.as_posix()}",
        )

    remote_root = experiment.runpod.remote_root
    install_cmd = f"cd {shlex.quote(remote_root)} && PYTHONPATH=src python -m pip install -e ."
    await pod.exec_ssh(install_cmd, timeout=1800)

    effective_steps = max_steps or experiment.training.max_steps
    train_cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f"PYTHONPATH=src python scripts/train.py --config {shlex.quote(config_path)} --max-steps {effective_steps}"
    )
    remote_log = f"{remote_root}/train.log"
    if detach:
        start_cmd = (
            f"cd {shlex.quote(remote_root)} && "
            f"nohup bash -lc {shlex.quote(train_cmd)} > {shlex.quote(remote_log)} 2>&1 & echo $!"
        )
        exit_code, stdout, stderr = await pod.exec_ssh(start_cmd, timeout=60)
        if exit_code != 0 or not stdout.strip():
            raise RuntimeError(f"Could not start remote training: {stderr or stdout}")
        pod_id_path = repo_path / "checkpoints" / "latest_pod_id.txt"
        pod_id_path.parent.mkdir(parents=True, exist_ok=True)
        pod_id_path.write_text(pod.id, encoding="utf-8")
        return {
            "pod_id": pod.id,
            "remote_log": remote_log,
            "pid": int(stdout.strip().splitlines()[-1]),
            "storage": storage_info,
            "remote_root": remote_root,
            "hf_token_synced": hf_token_synced,
        }

    exit_code, stdout, stderr = await pod.exec_ssh(train_cmd, timeout=7200)
    return {
        "pod_id": pod.id,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "storage": storage_info,
        "remote_root": remote_root,
        "hf_token_synced": hf_token_synced,
    }
