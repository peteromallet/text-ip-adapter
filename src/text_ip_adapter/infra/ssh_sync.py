from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path

from runpod_lifecycle import Pod

PROJECT_SYNC_INCLUDE = ("pyproject.toml", "README.md", "env.example", "src", "scripts", "configs")
PROJECT_SYNC_EXCLUDE = (".venv", "__pycache__", "wandb", "checkpoints", "data/raw", "data/processed", ".git", "eval_reports")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_archive(source_dir: Path) -> tuple[Path, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="text-ip-adapter-sync-"))
    archive_path = temp_dir / "payload.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
    return archive_path, _tree_hash(source_dir)


def _copy_project_subset(repo_root: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="text-ip-adapter-project-"))
    for rel_path in PROJECT_SYNC_INCLUDE:
        src = repo_root / rel_path
        dst = temp_root / rel_path
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*PROJECT_SYNC_EXCLUDE),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return temp_root


async def _read_remote_file(pod: Pod, remote_path: str) -> str:
    quoted = shlex.quote(remote_path)
    exit_code, stdout, _stderr = await pod.exec_ssh(f"test -f {quoted} && cat {quoted} || true")
    if exit_code not in (0, 1):
        return ""
    return stdout.strip()


async def _upload_archive(pod: Pod, archive_path: Path, remote_archive: str) -> None:
    ssh = await asyncio.to_thread(pod.open_ssh_client)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.put(str(archive_path), remote_archive)
        finally:
            sftp.close()
    finally:
        ssh.close()


async def _download_archive(pod: Pod, remote_archive: str, local_archive: Path) -> None:
    ssh = await asyncio.to_thread(pod.open_ssh_client)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.get(remote_archive, str(local_archive))
        finally:
            sftp.close()
    finally:
        ssh.close()


async def sync_project_root(pod: Pod, repo_root: str, remote_root: str = "/workspace/text-ip-adapter") -> dict[str, str | bool]:
    local_root = Path(repo_root).resolve()
    staged_root = _copy_project_subset(local_root)
    archive_path, payload_hash = _build_archive(staged_root)
    remote_hash_path = f"{remote_root}/.project_sync_hash"
    remote_archive = "/tmp/text-ip-adapter-project.tar.gz"

    if await _read_remote_file(pod, remote_hash_path) == payload_hash:
        return {"changed": False, "hash": payload_hash}

    await pod.exec_ssh(f"mkdir -p {shlex.quote(remote_root)}")
    await _upload_archive(pod, archive_path, remote_archive)
    await pod.exec_ssh(
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_root)} && "
        f"printf %s {shlex.quote(payload_hash)} > {shlex.quote(remote_hash_path)}"
    )
    return {"changed": True, "hash": payload_hash}


async def sync_dataset(pod: Pod, local_dir: str, remote_dir: str) -> dict[str, str | bool]:
    local_path = Path(local_dir).resolve()
    if not local_path.exists():
        return {"changed": False, "hash": ""}

    archive_path, payload_hash = _build_archive(local_path)
    remote_hash_path = f"{remote_dir}/.dataset_sync_hash"
    remote_archive = "/tmp/text-ip-adapter-dataset.tar.gz"
    if await _read_remote_file(pod, remote_hash_path) == payload_hash:
        return {"changed": False, "hash": payload_hash}

    await pod.exec_ssh(f"mkdir -p {shlex.quote(remote_dir)}")
    await _upload_archive(pod, archive_path, remote_archive)
    await pod.exec_ssh(
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_dir)} && "
        f"printf %s {shlex.quote(payload_hash)} > {shlex.quote(remote_hash_path)}"
    )
    return {"changed": True, "hash": payload_hash}


async def download_path(pod: Pod, remote_path: str, local_dir: str) -> None:
    local_root = Path(local_dir).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    local_archive = local_root / "download.tar.gz"
    remote_archive = "/tmp/text-ip-adapter-download.tar.gz"
    await pod.exec_ssh(
        f"tar -czf {shlex.quote(remote_archive)} -C / {shlex.quote(remote_path).lstrip('/')} || true"
    )
    await _download_archive(pod, remote_archive, local_archive)
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(local_root)


async def tail_remote_log(pod: Pod, remote_log_path: str, stop_event: asyncio.Event, interval_seconds: int = 5) -> None:
    last_seen = ""
    quoted = shlex.quote(remote_log_path)
    while not stop_event.is_set():
        _exit_code, stdout, _stderr = await pod.exec_ssh(f"test -f {quoted} && tail -n 40 {quoted} || true")
        if stdout and stdout != last_seen:
            print(stdout, flush=True)
            last_seen = stdout
        await asyncio.sleep(interval_seconds)


async def wait_for_remote_pid_exit(pod: Pod, pid: int, interval_seconds: int = 10) -> None:
    while True:
        exit_code, stdout, _stderr = await pod.exec_ssh(f"if kill -0 {pid} 2>/dev/null; then echo running; else echo exited; fi")
        if exit_code == 0 and stdout.strip() == "exited":
            return
        await asyncio.sleep(interval_seconds)
