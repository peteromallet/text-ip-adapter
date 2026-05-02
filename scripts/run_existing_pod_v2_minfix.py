from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from pathlib import Path

from runpod_lifecycle import get_pod

from text_ip_adapter.infra.runpod_runner import load_runpod_config
from text_ip_adapter.infra.ssh_sync import sync_project_root


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 curation and minfix training on an existing RunPod pod.")
    parser.add_argument("--pod-id-file", default="checkpoints/latest_pod_id.txt")
    parser.add_argument("--pod-id")
    parser.add_argument("--config", default="configs/stage1_v2_warmstart_minfix.yaml")
    parser.add_argument("--remote-root", default="/workspace/text-ip-adapter")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    pod_id = args.pod_id or (repo_root / args.pod_id_file).read_text(encoding="utf-8").strip()
    config = load_runpod_config(repo_root)
    pod = await get_pod(pod_id, config)
    await pod.wait_ready(timeout=300)

    sync_result = await sync_project_root(pod, repo_root=str(repo_root), remote_root=args.remote_root)
    commands: list[str] = []

    install_cmd = f"cd {shlex.quote(args.remote_root)} && PYTHONPATH=src python -m pip install -e ."
    exit_code, stdout, stderr = await pod.exec_ssh(install_cmd, timeout=1800)
    if exit_code != 0:
        raise RuntimeError(f"install failed: {stderr or stdout}")
    commands.append(install_cmd)

    if not args.skip_audit:
        # The text-ip-adapter project sync does not include prompt-adapters; copy
        # the script into the remote repo scripts directory before invoking it.
        local_script = repo_root.parent / "prompt-adapters" / "scripts_staging" / "curate_v2_pairs.py"
        remote_script = f"{args.remote_root}/scripts/curate_v2_pairs.py"
        ssh = await asyncio.to_thread(pod.open_ssh_client)
        try:
            sftp = ssh.open_sftp()
            try:
                sftp.put(str(local_script), remote_script)
            finally:
                sftp.close()
        finally:
            ssh.close()
        audit_cmd = (
            f"cd {shlex.quote(args.remote_root)} && "
            "python scripts/curate_v2_pairs.py --input-dir data/pairs_v2 --output-dir data/pairs_v2"
        )
        exit_code, stdout, stderr = await pod.exec_ssh(audit_cmd, timeout=600)
        commands.append(audit_cmd)
        if exit_code != 0:
            raise RuntimeError(f"curation failed: {stderr or stdout}")

    train_cmd = (
        f"cd {shlex.quote(args.remote_root)} && "
        f"PYTHONPATH=src python scripts/train.py --config {shlex.quote(args.config)}"
    )
    remote_log = f"{args.remote_root}/train_v2_minfix.log"
    start_cmd = (
        f"cd {shlex.quote(args.remote_root)} && "
        f"nohup bash -lc {shlex.quote(train_cmd)} > {shlex.quote(remote_log)} 2>&1 & echo $!"
    )
    exit_code, stdout, stderr = await pod.exec_ssh(start_cmd, timeout=60)
    commands.append(start_cmd)
    if exit_code != 0 or not stdout.strip():
        raise RuntimeError(f"could not start training: {stderr or stdout}")

    result = {
        "pod_id": pod.id,
        "sync": sync_result,
        "remote_root": args.remote_root,
        "remote_log": remote_log,
        "pid": int(stdout.strip().splitlines()[-1]),
        "commands": commands,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
