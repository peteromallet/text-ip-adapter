from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from text_ip_adapter.infra.runpod_runner import launch_training_run


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Launch a RunPod training run.")
    parser.add_argument("--config", default="configs/stage1_gemma.yaml")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    result = await launch_training_run(
        repo_root=str(repo_root),
        config_path=args.config,
        detach=args.detach,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
