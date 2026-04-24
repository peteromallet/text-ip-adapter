from __future__ import annotations

import asyncio
import json
from pathlib import Path

from text_ip_adapter.infra.runpod_runner import load_runpod_config, verify_storage


async def _main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config = load_runpod_config(repo_root)
    result = await verify_storage(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
