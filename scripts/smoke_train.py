from __future__ import annotations

from pathlib import Path

from text_ip_adapter import load_experiment_config
from text_ip_adapter.train.loop import train


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config = load_experiment_config(repo_root / "configs" / "smoke.yaml")
    config.training.output_dir = str(repo_root / "checkpoints" / "smoke")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
