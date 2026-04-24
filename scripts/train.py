from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_ip_adapter import load_experiment_config
from text_ip_adapter.train.loop import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the text-ip-adapter on Gemma-3.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.output_dir:
        config.training.output_dir = args.output_dir

    summary = train(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
