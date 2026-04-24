#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNPOD_SRC="$REPO_ROOT/../runpod-lifecycle/src"
export PYENV_VERSION="${PYENV_VERSION:-3.11.11}"
export PYTHONPATH="$REPO_ROOT/src:$RUNPOD_SRC${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

python scripts/doctor_runpod.py
python scripts/runpod_bootstrap.py
python scripts/train_runpod.py --detach "$@"
