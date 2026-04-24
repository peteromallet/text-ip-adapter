PYTHON ?= PYENV_VERSION=3.11.11 python

.PHONY: smoke-train train-local train-runpod bootstrap test

smoke-train:
	$(PYTHON) scripts/smoke_train.py

train-local:
	$(PYTHON) scripts/train.py --config configs/stage1_frozen.yaml

train-runpod:
	$(PYTHON) scripts/train_runpod.py --detach

bootstrap:
	$(PYTHON) scripts/runpod_bootstrap.py

test:
	PYENV_VERSION=3.11.11 python -m pytest tests
