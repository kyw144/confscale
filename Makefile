PYTHON ?= python3

.PHONY: check runtime-env runtime-check cluster-plan cluster-up experiment-plan
check:
	$(PYTHON) scripts/validate.py

runtime-env:
	uv venv --python 3.12 .venv
	uv pip sync --python .venv/bin/python --require-hashes --torch-backend cpu requirements/runtime.lock requirements/test.lock

runtime-check:
	.venv/bin/python scripts/validate.py --runtime-tests

cluster-plan:
	.venv/bin/python -m confscale.cluster render

cluster-up:
	.venv/bin/python -m confscale.cluster up

experiment-plan:
	$(PYTHON) -m confscale.experiment plan
