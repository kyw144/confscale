PYTHON ?= python3

.PHONY: check runtime-env runtime-check cluster-plan cluster-up experiment-plan
check:
	$(PYTHON) scripts/validate.py

runtime-env:
	$(PYTHON) scripts/setup_runtime.py

runtime-check:
	.venv/bin/python scripts/validate.py --runtime-tests

cluster-plan:
	.venv/bin/python -m confscale.cluster render

cluster-up:
	.venv/bin/python -m confscale.cluster up

experiment-plan:
	$(PYTHON) -m confscale.experiment plan
