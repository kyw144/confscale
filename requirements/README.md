# Environments

The offline CLI requires Python >=3.10 and no dependencies. Live execution and
the complete regression suite use **Python 3.12**. Tested tooling: uv 0.11.6.

```sh
python3 scripts/setup_runtime.py
.venv/bin/python scripts/validate.py --runtime-tests
```

`runtime.in` retains the original direct runtime version pins. `runtime.lock`
adds all transitive dependencies and distribution hashes; on Linux/Windows
it selects CPU PyTorch from the PyTorch index, and on macOS the ordinary
PyTorch distribution from PyPI. CPU is the reference controller's inference
device. The setup script adds `--torch-backend cpu` only on Linux/Windows;
macOS must use the PyPI wheel whose hash is in the lock. The CPU index also
serves a same-version Mac wheel with different bytes, which correctly fails
hash verification. Do not disable hashes to work around this.

A pip alternative is `python -m pip install --require-hashes -r
requirements/runtime.lock` on macOS, adding `--extra-index-url
https://download.pytorch.org/whl/cpu` on Linux/Windows, inside Python 3.12.

For only the offline/configuration/failure-injection tests, sync `test.lock`.
Reference controller tests also need `runtime.lock`. The service images use
`services.lock` with Flask, its Prometheus exporter, Requests, Redis and
Gunicorn; the old unpinned reference Dockerfiles are retained for provenance.

To deliberately update the lock files:

```sh
uv pip compile requirements/runtime.in --python-version 3.12 --universal \
  --generate-hashes --torch-backend cpu --emit-index-url -o requirements/runtime.lock
uv pip compile requirements/test.in --python-version 3.12 --universal \
  --generate-hashes -o requirements/test.lock
uv pip compile requirements/services.in --python-version 3.12 --universal \
  --generate-hashes -o requirements/services.lock
```

Review the diff and repeat the checks after an update. Do not regenerate locks
as part of installation. Locking package versions does not freeze CPU scheduling,
container build timestamps or Kubernetes timing. The live receipt records the
installed packages, source hashes, platform, image IDs and input hashes.

Upstream documentation: [uv locking](https://docs.astral.sh/uv/pip/compile/),
[uv PyTorch support](https://docs.astral.sh/uv/guides/integration/pytorch/).
