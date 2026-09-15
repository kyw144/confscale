# ConfScale

Coverage monitoring and online recalibration for predictive autoscaling.
Code and evidence for *Coverage Monitoring and Online Recalibration for
Predictive Autoscaling under Deployment Drift* (CNSM 2026).

## Quick start

Use Python 3.10 or newer. These commands need no additional packages or cluster:

```sh
git clone https://github.com/kyw144/confscale.git
cd confscale
python -m confscale verify
python -m confscale reproduce
python -m confscale demo
```

| Command | Output |
|---|---|
| `verify` | Check source and evidence hashes |
| `reproduce` | Five paper tables as Markdown/CSV in `generated/paper/`, plus supporting figures |
| `demo` | Synthetic decision trace and summary in `generated/demo/` |

`reproduce` rebuilds tables from retained results; Tables 1 and 3 start at
aggregates. `demo` illustrates recalibration using synthetic forecasts and
requested replica targets, without a trained model or latency simulation.
Neither command reruns the paper experiments.

Run all offline checks with `python scripts/validate.py`. To use the `confscale`
command from another directory, install with `python -m pip install -e .`.
Evidence commands require this checkout; a wheel contains only the Python code.

## Cluster runs

Use a Linux Docker engine, kind 0.31.0, and kubectl compatible with Kubernetes
1.35. Allow at least 8 GiB of Docker memory. Install uv 0.11.6, then:

```sh
python3 scripts/setup_runtime.py
.venv/bin/python scripts/validate.py --runtime-tests
.venv/bin/python -m confscale.cluster render
.venv/bin/python -m confscale.cluster up
```

The setup script installs Python 3.12 dependencies from hashed locks. On Windows,
use `.venv/Scripts/python.exe` for Python commands; live cluster validation covers
Apple Silicon macOS. Linux and macOS have runtime CI; Windows has offline CI.

Before `up`, inspect the rendered manifests in `generated/clusters/confscale-repro/`.
The cluster uses pinned images and a private kubeconfig; it does not change your
current kubectl context. `up` refuses an existing cluster name. Use
`python -m confscale.cluster deploy` with the runtime interpreter to update a
cluster created by this checkout. Deployment and experiment runs share a lock.

Edit [configs/cluster.json](configs/cluster.json) before creating a cluster:

| Setting | Default |
|---|---|
| Cluster name | `confscale-repro` |
| Frontend port | `32080` |
| Prometheus port | `9290` |
| Controller metrics port | `9291` |

The namespace is fixed to `infosys-benchmark`. The testbed includes the frontend,
processor, compute worker, Redis, Prometheus, kube-state-metrics and metrics-server.
The kind-specific metrics-server configuration accepts the node's kubelet
certificate. Image digests are in [cluster/images.lock.json](cluster/images.lock.json).

### Supply models

For predictive runs, provide an input bundle with `data/p3_runs/models/` matching
[provenance/omitted_inputs.json](provenance/omitted_inputs.json):

```sh
.venv/bin/python -m confscale.inputs --source /path/to/input-bundle
.venv/bin/python -m confscale.experiment check-inputs
```

Models are copied to ignored `inputs/models/` only after hash verification.
LF/CRLF conversion is allowed for text files when the resulting hash matches
exactly. Existing, different files are not overwritten. Replacement models need
an explicit input-manifest update; missing models never silently fall back to HPA.

For a model-free smoke test, set `methods` to `["hpa-reactive"]` in
[configs/smoke.json](configs/smoke.json) and skip model restoration.

### Configure and run

```sh
python -m confscale.experiment plan --config configs/smoke.json
.venv/bin/python -m confscale.experiment run --config configs/smoke.json \
  --output generated/runs/my-run
python -m confscale.experiment audit --output generated/runs/my-run
```

Choose a new output directory for every attempt. Planning and auditing are offline;
`run` contacts the configured cluster. The default smoke runs HPA and PID for
180 seconds each and takes roughly 7–9 minutes with setup and cooldown.

The experiment config sets methods, workloads, paired seeds, duration, cooldown,
complexity, model location and minimum evidence counts. Paths resolve relative
to the config file. Unknown keys and empty or duplicate selections are rejected.
Methods run in listed order within each workload/seed.

The wrapper supports HPA, static replicas, SCP/BE/QR, ACI/PID, rolling-origin and
ladder variants. GRU-only baselines, KEDA and historical study drivers remain in
`reference/`; they are outside the supported smoke workflow. Direct reference
entrypoints require `CONFSCALE_ENABLE_REFERENCE_RUNTIME=1`. Read them before
opting in: they can generate traffic and modify cluster resources.

Workloads A/B/C/D select diurnal/bursty/batch-ramp/signaling models. E/F/G/H use
the diurnal model. The plan prints this mapping. A fixed seed reproduces the
stochastic workload law, while timing and achieved traffic can vary.

### Read the results

Each run retains its plan, Git commit, source/input hashes, installed packages,
image IDs, resource snapshots, workload/metrics files and controller logs.
`receipt.json` records cell verdicts, restoration status and output hashes;
`audit` verifies those hashes and independently checks the retained evidence.

A technical pass requires:

- Successful execution, finite metrics and a workload trace reaching at least
  80% of its configured duration.
- Healthy HPA conditions, or the required forecasts and later interval matches.
  Coverage counts are recomputed from the controller log.
- Restoration to one ready worker with no benchmark HPA, and stopped forwarders.

Failures stop the matrix, return a nonzero exit status and retain their outputs.
Controller logs separate `target_replicas` and `last_requested_replicas` from
`observed_replicas` (ready replicas).

Application errors remain visible but do not automatically fail an overload smoke.
The dispatcher can deliver less than target RPS. `e2e.p95_ms` is the 95th percentile
of per-tick p95 values, not a pooled request p95; `e2e.slo_violation_rate` counts
ticks whose p95 exceeds 200 ms. A technical pass does not establish a paper effect.
For scientific comparisons, fix the workload, model lineage, warmup, replicate
order, exclusions and acceptance criteria before running.

### Troubleshooting and cleanup

Start with `generated/clusters/<name>/provision.log` and the per-cell workload and
controller logs. Provisioning retains failed nodes for inspection. An abrupt kill
can leave `experiment.lock`: stop that run's processes and verify baseline
restoration before removing the lock.

If Colima reports `Too many open files` while starting a node, inspect its inotify
limits. Raising the instance limit to 1024 resolved this on the tested host:

```sh
colima ssh -- sysctl fs.inotify.max_user_instances fs.inotify.max_user_watches
colima ssh -- sudo sysctl -w fs.inotify.max_user_instances=1024
```

This is a temporary VM setting. See [kind troubleshooting](https://kind.sigs.k8s.io/docs/user/known-issues/#pod-errors-due-to-too-many-open-files).
For PyTorch hash errors, use `scripts/setup_runtime.py`; it selects the matching
wheel index for each platform without disabling hash checks.

To remove the dedicated cluster after saving results:

```sh
kind delete cluster --name confscale-repro \
  --kubeconfig generated/clusters/confscale-repro/kubeconfig
```

Cluster deletion removes its temporary Prometheus data; exported run files remain.

## Code and evidence

| Location | Contents |
|---|---|
| `confscale/` | Recalibrators, coverage monitor, replica planner and supported CLI |
| `reference/stage3_scale/` | Controllers, predictors, uncertainty models and testbed |
| `reference/studies/` | Study-specific analysis and drivers |
| `configs/`, `cluster/`, `requirements/` | Experiment settings, image pins and dependency locks |
| `evidence/` | Frozen paper results |
| `provenance/`, `validation/` | Source/input manifests, adaptations and validation receipts |
| `inputs/`, `generated/` | Local inputs and outputs; ignored by Git |

ACI/PID update interval widths from residuals; the coverage monitor scores prior
intervals; the ladder widens intervals after persistent undercoverage. The planner
selects maximum demand across horizons and applies hysteresis before scaling.
ACI/PID adjust h0; another horizon can still determine the replica target.

| Paper table | Analysis / evidence entry point |
|---|---|
| 1: calibration | `reference/stage3_scale/analysis/calibration.py` |
| 2: tuned HPA cost | `reference/studies/tuned_hpa/` |
| 3: drift and escalation | `reference/stage3_scale/analysis/post_reframe.py` |
| 4: per-service coverage | `reference/studies/per_service/` |
| 5: worker-binding latency | `reference/studies/binding/rc5_driver.py` |

[confscale/reproduce.py](confscale/reproduce.py) maps these tables to retained
inputs. [source_manifest.json](provenance/source_manifest.json) records original
source paths/hashes and shipped hashes; adaptation records preserve later changes.
Git preserves source/evidence line endings for hash verification. Model weights,
Alibaba traces, raw episodes and manuscript files are not bundled. No distribution
license has been assigned; third-party notices are retained.

## Validation and maintenance

| Check | Recorded result |
|---|---|
| Source integrity | 119 source-derived files verified |
| Paper tables | 132 data cells, 195 numeric values and 24 row labels matched |
| Software tests | 111 tests and 18 subtests passed |
| Live smoke | HPA and PID passed; PID issued 7 forecasts and validated 6 intervals |

The [live receipt](validation/mac-arm64-smoke.json) pins its tested commit, inputs,
checks and observed errors. Full paper-matrix replication, retraining and Alibaba
trace replay remain unverified. Historical test environments and CI results are
in the [previous validation record](https://github.com/kyw144/confscale/blob/d10776f040a55256d81a7793bd0825974cef2c4d/docs/VALIDATION.md).
The [documentation cleanup receipt](provenance/comment_cleanup.json) records
software checks and code/config equivalence; it includes no new live runs.

Run `python scripts/validate.py` for offline checks or
`.venv/bin/python scripts/validate.py --runtime-tests` for the full suite.
CI runs both where supported. Keep frozen evidence unchanged and save new runs
under `generated/`.

To update dependencies, edit the `.in` files, regenerate locks, review the diff
and rerun validation:

```sh
uv pip compile requirements/runtime.in --python-version 3.12 --universal \
  --generate-hashes --torch-backend cpu --emit-index-url -o requirements/runtime.lock
uv pip compile requirements/test.in --python-version 3.12 --universal \
  --generate-hashes -o requirements/test.lock
uv pip compile requirements/services.in --python-version 3.12 --universal \
  --generate-hashes -o requirements/services.lock
```
