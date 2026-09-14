# Standalone kind runs

The supported path builds the retained InfoSys applications and runs the original
controller through an explicit experiment wrapper. It uses a new single-node kind
cluster named `confscale-repro`; the default Kubernetes context is never selected
or modified. No parallel cluster bootstrap or historical study driver is required.

## Setup

Install Docker with a running Linux engine, kind and kubectl. The verified Mac
uses kind 0.31.0, Kubernetes 1.35.0 and Colima/Docker. Use a kubectl version within
one minor release of the API server. Allow at least 8 GiB of Docker memory for
this testbed; measure actual available CPU/memory before a scientific run.

```sh
python3 scripts/setup_runtime.py
.venv/bin/python scripts/validate.py --runtime-tests
.venv/bin/python -m confscale.cluster render
```

Inspect `generated/clusters/confscale-repro/`: kind topology, application and
monitoring manifests, and `plan.json` with selected images and manifest hashes.
`cluster/images.lock.json` pins upstream images by architecture and digest.
Application image tags derive from source, Dockerfile, base image and dependency
lock contents; deployment receipts also record actual image IDs. Python, Redis,
Prometheus, kube-state-metrics and metrics-server all have explicit versions.

Create the cluster and deploy:

```sh
.venv/bin/python -m confscale.cluster up
```

This command refuses an existing cluster name. The private kubeconfig and the
cluster's `kube-system` UID are saved below `generated/clusters/confscale-repro/`.
`deploy` rebuilds/reapplies a cluster already owned by this workspace and checks
that UID first. Failures retain their log and any created node for diagnosis;
provisioning never deletes an existing experiment cluster as a repair shortcut.

The benchmark namespace is intentionally fixed to `infosys-benchmark` because
the reference queries and service DNS assume it. Isolation is by cluster.
Prometheus scrapes application annotations plus kube-state-metrics and cAdvisor;
PodMonitor CRDs and a Prometheus operator are not needed. metrics-server supports
the CPU HPA. Its insecure kubelet TLS flag is confined to the local kind profile.
The initial deployment has one worker and no HPA; each cell selects its method.
KEDA and the GRU-only predictive baselines remain reference-only in this wrapper.

Host ports are explicit in `configs/cluster.json`: frontend 32080, Prometheus
9290, controller metrics 9291. The frontend mapping binds loopback. Each experiment
owns and checks its Prometheus forwarder, refuses an occupied metrics port, and
uses `--kubeconfig` and `--context` on every kubectl command. Change the profile
before creating a cluster to select different ports/name.

## Inputs and first run

For the author's existing dissertation checkout:

```sh
.venv/bin/python -m confscale.inputs --source "$HOME/claude-workspace/dissertation"
.venv/bin/python -m confscale.experiment check-inputs
.venv/bin/python -m confscale.experiment plan
.venv/bin/python -m confscale.experiment run --output generated/runs/smoke-001
```

The restore command checks all selected hashes before copying into ignored
`inputs/models/`. A text-only LF/CRLF conversion is accepted only if its full
output hash exactly matches the original export manifest, and is recorded in
`restore_receipt.json`. Changed weights fail. Repeated identical restoration is
allowed; different destination contents are never overwritten.

Other users need author-supplied inputs matching `provenance/omitted_inputs.json`.
Weights and Alibaba traces are not redistributed. A model-free baseline smoke
can use a copied config whose `methods` is `["hpa-reactive"]`; planning and this
baseline do not require model files. Training a replacement model creates a new
lineage: the current hash-pinned runner will reject it until an explicit input
manifest is reviewed and adopted.

The two-cell smoke takes roughly 7–9 minutes. It checks HPA execution, PID model
loading/prediction/coverage feedback, finite telemetry and cleanup. It does not
test a paper effect size. Use a new output directory for every attempt; completed
or failed runs are never resumed or overwritten. See [EXPERIMENTS.md](EXPERIMENTS.md).

## Troubleshooting and cleanup

Read `provision.log`, the experiment log, and each cell's workload/controller
logs first. On this Mac, creating an additional cluster initially exhausted the
Colima VM's inotify instances. The recorded temporary fix was:

```sh
colima ssh -- sysctl fs.inotify.max_user_instances fs.inotify.max_user_watches
colima ssh -- sudo sysctl -w fs.inotify.max_user_instances=1024
```

The prior instance limit was 512 and the watch limit was already 1048576. No
persistent sysctl file was changed. This is a host prerequisite, not a command
the bootstrap executes automatically. See [kind's documented issue](https://kind.sigs.k8s.io/docs/user/known-issues/#pod-errors-due-to-too-many-open-files).

An abrupt kill can leave `experiment.lock`. Inspect the referenced run, terminate
only its recorded controller/forwarder processes, and verify the one-worker/no-HPA
baseline before manually removing that lock. Ordinary exceptions and Ctrl-C run
cleanup and leave a failed receipt if restoration cannot be verified.

When finished with **this dedicated test cluster**, preserve receipts, then:

```sh
kind delete cluster --name confscale-repro \
  --kubeconfig generated/clusters/confscale-repro/kubeconfig
```

This removes live cluster state, including Prometheus' temporary TSDB; exported
run receipts remain. The default context and original dissertation clusters are
separate. kind usage reference: [quick start](https://kind.sigs.k8s.io/docs/user/quick-start/).
