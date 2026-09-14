# ConfScale: reproduce the tables and run the controller

Code and evidence artifact for **Coverage Monitoring and Online Recalibration for Predictive Autoscaling under Deployment Drift**, accepted to CNSM 2026. Repository: [kyw144/confscale](https://github.com/kyw144/confscale). No distribution license has been assigned yet.

The original experiments ran on the author's Mac using kind. This repository supports dependency-free paper-table regeneration and an explicit standalone cluster workflow with pinned dependencies, images, model hashes and experiment receipts. Start with the offline commands below, or follow [cluster setup and runs](docs/CLUSTER_RUNS.md). See the [validation record](docs/VALIDATION.md) for what has actually been checked.

Clone the complete artifact:

```console
git clone https://github.com/kyw144/confscale.git
cd confscale
```

From this directory, with Python 3.10 or newer:

```console
python -m confscale verify
python -m confscale reproduce
python -m confscale demo
python -m unittest discover -s tests -v
```

These commands need **no third-party packages, network, Docker, Kubernetes, trained models, or trace downloads**. They write only under `generated/` (or an explicit `--output` directory). `python scripts/validate.py` runs the offline checks together and saves a command receipt. GitHub Actions runs offline checks on Linux, macOS and Windows, plus the full runtime regression suite on Linux and macOS.

`reproduce` regenerates all **five main paper tables** as Markdown and CSV under `generated/paper/`, and supporting SVG figures under `generated/paper/supporting_entities/figures/`. It checks every table's numerical data and row identity against text extracted independently from the latest paper. Table 5 also recomputes means and sample deviations from the retained per-run values. Tables 1 and 3 start from frozen aggregate tables; this command does not rerun the original experiments or establish their external validity.

`demo` runs a new, deterministic teaching example through the original ACI, PID, coverage monitor and escalation ladder. It writes a per-decision `trace.csv` and a labeled `summary.json` under `generated/demo/`. These illustrative values are **not paper measurements**. Replica counts are requested targets before hysteresis, not observed replicas; there is no latency simulation. The demonstration supplies synthetic forecasts instead of using the paper's GRU.

`verify` checks source-derived files against the package's hash manifest. The tests check feedback direction, finite-sample rank behavior, FIFO measurement, persistence, causal ordering, repeatability, table reconstruction and disabled reference launch paths. If NumPy is installed, one additional test checks the stdlib upper-bound adapter against the extracted original planner.

For an editable install that exposes the `confscale` command from other directories:

```console
python -m pip install -e .
```

Use this repository layout or an editable install for evidence commands. The optional `.[planner]` extra adds NumPy for `confscale.planning_numpy`; the local demonstration and table reproduction do not need it. A code-only wheel is not the complete evidence artifact.

## Where to start

- [Tutorial](docs/TUTORIAL.md): trace one interval through observation, correction and a replica target.
- [Paper-to-code map](docs/PAPER_TO_CODE.md): find the code and evidence behind each main result.
- [Architecture](docs/ARCHITECTURE.md): identify what was copied, extracted or newly written.
- [Cluster setup and runs](docs/CLUSTER_RUNS.md): install the locked runtime, build an isolated testbed, restore inputs and run a smoke matrix.
- [Experiment assurance](docs/EXPERIMENTS.md): configuration, seeds, receipts, failure gates and scientific limits.
- [Requirements](requirements/README.md): Python 3.12 runtime, hashed dependency locks and update commands.
- [Provenance](docs/PROVENANCE.md): source commit, hashes, adaptations and omitted data.

The original runtime, GRU/UQ implementations, analysis, workload generators, InfoSys testbed and selected study drivers are in `reference/`. The supported live entrypoints are `python -m confscale.cluster` and `python -m confscale.experiment`; `experiment plan` is offline. Direct historical runtime entrypoints still require an explicit opt-in. The study drivers retain historical assumptions and are not covered by the standalone smoke workflow.

The artifact preserves warm-start behavior and the paper's negative boundaries: no general cost advantage over tuned HPA; horizon selection can mask a repaired interval; the latency benefit comes from a deliberately worker-binding configuration. Full trace replay, retraining and cluster reproduction remain unverified. Alibaba series, trained weights, raw episodes, manuscript files, private working records and original Git history are omitted.
