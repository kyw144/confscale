# Experiment assurance

Three different outcomes are recorded:

| Check | What a pass establishes | What it does not establish |
|---|---|---|
| `confscale verify` / `reproduce` | Source integrity and regeneration of all five frozen tables | New raw experimental replication |
| `scripts/validate.py --runtime-tests` | Behavioral, failure-path, configuration and regression checks | A functioning cluster or performance result |
| `confscale.experiment run` | Executed cells satisfy their technical assurance checks and restore baseline | The paper's coverage, latency or cost claims |

## Experiment configuration

`configs/smoke.json` is schema version 1. Unknown keys, empty/duplicate methods or
workloads, duplicate seeds, invalid integer values and unknown methods fail during
offline planning. Paths resolve relative to the config file. The plan lists each
cell, seed, method, model pattern and explicit cluster context.

Methods run in listed order within each workload/seed. The same declared seed is
passed to each method's workload generator. A/B/C/D select diurnal/bursty/batch-ramp/
signaling models; E/F/G/H preserve the original runtime's diurnal-model fallback.
This mapping is printed before execution. The seed fixes the stochastic workload
law; wall-clock request timings, achieved demand and CPU scheduling remain variable.
The generator is a closed-loop dispatcher and can deliver less than target RPS.
Record achieved traffic and errors, and do not interpret target RPS as measured
offered load.

`duration_s`, `cooldown_s` and `complexity` are explicit. The supported wrapper
uses the named registry methods with their retained algorithm settings and enables
coverage measurement for interval methods. It preserves rolling recalibration's
warm-start variant. The controller recalibrates h0; per-horizon residual analyses
in `reference/studies/` are separate historical analysis paths.

The smoke criteria require finite request/replica/latency metrics, ordered finite
workload rows with successful traffic, and (for predictive cells) forecasts and
interval validations. HPA methods do not need prediction logs. Application errors
are reported and do not automatically fail an apparatus smoke test: the system
is deliberately driven through overload. A scientific card needs its own traffic
fidelity/error/SLO criteria, fixed before execution.

## Retained evidence

Every run directory has:

- `plan.json`, `environment.json`, `inputs.json`, `deployment.json`: exact plan,
  Git commit and dirty state, source/config hashes, installed package versions,
  model hashes, cluster UID and pinned image IDs.
- `before.json` / `after.json`: deployed resources before and after the batch.
- Per-cell run configuration, workload stdout/stderr and CSV, metrics JSON and
  timeseries, controller stdout and scale log, and controller summary when relevant.
- `execution.json`, `collection_summary.json`, `after_reset.json`, `assurance.json`:
  execution status, technical verdict and reset evidence.
- `receipt.json`: final status, cell verdicts, restoration status and output hashes.

Missing models never silently become HPA runs. Nonzero workload exit, premature
controller exit, absent trace, degraded collection, failed reset or failed
assurance stops the matrix and returns a nonzero process status. Failed evidence
is retained. A lock prevents simultaneous wrapper runs against one cluster.
The baseline is one ready worker and no benchmark HPA; it is explicitly verified
after each cell and at final cleanup. Forwarders terminate on the same path.

Controller logs distinguish `raw_target`, `target_replicas`,
`last_requested_replicas`, and `observed_replicas` (Deployment ready replicas).
Collection's instant Prometheus queries are anchored to the recorded workload
end time, so later query execution cannot shift the measured window.

HPA cells retain their pre-reset status and require `AbleToScale=True` and
`ScalingActive=True`. Interval counts are independently recomputed from the
ordered forecast/observation log. The workload trace must reach at least 80% of
its configured duration. Recheck a completed run without cluster access:

```sh
python -m confscale.experiment audit --output generated/runs/smoke-001
```

This verifies the receipt's output hashes, planned cell identities and current
technical checks. The receipt is a local integrity record, not a cryptographic
signature or an independent scientific certification.

The inherited `e2e.p95_ms` is the **95th percentile of per-tick p95 values**,
not a pooled per-request p95. `e2e.slo_violation_rate` is the fraction of recorded
ticks whose p95 exceeds 200 ms. These definitions are retained for compatibility
and must be distinguished from the frontend histogram estimate and from the
fraction of individual requests violating an SLO.

## Scientific replication remains a separate protocol

Before a claims-bearing run, freeze its independent unit, model/trace lineage,
workload selections, paired seeds and order/counterbalancing, warmup and drift
windows, exclusions, abort criteria, effect metric and tolerance. Retain negative
and failed runs. Do not select a passing seed or redefine a criterion afterward.
The original source settings are visible in `reference/stage3_scale/orchestrator/`
and the study drivers, with paper-to-evidence mapping in [PAPER_TO_CODE.md](PAPER_TO_CODE.md).

Tables 1 and 3 still begin at frozen aggregates. Original raw episodes, omitted
Alibaba data, full retraining and worker-binding scientific replication are not
covered by the short smoke. Host co-tenancy and container/runtime differences can
change a new measurement even with the same model and workload seed.
