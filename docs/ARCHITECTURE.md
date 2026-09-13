# What the artifact contains

```text
confscale/                         Small offline package
  aci.py / conformal_pid.py         Copied research recalibrators
  coverage_monitor.py              Copied measurement, docstrings clarified
  escalation_ladder.py             Copied persistence state machine
  planning_numpy.py                Extracted original planner and hysteresis
  demo.py                         New illustrative replay and stdlib adapter
  entities_legacy.py               Existing evidence builder, relocated paths
  reproduce.py                    New five-table wrapper and numeric checks
reference/stage3_scale/             Sanitized original P3 code and testbed
reference/studies/                  Frozen study procedures, launch-disabled
evidence/data/p3_runs/              Small frozen aggregate/result files
provenance/                        Hash pins and paper table-cell goldens
tests/                             Local behavioral and reproduction checks
generated/                        Disposable local outputs; ignored by Git
```

The research controller reads served-traffic measurements, obtains GRU forecasts and uncertainty bounds, validates the previous interval, updates recalibration state, selects a replica target and applies hysteresis before Kubernetes execution. The original runtime integrates these components with Prometheus and the InfoSys services. The small package exposes the computations without importing the network-facing runtime.

The ACI/PID classes and ladder preserve their source logic. The coverage monitor differs only in historical roadmap docstrings. The NumPy planner and hysteresis were selected from the source by Python AST location; their paper behavior is unchanged. Out-of-paper experimental levers and live external telemetry were removed from the copied controller and registry. The paper's `--recal-warmstart` behavior remains present.

`demo.upper_bound_target` is a stdlib adaptation of the original `ci-upper` branch. It validates inputs and returns the same ceiling-and-clipping result. It does not stand in for the complete tier planner or hysteresis. `planning_numpy.py` retains those original functions for detailed inspection and optional parity checks.

`reference/` is not part of the installed Python package. Direct execution guards provide a default stop, not a security sandbox. The standalone load generator and three service apps also require the opt-in on import, before dependencies or top-level work, so Dockerfile/Gunicorn service startup is covered. Importing other arbitrary historical study files can still have top-level side effects. The supported offline commands do not import those files. Live application, workload generation, snapshotting and cluster mutation belong to the explicitly separated Mac validation stage.
