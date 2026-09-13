# Paper-to-code and evidence map

Paper numbering follows the latest nine-page camera-ready reading copy. The retained older entity builder uses additional supporting-table numbers; those are confined to `supporting_entities/`.

| Paper question | Inspectable implementation | Frozen evidence / supported local reproduction |
|---|---|---|
| III: how does coverage become a control signal? | `confscale/coverage_monitor.py`, `aci.py`, `conformal_pid.py`, `escalation_ladder.py`; `reference/stage3_scale/orchestrator/controller.py` | New teaching replay: `python -m confscale demo` (illustrative only) |
| III: which forecast determines replicas? | `confscale/planning_numpy.py`; original controller's `compute_target_replicas` and `HysteresisManager` | Local arithmetic/parity tests; actual controller timing remains unverified |
| III: forecaster/UQ, synthetic workloads, four-service testbed | `reference/stage3_scale/predictor/`, `uq/`, `workload_gen.py`, `infosys-benchmark/` | Original source and manifests retained; weights, training CSVs and raw episodes omitted |
| IV-A, Table 1: do pretrained intervals retain nominal coverage? | `reference/stage3_scale/analysis/calibration.py` | `evidence/data/p3_runs/results/real/tables/table_3_calibration.md` → `generated/paper/table_1.{md,csv}` |
| IV-B, Table 2: does the cost advantage survive a tuned HPA? | `reference/studies/tuned_hpa/`; original HPA registry in `reference/stage3_scale/orchestrator/methods.py` | `results/ev7_tuned_baseline_20260601_203020/ev7_cost_analysis.json` → Table 2; A/B/C cost JSONs also retained under `evidence/data/p3_runs/outputs/` |
| IV-C, Table 3 and Figure 2: how do correction and escalation behave under drift? | Copied recalibrators/ladder; original controller and method registry; original F/G/H workload functions | `results/post_reframe/tables/table_a_coverage_cost_drift.md` → Table 3 and `supporting_entities/figures/figure_1_cost_vs_coverage.svg`; retained rescue JSONs support additional comparisons |
| IV-C: raw rolling-origin and warm-start distinction | `reference/studies/raw_rolling/`, `warmstart/`; source `--recal-warmstart*` flags | Procedure visible; their raw reruns are not reconstructed by this package |
| IV-D, Table 4: does aggregate coverage hide service-local errors? | `reference/studies/per_service/`, `etl/` | `results/ev8b_pass2_full_20260601_210028/{pass2_full_summary,pass2_full_recal}.json` plus `results/ev8b_perservice_20260601_024537/recal_volatility.json` → Table 4 |
| IV-D: expanded level/volatility pool | `reference/studies/expanded_pool/` | Frozen `reopen_2026-06/T2/t2_coverage_sweep_r6.json` retained for inspection; trace replay not executed |
| IV-E: can another horizon mask the corrected bound? | `reference/studies/per_horizon/`; `planning_numpy.py` | `reopen_2026-06/T1/recal_perservice_h1.json`, `T1b/recal_pool_h1.json`, and `results/ev8b_perservice_20260601_024537/h1_binds.json` retained; residual-stream results do not establish a live modified-controller deployment |
| IV-F, Table 5: can repaired coverage affect latency when the worker binds? | `reference/studies/binding/rc5_driver.py` | `outputs/slo_binding_existence_proof_20260621_134302/rc5_results.json` → Table 5; per-run p95 values recomputed locally |

Paths beginning `results/`, `outputs/` or `reopen_2026-06/` in the evidence column are under `evidence/data/p3_runs/`.

The source manifest records an exact source path, commit and SHA-256 for every included source-derived file. No paper number is recomputed from the teaching demo. Table 1 and Table 3 reconstruction begins at archived aggregates, so it confirms presentation consistency rather than validating the original measurement pipeline. The remaining main tables use retained analysis JSON; Table 5 additionally retains the three p95 observations per cell.

The strict SLO-matched and cheapest gate-passing HPA rows are separate comparison questions. The real-data gap-distribution and recovery windows are separate cuts. Actuator reach means the corrected horizon supplies the selected upper bound; it is not automatically a replica change. The engineered worker-binding latency result does not transfer to the unmodified testbed, and no throughput claim is made.
