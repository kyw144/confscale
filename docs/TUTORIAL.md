# Follow one decision

Run `python -m confscale demo` from the repository root. Open `generated/demo/trace.csv` in a spreadsheet or text editor. It contains five rows per tick: frozen, rolling-origin, ACI, PID and ACI with the escalation ladder.

The harness supplies a smooth point forecast and a deterministic residual stream. Residual amplitude increases at tick 80. All methods see the same observations. Before each observation is scored, the method issues an interval from residuals it has already seen. A miss then changes the state available to the *next* interval. `test_demo_future_does_not_change_past` checks that changing the later stream cannot change earlier rows.

Read these columns together:

| Column | Interpretation |
|---|---|
| `lower`, `upper` | Interval issued for the current observation |
| `covered` | Whether the observation fell inside that issued interval |
| `trailing_coverage` | Coverage after scoring, over up to 30 recent decisions |
| `alpha` | Recalibrator state after this observation; used for the next interval |
| `ladder_level` | Width multiplier selected from prior coverage: 0, 1 or 2 |
| `upper_h1` | A fixed-width second-horizon bound included to expose masking |
| `h0_sets_max` | Whether the repaired first horizon reaches the planner's maximum |
| `requested_replicas` | `ceil(max(upper, upper_h1) / (10 × 0.7))`, clipped to 1–20 |

ACI updates its working miscoverage level using the signed coverage error. PID adds the source implementation's integral and derivative terms. The quantile comes from a trailing residual buffer. By default, the ladder escalates after five consecutive cycles below 85% coverage and recovers one level after ten consecutive cycles at or above the 90% target. Coverage from 85% up to 90% resets both persistence counters. The monitor itself only measures; reading an alert does not execute a repair.

The planner takes the maximum across horizons. Widening the first horizon need not change that maximum, and a changed maximum need not cross a replica boundary. For example, an upper bound of 108 RPS with capacity 10 RPS per replica and utilization 0.7 requests 16 replicas; a bound of 100.565 requests 15. These are arithmetic examples, not a new deployment experiment.

This harness has no trained forecaster, request generator, actual service state, controller cooldown, feedback from replicas to observations, or SLO measurement. Its short synthetic stream is selected to make the mechanisms inspectable. Do not use its method ranking as scientific evidence. The source controller and hysteresis are available separately for reading in `reference/stage3_scale/orchestrator/controller.py` and `confscale/planning_numpy.py`.

Then run `python -m confscale reproduce`. The `generated/paper/` directory is deliberately separate: it contains outputs reconstructed from frozen evidence already used in the paper. Compare Table 2's strict SLO-matched HPA and cheapest gate-passing HPA as different comparators; compare Table 4's recovery windows separately from the gap-distribution analysis described in its note.
