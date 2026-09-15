# Table A — Coverage and cost on F/G/H

Per-cell `coverage_rate` from `operator_metrics_summary.json.coverage_monitor`,
and `overhead_replica_seconds_per_hour = overhead_replica_seconds / (duration_s/3600)`
from `metrics.json.resources`. Laddered methods pool n=5 (drift batch + replication extension);
raw methods n=3 (drift batch). Coverage < 0.85 flagged ⚠️ per agenda §3.2.

| method | pattern | coverage (mean ± std) | overhead_rs/h (mean ± std) | n (cov/cost) |
|---|---|---|---|---|
| `confscale-pid` | F | 0.9000 ± 0.0000 | 29500.0 ± 692.8 | 3/3 |
| `confscale-pid` | G | 0.7000 ± 0.0441 ⚠️ | 54910.2 ± 711.5 | 3/3 |
| `confscale-pid` | H | 0.8611 ± 0.0255 | 54960.3 ± 1559.4 | 3/3 |
| `confscale-pid-laddered` | F | 0.8833 ± 0.0118 | 31420.7 ± 1008.0 | 5/5 |
| `confscale-pid-laddered` | G | 0.8833 ± 0.0167 | 57955.0 ± 1992.5 | 5/5 |
| `confscale-pid-laddered` | H | 0.9134 ± 0.0075 | 59884.5 ± 778.0 | 5/5 |
| `confscale-aci` | F | 0.8667 ± 0.0167 | 27750.2 ± 585.1 | 3/3 |
| `confscale-aci` | G | 0.6889 ± 0.0096 ⚠️ | 54330.2 ± 832.3 | 3/3 |
| `confscale-aci` | H | 0.8056 ± 0.0096 ⚠️ | 53450.1 ± 1500.2 | 3/3 |
| `confscale-aci-laddered` | F | 0.8933 ± 0.0091 | 32010.8 ± 2429.4 | 5/5 |
| `confscale-aci-laddered` | G | 0.8767 ± 0.0190 | 56826.4 ± 1331.6 | 5/5 |
| `confscale-aci-laddered` | H | 0.8866 ± 0.0075 | 58692.4 ± 833.8 | 5/5 |
| `confscale-rolling-origin` | F | — | 26296.8 ± 147.1 | 0/3 |
| `confscale-rolling-origin` | G | — | 44656.7 ± 807.9 | 0/3 |
| `confscale-rolling-origin` | H | — | 39255.0 ± 2076.5 | 0/3 |
| `confscale-rolling-origin-laddered` | F | 0.8733 ± 0.0494 | 40870.6 ± 1291.0 | 5/5 |
| `confscale-rolling-origin-laddered` | G | 0.8433 ± 0.0190 ⚠️ | 59947.3 ± 256.1 | 5/5 |
| `confscale-rolling-origin-laddered` | H | 0.8100 ± 0.0149 ⚠️ | 59490.6 ± 150.0 | 5/5 |
| `hpa-error-monitored` | F | — | 26075.0 ± 1013.0 | 0/3 |
| `hpa-error-monitored` | G | — | 45176.7 ± 543.1 | 0/3 |
| `hpa-error-monitored` | H | — | 36086.7 ± 244.4 | 0/3 |
| `hpa-qr-monitored` | F | 0.2444 ± 0.0096 ⚠️ | 24323.5 ± 165.3 | 3/3 |
| `hpa-qr-monitored` | G | 0.0833 ± 0.0167 ⚠️ | 31460.0 ± 69.3 | 3/3 |
| `hpa-qr-monitored` | H | 0.0222 ± 0.0192 ⚠️ | 21800.0 ± 91.7 | 3/3 |
| `hpa-reactive` | F | — | 39630.0 ± 1614.4 | 0/3 |
| `hpa-reactive` | G | — | 61050.0 ± 158.7 | 0/3 |
| `hpa-reactive` | H | — | 67600.0 ± 277.1 | 0/3 |

## Notes
- `hpa-reactive` has no coverage monitor by design; cost-only.
- `hpa-error-monitored` coverage is NaN until Stream B (gauge-enabled rerun) lands. Cost rows populated from the drift batch.
- Stream A produced Pattern F only — `hpa-reactive` G and H rows are empty pending additional runs.
