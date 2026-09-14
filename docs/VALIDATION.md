# Validation record

## Windows artifact check — 2026-09-14

Environment: Windows, AMD64, Python 3.11.9. The checked code, tests and frozen evidence come from artifact commit `ad12dec50040abad0db5d70cafb76a2311bb8d35`; publication changes only update documentation and citation metadata.

| Command | Result |
|---|---|
| `python -m confscale verify` | Passed: all 119 source-derived files match their recorded SHA-256 hashes. |
| `python -m confscale reproduce --output generated/windows-paper` | Passed: all five main paper tables regenerate; 132 data cells, 195 numeric values and 24 row labels match the retained paper-table expectations. |
| `python -m confscale demo --output generated/windows-demo` | Passed: generated the labeled synthetic trace and summary. These are illustrative outputs, not paper measurements. |
| `python -m unittest discover -s tests -v` | 16 tests discovered: 15 passed, 1 skipped. The skipped test compares the original NumPy planner with the stdlib adapter; NumPy was not installed. |
| `python scripts/mac_preflight.py` | Passed as an offline inventory on Windows. It contacted no cluster and reported no runtime validation. |

The test suite covers feedback behavior, causal ordering, repeatability, paper-table reconstruction, source integrity and the default stops on reference runtime entrypoints. Its paper-regeneration check also runs without site packages. Docker, kind, kubectl, trained models and training inputs were unavailable in this check environment.

Local command receipts and stdout/stderr were retained under `generated/windows-verification/`, which is intentionally ignored by Git. Table CSVs and supporting figures are under `generated/windows-paper/`; the demo outputs are under `generated/windows-demo/`.

## Original Mac handoff boundary

The original experiments ran on the author's Mac using kind. Validation of this exported repository in that environment is pending, including model loading, raw trace replay, standalone cluster execution and comparison with the original results. Follow [MAC_VERIFICATION.md](MAC_VERIFICATION.md) and record the checked repository commit, environment, input hashes, commands and results for each completed stage.

The Windows check establishes artifact integrity and offline regeneration. Tables 1 and 3 start from frozen aggregate evidence; it does not rerun the original experiments or establish their external validity.

## Mac standalone implementation — 2026-09-14/15

Environment: Apple Silicon arm64, macOS 26.2 (Darwin 25.2), Python 3.12.13,
uv 0.11.6. A fresh environment was installed from hashed runtime/test locks.

- Source verification: all 119 retained source-derived files match the updated
  manifest; original source hashes remain recorded and runtime adaptations have
  their own history. Frozen evidence was not changed.
- All five paper tables regenerate with the same 132 data cells, 195 numerical
  values and 24 row labels. Offline commands also work with Python's site packages
  disabled. Initial system-Python check used 3.14.4.
- `scripts/validate.py --runtime-tests`: 26 unittest tests pass, including the
  optional NumPy parity check; pytest runs 109 tests and 18 subtests successfully.
- 179 local model inputs hash-match the export. 107 were already byte-identical;
  72 text files needed a recorded LF-to-CRLF conversion to match the exact export
  hashes. No weights or traces were added to Git.
- A fresh `confscale-repro` cluster was built with Kubernetes 1.35.0, kind 0.31.0,
  pinned upstream image digests and locked application dependencies. All seven
  deployments became ready. The original five clusters and default `kind-p4-hotel`
  context were preserved.
- Initial two-cell technical smoke (`generated/runs/smoke-001`) completed:
  HPA had 59 workload ticks; PID had 54 ticks, seven predictions and six validated
  intervals. Both cells restored baseline; the owned forwarder exited. This first
  diagnostic used the initial assurance checks. The final checks additionally
  retain HPA health before reset and independently recompute interval counts;
  a fresh run is required to validate that stronger contract.

The diagnostic recorded 330 failed requests for HPA and 292 for PID under overload.
These are retained anomalies, not evidence of a paper effect. Its public software
validation status does not certify traffic fidelity or latency/coverage/cost claims.

The first cluster create failed because Colima exhausted its 512 inotify instances.
The failed node log was retained, only the newly created failed node was removed,
and the VM's temporary instance limit was raised to 1024. The watch limit stayed
1048576 and no persistent host configuration was changed. The subsequent create
succeeded. See [CLUSTER_RUNS.md](CLUSTER_RUNS.md) for the documented prerequisite.

Local command logs and full receipts remain under ignored `generated/validation/`,
`generated/clusters/confscale-repro/`, and `generated/runs/`. Final commit-pinned
live/CI results are appended below after verification.
