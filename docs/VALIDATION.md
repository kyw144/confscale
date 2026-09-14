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

## Pending validation on the experiment Mac

The original experiments ran on the author's Mac using kind. Validation of this exported repository in that environment is pending, including model loading, raw trace replay, standalone cluster execution and comparison with the original results. Follow [MAC_VERIFICATION.md](MAC_VERIFICATION.md) and record the checked repository commit, environment, input hashes, commands and results for each completed stage.

The Windows check establishes artifact integrity and offline regeneration. Tables 1 and 3 start from frozen aggregate evidence; it does not rerun the original experiments or establish their external validity.
