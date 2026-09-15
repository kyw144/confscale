# Working on ConfScale

This repository is the public artifact and supported local runtime. Treat external
source checkouts as read-only references; do not edit their documents, evidence
or cluster resources as part of maintenance here.

- Keep `evidence/` byte-identical. New runs belong under ignored `generated/`.
- Keep models/traces/credentials private under ignored `inputs/` or explicit
  external input paths. Never commit a kubeconfig or copy a whole source workspace.
- Preserve the source commit and source hashes in `provenance/source_manifest.json`.
  For a changed source-derived file, record the adaptation and previous shipped
  hash, then update only its destination hash. Never change a hash to hide an
  unexplained difference.
- Use `python scripts/validate.py` for dependency-free changes; run
  `.venv/bin/python scripts/validate.py --runtime-tests` for runtime changes.
- Live tests use the dedicated configured cluster and a new output directory.
  Do not use the current kubectl context. Keep the original runtime's opt-in guard.
- Software verification, a technical smoke, and scientific replication are
  different outcomes. Preserve failed runs and report limitations honestly.
- Update the run guide/validation record with actual verification. Push only
  scoped changes; never rewrite pushed history.

Use short docstrings and comments for non-obvious constraints. Keep user-facing
documentation in README.md; store validation and provenance details in receipts.
