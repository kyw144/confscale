# Reproducibility work record

The owner requested on 2026-09-14 that this exported project be updated with
reproducible cluster configurations, requirements and experiment assurance,
then pushed back to GitHub. Work starts at artifact commit
`8adef807d1c95696491d95eb800a74136a1bcad4`.

The dissertation checkout is a read-only source of historical configuration
and hash-pinned local inputs. Its current HEAD at inspection was
`adfa5afc4718078dde6e66b72eff8b7d2aed9541`; the artifact extraction remains pinned
to `18aa3ebba5d5dd0e5fac594c2a6d1aebb426036d`. No paper result or scientific
acceptance criterion is changed by this maintenance work.

## Implementation decisions

- Keep offline reproduction dependency-free; pin Python 3.12 runtime and test
  dependencies separately, with transitive package hashes. Service images use
  their own small shared lock.
- Use a dedicated kind cluster and a private, ignored kubeconfig. Keep the
  original single-node topology and InfoSys resource settings. Derive runnable
  manifests from the retained sources; remove PodMonitors when using the
  existing annotation-scraping Prometheus deployment.
- Restore optional inputs only into ignored local storage after hash checking.
  No trace, model or credential is added to the public repository.
- Make workload seeds explicit, retain failed runs, and fail when required
  inputs, telemetry, controller behavior or cleanup cannot be verified.
- Frozen table regeneration, live technical smoke tests and scientific
  replication have separate verdicts. A short smoke test is never evidence
  for paper coverage, cost or latency claims.

Changes are reversible using Git; original frozen evidence remains byte-identical.
The retained reference manifest records each runtime adaptation separately.
