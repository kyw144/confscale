# Provenance, adaptations and omissions

## Standalone runtime maintenance — 2026-09-14

The original extraction commit and source hashes below remain unchanged.
`provenance/runtime_adaptations.json` records subsequent fixes to runtime paths,
explicit workload seeds, failure propagation, replica observations, collection
timestamps and stale exported tests, including each previous shipped hash.
`confscale verify` checks their new destination hashes as well as unchanged evidence.
New cluster/configuration/assurance code, dependency locks, CI and run guides are
versioned normally in Git. These changes establish a supported execution route;
they do not adopt new scientific findings or alter the frozen tables.

The optional local restore can convert text line endings only when the converted
bytes hash-match the exported input manifest exactly. Models and trace inputs
remain excluded from Git. Runtime setup and outcomes are documented in
[CLUSTER_RUNS.md](CLUSTER_RUNS.md) and [VALIDATION.md](VALIDATION.md).

## Original extraction

`provenance/source_manifest.json` pins each included source-derived file to its relative source path and SHA-256 at source commit `18aa3ebba5d5dd0e5fac594c2a6d1aebb426036d`. It separately records the shipped hash and adaptation description. This is a fresh artifact tree, not a copy of the source repository's history. The private source workspace is not required for supported local commands.

The artifact's `.gitattributes` uses `* -text` so Git preserves the original bytes rather than converting line endings between Windows and Mac. This is required for the source/evidence hashes to remain meaningful after a checkout.

`provenance/paper_tables_expected.json` contains only table-cell text extracted from the nine-page reading copy, whose SHA-256 is `7b0555d8f842331e22022c452f5865fd60cec399ef5c04d171b8703f00948bf9`. The manuscript itself is not bundled. Numerical comparisons read these goldens independently of the aggregate-to-table code.

The source-derived code was copied directly except for these documented adaptations:

- Four small core components were relocated into the `confscale` package; coverage-monitor roadmap docstrings were clarified without behavioral changes.
- The planner and hysteresis were selected from the controller's AST. The separate, new stdlib upper-bound adapter is tested against the original NumPy path when NumPy is available.
- The copied controller and registry exclude experimental levers and external live telemetry outside this paper's scope. Published recalibration warm-start is retained. A source-side extraction script and full adaptation diff are retained by the author outside the visitor artifact.
- The copied registry/profiler use the active Python interpreter instead of an assumed source-workspace virtualenv.
- Direct network/cluster runtime entrypoints and selected study drivers have an early default stop. The load generator and three service apps require the same explicit opt-in on import as well, covering top-level load generation and the Dockerfile/Gunicorn route. The cluster create/delete functions also require explicit reference-runtime enablement. This does not make arbitrary reference imports side-effect free.
- Private home-directory strings in selected driver copies were replaced with `<SOURCE_WORKSPACE>`. They must become explicit paths during Mac preparation.
- The existing entity builder's root and output discovery were adapted to the bundled evidence directory. The new wrapper maps the five main paper tables to current numbering, clarifies the Table 3 supporting-table pointer, and uses the corrected descriptive RSS-SD label in Table 5. Reported numerical values are unchanged.

New code consists of the offline CLI, teaching harness, table wrapper/checks, preflight, documentation and local tests. This work is an AI-assisted preparation of the author's existing code and evidence; it does not establish additional scientific results. Original algorithm attributions remain in their source modules.

## Deliberate exclusions

`provenance/omitted_inputs.json` records available hashes and sizes for omitted model/trace/training files. These entries identify optional source inputs, not permission to redistribute them.

| Omitted surface | Reason / recovery route |
|---|---|
| Raw live episodes and TSDB snapshots | Large archival evidence; obtain from the author for raw-measurement audit. Tables 1/3 currently start at aggregate output. |
| GRU/UQ weight files and training CSVs | Keep the initial local package small and dependency-light; restore hash-pinned inputs or retrain with a new recorded lineage. |
| Alibaba raw and derived series | Redistribution terms have not been established for this artifact. Bring locally obtained inputs after checking upstream terms; ETL/protocol reference code is included. |
| Parallel bootstrap and historical shell sweep | Privileged setup/destructive orchestration and source-layout assumptions require a separate reviewed Mac adaptation. |
| Legacy operator/compute-service scaffolds and open-loop generator | Outside the paper's selected testbed/runtime surface. The four-service InfoSys testbed remains included. |
| Other-paper experiments, working records, reviewer material, manuscripts and Git history | Outside the visitor artifact's purpose. |

The public source repository is [kyw144/confscale](https://github.com/kyw144/confscale). No distribution license has been assigned. Omitted inputs remain excluded; any future inclusion requires a separate data-term decision. See [VALIDATION.md](VALIDATION.md) for tested environments and [MAC_VERIFICATION.md](MAC_VERIFICATION.md) for the pending validation on the original experiment Mac and kind cluster.
