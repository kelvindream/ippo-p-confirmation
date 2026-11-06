# MANIFEST.md — Data Availability cross-reference

Each item promised in the manuscript's Data Availability statement, mapped to its
location in this package.

| Data Availability item | Package path | Status |
|---|---|---|
| Frozen protocol file (`protocol_frozen.json`) + SHA-256 hash file | `data/results/ippo_prior_confirmation_v1/protocol_frozen.json`, `.sha256` | present |
| Confirmation summary with all seed-level effects (`confirmation_summary.json`) | `data/results/ippo_prior_confirmation_v1/confirmation_summary.json` | present |
| Per-task manifests (source-file SHA-256, route bindings, variant configs, checkpoint paths) | `data/results/ippo_prior_confirmation_v1/*_seed*/` (also pilot/scaleup/probe task dirs) | present |
| Per-task training logs | `data/results/*/**/training_log.jsonl` | present |
| Per-evaluation-seed metrics | `data/results/*/**/evaluation_metrics.json` (+ `protocol_state.json`) | present |
| Checkpoints | **not copied**; binaries hashed in `checkpoint_hashes.sha256` (420 files) | hashes present |
| Scenario network and route files | `code/nets/grid-3x3/`, `code/nets/2x2-grid/` | present |
| Exploratory outputs (`ippo_prior_scaleup_v1`, `_v2`, `ippo_prior_probe_700`) | `data/results/ippo_prior_scaleup_v1/`, `_v2/`, `ippo_prior_probe_700/` | present |
| Pilot four-grid outputs (`formal_four_grid_v1`) | `data/results/formal_four_grid_v1/` | present |
| External anchors (`baselines_descriptive_v1` + generator script + shell entry) | `data/results/baselines_descriptive_v1/`; generators in `code/agents/baselines_descriptive.py`, `code/tools/run_baselines_descriptive_v1.sh` | present |
| Post-review deposits: `audit_recheck_v1`, `eval_realization_sensitivity`, `censoring_decomposition_v1` with generating scripts | `data/results/…`; generators in `code/tools/audit_recheck_recompute.py`, `code/tools/audit_recheck_rerun_eval.py`, `code/tools/audit_recheck_trace_actions.py`, `code/tools/analyze_eval_realization_sensitivity.py`, `code/tools/reeval_tripinfo_h1_seed47.py` | present |
| Post-confirmation supplementary set (Section 4.7): lock files with `locked_before_results = true` + `locks.sha256`, per-task results (`s1_matched_arm/`, `s2_presslight/`, `s3_scale_probe/`), analysis summaries | `data/results/supplementary_v1/`; generators in `code/tools/run_s1_matched_arm.sh`, `run_s2_presslight.sh`, `run_s3_scale_probe.sh`, `analyze_s1_matched_arm.py`, `analyze_s2_s3.py` | present |
| Method-lock configuration id `pressure_zscore_hold` | recorded inside `protocol_frozen.json` | present |
| Git commit / runtime-library versions / hardware / checkpoint-file hashes | checkpoint-file hashes: `checkpoint_hashes.sha256` (420 files); runtime/library versions and hardware: `environment_snapshot/` (pip freeze, Python 3.9.25, SUMO v1_25_0, hardware record); git commit identifiers remain disclosed-unavailable (manuscript Sections 4.8, 6.6) | largely resolved, as disclosed |
| Public archival DOI/repository | GitHub `kelvindream/ippo-p-confirmation`; Zenodo concept DOI 10.5281/zenodo.23920343 (version-specific DOI for snapshot v1.0.1: 10.5281/zenodo.23920344) | present |

Non-promise supporting materials included for provenance: manuscript (`manuscript/`),
execution log with post-review erratum (`docs/实验执行记录.md`), experiment plan
(`docs/SCI2区论文实验方案.md`), evidence mapping (`docs/论文骨架与证据映射.md`), license
(`docs/LICENSE`).
