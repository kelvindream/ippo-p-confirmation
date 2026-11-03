# ippo-p-confirmation — Training-Consistent Pressure-Prior Integration for Independent Multi-Agent Traffic Signal Control

This folder is the submission/deposit package for the manuscript. It gathers, in
categorized form, every code artifact and experimental dataset that the paper's
Data Availability statement promises, plus the manuscript itself.

## Layout

```
ippo-p-confirmation/
├── README.md                     this file
├── REPRODUCE.md                  verified reproduction guide (layered, with observed results)
├── MANIFEST.md                   item-by-item map: Data Availability claim -> path
├── MANIFEST.sha256               SHA-256 of every file in this package (self-excluded)
├── checkpoint_hashes.sha256      SHA-256 of the 288 checkpoint binaries NOT copied
│                                 (144 tasks x {checkpoint.pt, resume_state.pt}), paths
│                                 relative to the source repository root
├── environment_snapshot/         Python 3.9.25, pip freeze of the training venv,
│                                 SUMO v1_25_0, hardware/OS/CUDA record
├── code/                         full pipeline code (see mapping below)
├── data/results/                 experimental data, one directory per study, original
│                                 names preserved (protocol file, manifests, per-episode
│                                 metrics, training logs, audit artifacts)
├── manuscript/                   draft.md, LaTeX source + built PDF, figures + scripts
└── docs/                         execution log (with erratum), experiment plan,
                                   evidence mapping, repo README, LICENSE
```

## Repository -> package mapping

| Repository path | Package path |
|---|---|
| `agents/`, `environment/`, `nets/`, `tools/` | `code/` |
| `paper_protocol.py`, `paper_statistics.py`, `main.py`, `networks.py`, `replay.py`, `requirements.txt` | `code/` |
| `run_phase3.sh`, `run_phase3_parallel.py`, `test_formal_four_grid.py`, `test_algorithms.py`, `test_dpg_mappo.py`, `test_gated_graph_residual_ippo.py` | `code/` |
| `results/ippo_prior_confirmation_v1/` | `data/results/ippo_prior_confirmation_v1/` |
| `results/formal_four_grid_v1/` (pilot) | `data/results/formal_four_grid_v1/` |
| `results/ippo_prior_scaleup_v1/`, `_v2/`, `results/ippo_prior_probe_700/` | `data/results/…` |
| `results/baselines_descriptive_v1/` (external anchors) | `data/results/baselines_descriptive_v1/` |
| `results/audit_recheck_v1/`, `results/eval_realization_sensitivity/`, `results/censoring_decomposition_v1/` | `data/results/…` |
| `paper/draft.md`, `paper/references_verified.md` | `manuscript/` |
| `paper/latex/{build.py,main.tex,main.pdf,_preproc.md}` | `manuscript/latex/` |
| `paper/figures/` (three figures + generating scripts + QA artifacts) | `manuscript/figures/` |
| `实验执行记录.md`, `论文骨架与证据映射.md`, `SCI2区论文实验方案.md`, `README.md`, `README.en.md`, `LICENSE` | `docs/` |

## What is intentionally not copied

- **Checkpoint binaries** (`checkpoint.pt`, `resume_state.pt`, 288 files): replaced by
  `checkpoint_hashes.sha256`, which fixes their content identity as disclosed in
  manuscript Sections 4.7 and 6.6. The source repository retains them.
- **Development-era result directories** not referenced by the paper (`p1_dev_v1`,
  `p2_formal`, `dpg_screen_v1`, `gc_opt_v1`, early smoke/quarantine outputs).
- **Internal review materials** (`paper/review_files/`, `paper/review_external_round1.md`).
- LaTeX build intermediates (`main.log`) and Python caches.

## Integrity verification

```bash
sha256sum -c MANIFEST.sha256          # from this folder; verify all listed files
sha256sum -c checkpoint_hashes.sha256 # from the source repository root
```

## Reproduction pointers

- Frozen protocol: `data/results/ippo_prior_confirmation_v1/protocol_frozen.json`
  (+ `protocol_frozen.sha256`), configuration id `pressure_zscore_hold`.
- Confirmatory aggregates: `data/results/ippo_prior_confirmation_v1/confirmation_summary.json`.
- Per-task manifests with source-file hashes: each task directory
  (`*_seed*/manifest.json`); note the disclosed post-freeze drift of
  `paper_protocol.py` (manuscript Sections 4.7, 6.6).
- Post-review analyses: `data/results/eval_realization_sensitivity/` (per-evaluation-
  episode reanalysis, manuscript Section 5.3) and
  `data/results/censoring_decomposition_v1/` (tripinfo-verified censoring
  decomposition, manuscript Section 6.6); forensic pilot recheck in
  `data/results/audit_recheck_v1/`.
- Manuscript rebuild: `python manuscript/latex/build.py` then
  `tectonic manuscript/latex/main.tex`; figures regenerate via
  `manuscript/figures/make_figure1.py` and `make_figures.py`.
- Environment: Python + PyTorch + SUMO (see `code/requirements.txt`).

## Placeholder

The archival DOI/repository identifier (`[FILL]` in the manuscript's Data
Availability statement) is to be inserted upon deposit; this package is the
deposit payload.

## Publication workflow (GitHub + Zenodo DOI)

1. Create a GitHub repository named `ippo-p-confirmation` (public; single-blind
   review confirmed), then push this folder as the repository root:
   `git init -b main && git add -A && git commit` (identity is configured
   locally) and add the remote/push.
2. On Zenodo, enable the GitHub integration (zenodo.org/account/settings/github),
   flip the repo on, then create a GitHub release (suggested tag: `v1.0.0`).
   Zenodo versions the release snapshot and mints the DOI.
3. Fill the Data Availability `[FILL]` in `manuscript/draft.md` line 298 with
   the Zenodo DOI, rebuild the PDF (`manuscript/latex/build.py` + tectonic),
   re-run the manifest refresh below, and push release `v1.0.1` so the archived
   snapshot matches the deposited statement.

Manifest refresh after any content change:

```bash
find . -type f ! -name 'MANIFEST.sha256' | sort | sed 's|^\./||' \
  | xargs sha256sum > MANIFEST.sha256 && sha256sum -c MANIFEST.sha256 --quiet
```

## License

Released under the Creative Commons Attribution 4.0 International license
(CC-BY-4.0); see `LICENSE`.
