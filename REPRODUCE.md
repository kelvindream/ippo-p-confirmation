# REPRODUCE.md — what can be reproduced from this package, and how

Verified 2026-11-02 by executing each layer in a throwaway copy of this package
(`/tmp`), comparing every regenerated artifact against the archived originals.
Nothing in the package was modified by the verification; results below are the
observed outcomes, including the negative ones.

## Layer 1 — statistical re-analysis and audits (VERIFIED, package-only)

Needs: Python 3.9+ with numpy. No SUMO, no GPU, no checkpoints.

```bash
python code/tools/analyze_eval_realization_sensitivity.py --root .
```

The script auto-detects either the source layout (`results/`) or this deposit
layout (`data/results/`) and writes only inside the selected root unless
`--output-dir` is supplied.

Observed: exits 0; console table identical to the archived run (per-evaluation-
seed sign counts 10/10, 10/10, 10/10, 10/10, 9/10; Jensen gap 3.361e-05);
regenerated `summary.json` equals the archived file after removing its single
`generated_at` field; `per_eval_seed.md` differs only in the timestamp line.

## Layer 2 — figures (content verified; byte identity environment-dependent)

```bash
python manuscript/figures/make_figures.py     # figures 2 and 3
python manuscript/figures/make_figure1.py     # figure 1 (writes to CWD)
```

Observed across three independent renders: the **text layer of every figure is
identical** to the archived PDFs (all labels, numbers, legends). The PDF
content stream is deterministic across consecutive renders inside one
environment, but differs when the rendering environment differs (matplotlib /
interpreter version), and PDF bytes additionally embed creation timestamps.
Conclusions: figures reproduce in content; byte-identical copies require the
original rendering environment, which the archived PNG/SVG/PDF files in
`manuscript/figures/` already fix.

## Layer 3 — manuscript PDF (VERIFIED, package-only)

Needs: pandoc (`~/.local/bin/pandoc`) and tectonic.

```bash
python manuscript/latex/build.py    # regenerates main.tex from manuscript/draft.md
tectonic manuscript/latex/main.tex  # run from the package root; main.tex references
                                    # figures under manuscript/figures/
```

Observed (re-verified 2026-11-02 after the final rigor-audit revisions):
build.py emits main.tex (95,959 bytes; 21 references; figures 1-3 wired);
tectonic produces a 16-page PDF whose extracted text is **identical to the
archived `manuscript/latex/main.pdf`**.

## Layer 4 — bit-level evaluation re-runs (needs source repository)

Requires the 288 checkpoint binaries, which this package ships as hashes only
(`checkpoint_hashes.sha256`). With the source repository (checkpoints retained)
the evaluation path was previously verified bit-exact twice (see
`data/results/audit_recheck_v1/`); any copy whose checkpoints match the hash
file behaves identically under the recorded evaluation configuration. Evaluation
uses deterministic argmax control, explicit Python/NumPy/PyTorch reseeding, and
the recorded SUMO `--seed`; no stronger claim about a sole stochastic source is
needed for the bit-exact result.

## Layer 5 — retraining from scratch (attemptable; bit-exactness not claimed)

All 60 confirmatory tasks are fully specified: seeds, task matrix
(`code/tools/run_ippo_prior_confirmation.sh`), networks and routes
(`code/nets/`), and the frozen protocol
(`data/results/ippo_prior_confirmation_v1/protocol_frozen.json`). The training
environment is recorded in `environment_snapshot/` (Python 3.9.25, pip freeze
of the training venv, SUMO v1_25_0, hardware, CUDA-enabled torch 2.8.0+cu126).
Bit-exact retraining is **not claimed**: the manuscript discloses (Sections
4.7, 6.6) that git provenance was not captured and the frozen-hash copy of
`paper_protocol.py` drifted post-freeze in a functionally verified way
(bit-exact re-evaluation reproduced the stored evaluation vectors). Two
practical notes for a fresh run: the runner hard-codes
`ROOT=/home/zrt/ai/GC-MAPPO` and a `.venv` interpreter — adjust both to your
paths; the archived per-episode `training_log.jsonl` files are the reference
for learning-curve comparison.

## Not included by design

Checkpoint binaries (hashed instead), development-era result directories not
referenced by the paper, and internal review materials (see README.md).
