# Descriptive Non-RL Baselines (baselines_descriptive_v1)

**Status: DESCRIPTIVE ONLY.** These results anchor the frozen IPPO confirmation
(`results/ippo_prior_confirmation_v1/`, see `protocol_frozen.json`); they are NOT
part of the pre-registered confirmatory hypothesis test (IPPO-P vs IPPO on
`demand_service_time`). Nothing under `results/ippo_prior_confirmation_v1/` or any
other existing results directory was read-modified by this run.

## What was run

Two non-learning, deterministic controllers evaluated through the **same**
environment / metric path as the confirmation tasks (`paper_protocol.py
_evaluate`), so `demand_service_time` (U), teleport counts, demand-conservation
checks, completion rates and all other fields are computed identically:

| Controller (dir suffix) | paper_protocol method | Rule (as coded, `agents/baselines_descriptive.py`) |
|---|---|---|
| `FixedTime` | `Fixed-Time-Program` | No-controller anchor: at every decision slot request the successor green phase in the net program's green-phase order (`(current+1) % n`). Zero traffic adaptivity. The env enforces each phase's net duration (33 s minimum green) and inserts 2 s yellow, so realized greens are 33–43 s, realized cycle ~156–172 s. |
| `MaxPressure` | `Max-Pressure-Descriptive` | Varaiya-style max-pressure: at each decision slot pick `argmax_p [sum over phase-p link pairs of halting(incoming) − halting(receiving lane)]`; ties → first phase in program order; current phase held while it remains argmax. Behaviorally identical to the harness method `Max-Pressure` (same per-signal call `TrafficSignal.max_pressure_action()`). |

A literal "leave SUMO's program alone" pass-through is not realizable in this
harness because `SumoEnv.reset()` overrides every intersection with a random
initial phase via `setRedYellowGreenState` (after which SUMO holds that state
until the next TraCI call); the order-replay above is the closest faithful
"program as-is" realizable through the shared env path.

## Net signal programs (verified from the net files)

Both nets embed one static program (`type="static"`, `programID="0"`,
`offset="0"`) identical at every intersection:

- `nets/grid-3x3/3x3.net.xml` (H1; 9 TLS A0–A2/B0–B2/C0–C2): 4 green phases × 33 s
  + 2 s yellow each → **140 s nominal cycle**.
- `nets/2x2-grid/2x2_oversaturated.net.xml` (H2/H3; 4 TLS A0/A1/B0/B1): same
  **140 s cycle** with the same 33/33/33/33 splits.

## Protocol

- Scenarios (identical bindings to the frozen confirmation): H1 =
  `3x3.net.xml` + `3x3-medium.rou.xml` (1800 veh/h); H2 = `2x2_oversaturated.net.xml`
  + `2x2-formal-h2.rou.xml` (2400 veh/h); H3 = `2x2_oversaturated.net.xml` +
  `2x2-formal-h3.rou.xml` (3000 veh/h).
- Evaluation-only: **no training** (`paper_protocol.py` skips the training loop for
  these methods; `train_episodes=1` in `protocol_state.json` is a placeholder, and
  `training_metrics` is null).
- Evaluation seeds **1042–1046** (same 5 seeds as the confirmation), one
  deterministic episode per seed, 10000 s simulation, `yellow_time=2`,
  `delta_rs_update_time=10` (10 s decision grid), `tls_min_links=4` — identical to
  the confirmation evaluation configuration (`_configure` defaults).
- 3 scenarios × 2 controllers × 5 seeds = **30 episodes**.
- `--train-seed 47 / --sumo-seed 47` are placeholders mirroring the confirmation
  convention; every evaluation episode is re-seeded with its own eval seed exactly
  as in `paper_protocol._evaluate`.

## Layout / reproduction

- `<SCENARIO>_<CONTROLLER>/evaluation_metrics.json` — per-task metrics (same schema
  as confirmation tasks). `<SCENARIO>_<CONTROLLER>/protocol_state.json` — full
  config, environment parameters, source hashes (includes
  `agents/baselines_descriptive.py`).
- `_logs/<task>.log` — per-task run logs; `run.log` — orchestrator log.
- Reproduce: `bash tools/run_baselines_descriptive_v1.sh [max_parallel]`
  (launched under tmux session `baselines_v1`).
- Binding guard: the runner first runs
  `agents.baselines_descriptive.verify_harness_binding()`, which fails fast if the
  harness stops routing these method labels through the documented controllers.

## Known environment note

SUMO fails to start when the IDE injects its bundled `libstdc++` via
`LD_LIBRARY_PATH` (`GLIBCXX_3.4.30 not found`); the runner strips
`/.trae-cn-server/` entries from `LD_LIBRARY_PATH` before launching tasks.
