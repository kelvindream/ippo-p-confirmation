#!/usr/bin/env bash
# Descriptive non-RL baselines for the frozen IPPO confirmation protocol.
# Controllers: Fixed-Time-Program (net program replay anchor) + Max-Pressure-Descriptive.
# 2 controllers x 3 scenarios (H1/H2/H3) x 5 evaluation seeds (1042-1046) = 30 episodes.
# Evaluation ONLY: paper_protocol.py skips training for these methods (train_metrics=None).
# Outputs: results/baselines_descriptive_v1/<SCENARIO>_<CONTROLLER>/evaluation_metrics.json
# Does NOT touch results/ippo_prior_confirmation_v1/ or any other existing results dir.
# Usage: ./tools/run_baselines_descriptive_v1.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

MAX_PARALLEL="${1:-3}"
OUT_ROOT="results/baselines_descriptive_v1"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

# Sanitize IDE-injected LD_LIBRARY_PATH entries: the bundled Trae libstdc++ breaks
# SUMO startup (GLIBCXX_3.4.30 not found). System paths suffice for SUMO + torch.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  CLEANED_LD="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/.trae-cn-server/' | paste -sd: -)"
  export LD_LIBRARY_PATH="$CLEANED_LD"
fi

EVAL_SEEDS=(1042 1043 1044 1045 1046)

echo "=== Harness binding check (agents/baselines_descriptive.py) ==="
"$PY" - <<'PY' || exit 1
import json
from agents.baselines_descriptive import summarize_net_programs, verify_harness_binding
print(json.dumps(verify_harness_binding(), indent=2))
for net in ('nets/grid-3x3/3x3.net.xml', 'nets/2x2-grid/2x2_oversaturated.net.xml'):
    programs = summarize_net_programs(net)
    cycles = {p['cycle_length_s'] for p in programs.values()}
    green_durations = {
        tuple(sorted({phase['duration_s'] for phase in p['phases'] if phase['is_green']}))
        for p in programs.values()
    }
    yellow_durations = {
        tuple(sorted({phase['duration_s'] for phase in p['phases'] if not phase['is_green']}))
        for p in programs.values()
    }
    print(f"[NET] {net}: tls={len(programs)} cycle_s={sorted(cycles)} "
          f"green_durations_s={sorted(green_durations)} yellow_durations_s={sorted(yellow_durations)}")
PY

scenario_files() {
  case "$1" in
    H1) printf '%s|%s\n' "nets/grid-3x3/3x3.net.xml" "nets/grid-3x3/3x3-medium.rou.xml" ;;
    H2) printf '%s|%s\n' "nets/2x2-grid/2x2_oversaturated.net.xml" "nets/2x2-grid/2x2-formal-h2.rou.xml" ;;
    H3) printf '%s|%s\n' "nets/2x2-grid/2x2_oversaturated.net.xml" "nets/2x2-grid/2x2-formal-h3.rou.xml" ;;
    *) return 2 ;;
  esac
}

state_status() {
  "$PY" - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding='utf-8') as handle:
        print(json.load(handle).get('status', 'unknown'))
except Exception:
    print('invalid')
PY
}

archive_stale() {
  local out="$1" task="$2" stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "$out" "${STALE_DIR}/${task}_${stamp}"
  echo "[ARCHIVE] ${task} -> ${STALE_DIR}/${task}_${stamp}"
}

run_task() {
  local controller="$1" method="$2" scenario="$3"
  local pair net route task out log state status
  pair=$(scenario_files "$scenario") || return 2
  IFS='|' read -r net route <<< "$pair"
  task="${scenario}_${controller}"
  out="${OUT_ROOT}/${task}"
  log="${LOG_DIR}/${task}.log"
  state="${out}/protocol_state.json"

  if [ -f "$state" ]; then
    status=$(state_status "$state")
    if [ "$status" = "completed" ]; then
      echo "[SKIP] ${task} completed"
      return 0
    fi
    archive_stale "$out" "$task"
  elif [ -d "$out" ]; then
    archive_stale "$out" "$task"
  fi

  mkdir -p "$out"
  echo "[START] ${task} method=${method} net=${net} route=${route}"
  # Evaluation-only task: --train-episodes 1 is a placeholder (no training loop is
  # executed for these methods); --train-seed/--sumo-seed mirror the confirmation
  # convention but every evaluation episode is re-seeded with its eval seed (1042-1046).
  "$PY" paper_protocol.py \
    --method "$method" \
    --train-seed 47 \
    --sumo-seed 47 \
    --train-episodes 1 \
    --eval-seeds "${EVAL_SEEDS[@]}" \
    --net-file "$net" \
    --route-file "$route" \
    --output-dir "$out" \
    > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] ${task}"
  else
    echo "[FAIL] ${task} rc=${rc} log=${log}"
  fi
  return "$rc"
}

echo "=== Descriptive baselines: 6 tasks = 30 evaluation episodes, parallel=${MAX_PARALLEL} ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

for scenario in H1 H2 H3; do
  for spec in "FixedTime:Fixed-Time-Program" "MaxPressure:Max-Pressure-Descriptive"; do
    controller="${spec%%:*}"
    method="${spec##*:}"
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n 2>/dev/null || true
    done
    run_task "$controller" "$method" "$scenario" &
  done
done
wait || true

echo "=== Finished: $(date --iso-8601=seconds) ==="
"$PY" - <<'PY'
import glob
import json
import os

root = 'results/baselines_descriptive_v1'
for path in sorted(glob.glob(os.path.join(root, '*_*', 'evaluation_metrics.json'))):
    task = os.path.basename(os.path.dirname(path))
    with open(path, encoding='utf-8') as handle:
        data = json.load(handle)
    u = data.get('demand_service_times') or []
    teleports = data.get('teleport_counts') or []
    conservation = data.get('demand_conservation_ok') or []
    completion = data.get('completion_rates') or []
    if u:
        print(
            f"[SUMMARY] {task}: U_mean={sum(u) / len(u):.2f}s "
            f"U_per_seed={[round(x, 1) for x in u]} teleports={teleports} "
            f"conservation_ok={all(conservation)} "
            f"completion_rates={[round(c, 4) for c in completion]}"
        )
    else:
        print(f"[SUMMARY] {task}: INCOMPLETE (no demand_service_times)")
PY
