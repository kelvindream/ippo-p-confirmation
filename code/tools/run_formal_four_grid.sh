#!/usr/bin/env bash
# Formal four-grid confirmation matrix:
# 4 methods x 3 scenarios x 5 training seeds = 60 tasks.
# Each task trains 50 episodes and evaluates seeds 1042-1046.
# Usage: ./tools/run_formal_four_grid.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

MAX_PARALLEL="${1:-4}"
OUT_ROOT="results/formal_four_grid_v1"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

METHODS=("IPPO" "IPPO-P" "GC-MAPPO-ind-noP" "GC-MAPPO-ind-P")
SCENARIOS=("H1" "H2" "H3")
SEEDS=(42 43 44 45 46)
EVAL_SEEDS=(1042 1043 1044 1045 1046)

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
  local method="$1" scenario="$2" seed="$3"
  local pair net route task out log state status resume_flag=""
  pair=$(scenario_files "$scenario") || return 2
  IFS='|' read -r net route <<< "$pair"
  task="${scenario}_${method}_seed${seed}"
  out="${OUT_ROOT}/${task}"
  log="${LOG_DIR}/${task}.log"
  state="${out}/protocol_state.json"

  if [ -f "$state" ]; then
    status=$(state_status "$state")
    if [ "$status" = "completed" ]; then
      echo "[SKIP] ${task} completed"
      return 0
    fi
    if [ "$status" = "running" ] && [ -f "${out}/resume_state.pt" ]; then
      resume_flag="--resume"
      echo "[RESUME] ${task}"
    else
      archive_stale "$out" "$task"
    fi
  elif [ -d "$out" ]; then
    archive_stale "$out" "$task"
  fi

  mkdir -p "$out"
  echo "[START] ${task} net=${net} route=${route}"
  "$PY" paper_protocol.py \
    --method "$method" \
    --train-seed "$seed" \
    --sumo-seed "$seed" \
    --train-episodes 50 \
    --eval-seeds "${EVAL_SEEDS[@]}" \
    --net-file "$net" \
    --route-file "$route" \
    --output-dir "$out" \
    ${resume_flag} \
    > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] ${task}"
  else
    echo "[FAIL] ${task} rc=${rc} log=${log}"
  fi
  return "$rc"
}

echo "=== Formal four-grid: 60 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

for scenario in "${SCENARIOS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || true
      done
      run_task "$method" "$scenario" "$seed" &
    done
  done
done
wait || true

echo "=== Finished: $(date --iso-8601=seconds) ==="
"$PY" tools/check_formal_four_grid.py --root "$OUT_ROOT"
