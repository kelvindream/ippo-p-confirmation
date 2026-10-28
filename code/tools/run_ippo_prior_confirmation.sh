#!/usr/bin/env bash
# Frozen confirmatory experiment: IPPO-P vs IPPO only.
# 2 methods x 3 scenarios x 10 new training seeds = 60 tasks.
# Usage: ./tools/run_ippo_prior_confirmation.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

MAX_PARALLEL="${1:-4}"
OUT_ROOT="results/ippo_prior_confirmation_v1"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

METHODS=("IPPO" "IPPO-P")
SCENARIOS=("H1" "H2" "H3")
SEEDS=(47 48 49 50 51 52 53 54 55 56)
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

echo "=== Frozen IPPO prior confirmation: 60 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Primary contrast: IPPO-P vs IPPO; new seeds 47-56 ==="
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
"$PY" tools/check_ippo_prior_confirmation.py --root "$OUT_ROOT"
