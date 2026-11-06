#!/usr/bin/env bash
# Post-confirmation supplementary S1: PressLight matched arm (locked:
# results/supplementary_v1/s2_presslight_lock.json).
# 1 method x 3 scenarios x 10 seeds (47-56) = 30 tasks.
# Paired against archived IPPO results from ippo_prior_confirmation_v1.
# Usage: ./tools/run_s2_presslight.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

# Trae/IDE 注入的 LD_LIBRARY_PATH 会破坏 sumo 子进程的库加载（TraCI 连接失败）。
unset LD_LIBRARY_PATH

MAX_PARALLEL="${1:-8}"
OUT_ROOT="results/supplementary_v1/s2_presslight"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

METHODS=("PressLight")
SCENARIOS=("H1" "H2" "H3")
SEEDS=(57 58 59 60 61 62 63 64 65 66)
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

echo "=== S2 PressLight baseline: 30 tasks (seeds 57-66), parallel=${MAX_PARALLEL} ==="
echo "=== Locked: results/supplementary_v1/s2_presslight_lock.json ==="
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
completed=0
for scenario in "${SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    state="${OUT_ROOT}/${scenario}_PressLight_seed${seed}/protocol_state.json"
    if [ "$(state_status "$state")" = "completed" ]; then
      completed=$((completed + 1))
    fi
  done
done
echo "=== Completed tasks: ${completed}/30 ==="
