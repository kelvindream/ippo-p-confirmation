#!/usr/bin/env bash
# Post-confirmation supplementary S3: IPPO-P prior scale probe (locked:
# results/supplementary_v1/s3_scale_probe_lock.json).
# H1 x scales {1.0, 1.5, 3.0} x seeds {67, 68} = 6 tasks.
# Usage: ./tools/run_s3_scale_probe.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

# Trae/IDE 注入的 LD_LIBRARY_PATH 会破坏 sumo 子进程的库加载（TraCI 连接失败）。
unset LD_LIBRARY_PATH

MAX_PARALLEL="${1:-6}"
OUT_ROOT="results/supplementary_v1/s3_scale_probe"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

SCENARIOS=("H1")
SCALES=(1.0 1.5 3.0)
SEEDS=(67 68)
EVAL_SEEDS=(1042 1043 1044 1045 1046)

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
  local scenario="$1" scale="$2" seed="$3"
  local net route task out log state status resume_flag=""
  net="nets/grid-3x3/3x3.net.xml"
  route="nets/grid-3x3/3x3-medium.rou.xml"
  task="${scenario}_IPPO-P_s${scale}_seed${seed}"
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
  echo "[START] ${task} scale=${scale}"
  GC_MAPPO_IPPO_PRIOR_SCALE="$scale" "$PY" paper_protocol.py \
    --method IPPO-P \
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

echo "=== S3 prior scale probe: 6 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Locked: results/supplementary_v1/s3_scale_probe_lock.json ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

for scenario in "${SCENARIOS[@]}"; do
  for scale in "${SCALES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || true
      done
      run_task "$scenario" "$scale" "$seed" &
    done
  done
done
wait || true

echo "=== Finished: $(date --iso-8601=seconds) ==="
completed=0
for scale in "${SCALES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    state="${OUT_ROOT}/H1_IPPO-P_s${scale}_seed${seed}/protocol_state.json"
    if [ "$(state_status "$state")" = "completed" ]; then
      completed=$((completed + 1))
    fi
  done
done
echo "=== Completed tasks: ${completed}/6 ==="
