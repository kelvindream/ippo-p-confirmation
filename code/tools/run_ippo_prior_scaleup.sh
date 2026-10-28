#!/usr/bin/env bash
# Scale-up extension (confirmed mainline A): 3x3-high (2700 veh/h).
# 2 methods x 5 fresh seeds x 1 scenario = 10 tasks.
# Same frozen protocol as ippo_prior_confirmation_v1; no re-registration.
# Usage: ./tools/run_ippo_prior_scaleup.sh [max_parallel]
set -u

ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"

MAX_PARALLEL="${1:-4}"
OUT_ROOT="results/ippo_prior_scaleup_v1"
LOG_DIR="${OUT_ROOT}/_logs"
STALE_DIR="${OUT_ROOT}/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"

METHODS=("IPPO" "IPPO-P")
SEEDS=(62 63 64 65 66)
EVAL_SEEDS=(1042 1043 1044 1045 1046)
NET="nets/grid-3x3/3x3.net.xml"
ROUTE="nets/grid-3x3/3x3-high.rou.xml"

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
  local method="$1" seed="$2"
  local task out log state status resume_flag=""
  task="H1HIGH_${method}_seed${seed}"
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
  echo "[START] ${task}"
  "$PY" paper_protocol.py \
    --method "$method" \
    --train-seed "$seed" \
    --sumo-seed "$seed" \
    --train-episodes 50 \
    --eval-seeds "${EVAL_SEEDS[@]}" \
    --net-file "$NET" \
    --route-file "$ROUTE" \
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

echo "=== Scale-up: 3x3-high 2700 veh/h, 10 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Seeds 62-66 (57-61 untouched, reserved by decision tree) ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n 2>/dev/null || true
    done
    run_task "$method" "$seed" &
  done
done
wait || true

echo "=== Finished: $(date --iso-8601=seconds) ==="
"$PY" - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
root = Path(sys.argv[1])
counts = Counter()
for task_dir in sorted(root.iterdir()):
    if not task_dir.is_dir() or task_dir.name.startswith('_'):
        continue
    state_path = task_dir / 'protocol_state.json'
    try:
        status = json.loads(state_path.read_text()).get('status', 'unknown')
    except Exception:
        status = 'invalid'
    counts[status] += 1
print('Scale-up status:', dict(counts))
PY
