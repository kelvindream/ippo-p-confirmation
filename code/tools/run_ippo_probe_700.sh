#!/usr/bin/env bash
set -u
ROOT="/home/zrt/ai/GC-MAPPO"
cd "$ROOT"
OUT_ROOT="results/ippo_prior_probe_700"
LOG_DIR="$OUT_ROOT/_logs"
STALE_DIR="$OUT_ROOT/_stale"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR" "$STALE_DIR"
METHODS=("IPPO" "IPPO-P")
SEEDS=(72 73)
EVAL_SEEDS=(1042 1043 1044 1045 1046)
NET="nets/grid-3x3/3x3.net.xml"
ROUTE="nets/grid-3x3/3x3-probe-700.rou.xml"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    task="H1PROBE700_${method}_seed${seed}"
    out="$OUT_ROOT/$task"
    log="$LOG_DIR/$task.log"
    state="$out/protocol_state.json"
    resume_flag=""
    if [ -f "$state" ]; then
      status=$("$PY" -c "import json; print(json.load(open('$state')).get('status','unknown'))")
      [ "$status" = "completed" ] && { echo "[SKIP] $task"; continue; }
      [ "$status" = "running" ] && resume_flag="--resume"
    elif [ -d "$out" ]; then
      mv "$out" "$STALE_DIR/${task}_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$out"
    echo "[START] $task"
    "$PY" paper_protocol.py --method "$method" --train-seed "$seed" --sumo-seed "$seed" --train-episodes 50 --eval-seeds "${EVAL_SEEDS[@]}" --net-file "$NET" --route-file "$ROUTE" --output-dir "$out" $resume_flag > "$log" 2>&1 &
  done
done
wait
printf 'FINISHED %s\n' "$(date --iso-8601=seconds)"
