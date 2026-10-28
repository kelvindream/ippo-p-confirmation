#!/usr/bin/env bash
# DPG 低成本筛选：DPG-MAPPO / GC-MAPPO / IPPO × H1/H2 × seed 42/43/44
# 每任务 15 episodes + 固定评估 seed 1042/1043/1044；结果与旧 P1/P2 完全隔离。
set -u
cd /home/zrt/ai/GC-MAPPO

MAX_PARALLEL="${1:-3}"
OUT_ROOT="results/dpg_screen_v1"
LOG_DIR="${OUT_ROOT}/_logs"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR"

run_task() {
  local method="$1" scenario="$2" seed="$3" net="$4" route="$5"
  local safe_method="${method// /_}"
  local output="${OUT_ROOT}/${scenario}_${safe_method}_seed${seed}"
  local log="${LOG_DIR}/${scenario}_${safe_method}_seed${seed}.log"

  if [ -f "${output}/protocol_state.json" ]; then
    local status
    status=$($PY -c "import json;print(json.load(open('${output}/protocol_state.json')).get('status',''))" 2>/dev/null || true)
    if [ "$status" = "completed" ]; then
      echo "[SKIP] ${scenario} ${method} seed${seed} completed"
      return 0
    fi
    echo "[SKIP] ${scenario} ${method} seed${seed} existing status=${status:-unknown}"
    return 0
  fi

  echo "[START] ${scenario} ${method} seed${seed}"
  $PY paper_protocol.py \
    --method "$method" \
    --train-seed "$seed" \
    --train-episodes 15 \
    --eval-seeds 1042 1043 1044 \
    --net-file "$net" \
    --route-file "$route" \
    --output-dir "$output" \
    > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] ${scenario} ${method} seed${seed}"
  else
    echo "[FAIL] ${scenario} ${method} seed${seed} rc=${rc}; log=${log}"
  fi
}

TASK_FILE=$(mktemp)
cat > "$TASK_FILE" <<'EOF'
DPG-MAPPO|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
DPG-MAPPO|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
DPG-MAPPO|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
DPG-MAPPO|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
DPG-MAPPO|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
DPG-MAPPO|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GC-MAPPO|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GC-MAPPO|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GC-MAPPO|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GC-MAPPO|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GC-MAPPO|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GC-MAPPO|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
IPPO|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
IPPO|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
IPPO|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
IPPO|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
IPPO|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
IPPO|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
EOF

echo "=== DPG screening matrix: 18 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

running=0
while IFS='|' read -r method scenario seed net route; do
  [ -z "$method" ] && continue
  run_task "$method" "$scenario" "$seed" "$net" "$route" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done < "$TASK_FILE"
rm -f "$TASK_FILE"
wait

echo "=== All tasks returned: $(date --iso-8601=seconds) ==="
for method in DPG-MAPPO GC-MAPPO IPPO; do
  safe_method="${method// /_}"
  for scenario in h1 h2; do
    for seed in 42 43 44; do
      output="${OUT_ROOT}/${scenario}_${safe_method}_seed${seed}"
      if [ -f "${output}/protocol_state.json" ]; then
        status=$($PY -c "import json;print(json.load(open('${output}/protocol_state.json')).get('status','unknown'))" 2>/dev/null || echo error)
      else
        status="NOT_RUN"
      fi
      echo "${scenario} ${method} seed${seed}: ${status}"
    done
  done
done
