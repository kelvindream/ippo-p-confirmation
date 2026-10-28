#!/usr/bin/env bash
# GC 优化定位矩阵（P0+O1）：
#   gc_base: GC-MAPPO @50ep 共享 actor（P0 基线）
#   gc_ind : GC-MAPPO @50ep 独立图 actor（O1，GC_MAPPO_INDEPENDENT_ACTOR=1）
# 场景 H1(3x3 medium)/H2(2x2 oversaturated) × seed 42/43/44，评估 seed 1042/1043/1044。
# 与 P1 IPPO（H1 135.11 / H2 112.05）及 results/gc_opt_v1 内部对照比较。
set -u
cd /home/zrt/ai/GC-MAPPO

MAX_PARALLEL="${1:-3}"
OUT_ROOT="results/gc_opt_v1"
LOG_DIR="${OUT_ROOT}/_logs"
PY=".venv/bin/python"
mkdir -p "$LOG_DIR"

run_task() {
  local variant="$1" scenario="$2" seed="$3" net="$4" route="$5"
  local output="${OUT_ROOT}/${scenario}_${variant}_seed${seed}"
  local log="${LOG_DIR}/${scenario}_${variant}_seed${seed}.log"

  if [ -f "${output}/protocol_state.json" ]; then
    echo "[SKIP] ${scenario} ${variant} seed${seed} exists"
    return 0
  fi

  local env_prefix=""
  if [ "$variant" = "gc_ind" ]; then
    env_prefix="GC_MAPPO_INDEPENDENT_ACTOR=1"
  fi

  echo "[START] ${scenario} ${variant} seed${seed}"
  env $env_prefix $PY paper_protocol.py \
    --method "GC-MAPPO" \
    --train-seed "$seed" \
    --train-episodes 50 \
    --eval-seeds 1042 1043 1044 \
    --net-file "$net" \
    --route-file "$route" \
    --output-dir "$output" \
    > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] ${scenario} ${variant} seed${seed}"
  else
    echo "[FAIL] ${scenario} ${variant} seed${seed} rc=${rc}; log=${log}"
  fi
}

TASK_FILE=$(mktemp)
cat > "$TASK_FILE" <<'EOF'
gc_base|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_base|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_base|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_base|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
gc_base|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
gc_base|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
gc_ind|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_ind|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_ind|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
gc_ind|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
gc_ind|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
gc_ind|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
EOF

echo "=== GC optimization matrix: 12 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Started: $(date --iso-8601=seconds) ==="

running=0
while IFS='|' read -r variant scenario seed net route; do
  [ -z "$variant" ] && continue
  run_task "$variant" "$scenario" "$seed" "$net" "$route" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done < "$TASK_FILE"
rm -f "$TASK_FILE"
wait

echo "=== All tasks returned: $(date --iso-8601=seconds) ==="
for variant in gc_base gc_ind; do
  for scenario in h1 h2; do
    for seed in 42 43 44; do
      output="${OUT_ROOT}/${scenario}_${variant}_seed${seed}"
      if [ -f "${output}/protocol_state.json" ]; then
        status=$($PY -c "import json;print(json.load(open('${output}/protocol_state.json')).get('status','unknown'))" 2>/dev/null || echo error)
      else
        status="NOT_RUN"
      fi
      echo "${scenario} ${variant} seed${seed}: ${status}"
    done
  done
done
