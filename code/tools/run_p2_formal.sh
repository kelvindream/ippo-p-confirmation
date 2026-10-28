#!/usr/bin/env bash
# P2 正式矩阵：BR-IPPO + GR-GMAPPO-L × S0/H1/H2 × seed 42/43/44
# 每任务 50 episodes 训练 + 3 评估 seed (1042/1043/1044)
# 用法: ./tools/run_p2_formal.sh [并发数，默认3]
set -u
cd /home/zrt/ai/GC-MAPPO

MAX_PARALLEL="${1:-3}"
OUT_ROOT="results/p2_formal"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "$LOG_DIR"

PY=".venv/bin/python"

run_task() {
  local method="$1" sc="$2" seed="$3" net="$4" route="$5"
  local out="${OUT_ROOT}/${sc}_${method}_seed${seed}"
  local ckpt="results/p1_dev_v1/${sc}_IPPO_seed${seed}/checkpoint.pt"
  local log="${LOG_DIR}/${sc}_${method}_seed${seed}.log"

  # 已完成则跳过
  if [ -f "${out}/protocol_state.json" ]; then
    local st
    st=$($PY -c "import json;print(json.load(open('${out}/protocol_state.json')).get('status',''))" 2>/dev/null || echo "")
    if [ "$st" = "completed" ]; then
      echo "[SKIP] ${sc} ${method} seed${seed} completed"
      return 0
    fi
  fi

  echo "[START] ${sc} ${method} seed${seed}"
  $PY paper_protocol.py \
    --method "$method" \
    --train-seed "$seed" \
    --train-episodes 50 \
    --eval-seeds 1042 1043 1044 \
    --ippo-init-checkpoint "$ckpt" \
    --net-file "$net" \
    --route-file "$route" \
    --output-dir "$out" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[DONE]  ${sc} ${method} seed${seed}"
  else
    echo "[FAIL]  ${sc} ${method} seed${seed} rc=$rc"
  fi
}

echo "=== P2 formal matrix: 18 tasks, parallel=${MAX_PARALLEL} ==="
echo "=== Started: $(date) ==="

# 任务列表文件
TASK_FILE=$(mktemp)
cat > "$TASK_FILE" << 'EOF'
BR-IPPO|s0|42|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
BR-IPPO|s0|43|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
BR-IPPO|s0|44|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
BR-IPPO|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
BR-IPPO|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
BR-IPPO|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
BR-IPPO|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
BR-IPPO|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
BR-IPPO|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GR-GMAPPO-L|s0|42|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
GR-GMAPPO-L|s0|43|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
GR-GMAPPO-L|s0|44|nets/2x2-grid/2x2.net.xml|nets/2x2-grid/2x2-medium.rou.xml
GR-GMAPPO-L|h1|42|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GR-GMAPPO-L|h1|43|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GR-GMAPPO-L|h1|44|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml
GR-GMAPPO-L|h2|42|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GR-GMAPPO-L|h2|43|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
GR-GMAPPO-L|h2|44|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml
EOF

# 并发控制：用后台进程+计数
running=0
while IFS='|' read -r method sc seed net route; do
  [ -z "$method" ] && continue
  run_task "$method" "$sc" "$seed" "$net" "$route" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done < "$TASK_FILE"
rm -f "$TASK_FILE"

wait
echo "=== All done: $(date) ==="
echo ""
echo "=== Summary ==="
for method in BR-IPPO GR-GMAPPO-L; do
  for sc in s0 h1 h2; do
    for seed in 42 43 44; do
      out="${OUT_ROOT}/${sc}_${method}_seed${seed}"
      if [ -f "${out}/protocol_state.json" ]; then
        st=$($PY -c "import json;print(json.load(open('${out}/protocol_state.json')).get('status','unknown'))" 2>/dev/null || echo "error")
        echo "${sc} ${method} seed${seed}: ${st}"
      else
        echo "${sc} ${method} seed${seed}: NOT_RUN"
      fi
    done
  done
done
