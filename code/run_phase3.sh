#!/usr/bin/env bash
# Phase 3 batch launcher (Linux port of run_phase3.ps1):
#   10 algorithms x H1/H2 x 10 train seeds = 200 tasks
#   GC-MAPPO uses C1 config (aps=1.0, temp_end=0.30) via env vars
#   Max-Pressure skips training but still runs to satisfy paper_statistics.py completeness check
#   Serial execution (RAM-limited), skips completed tasks, resumes mid-training when possible.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

# 激活项目 venv（activate 脚本已注入 SUMO_HOME/bin 到 PATH）
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run environment setup first." >&2
    exit 1
fi

export GC_MAPPO_ACTION_PRIOR_SCALE=1.0
export GC_MAPPO_TEMP_END=0.30
export GC_MAPPO_PRESSURE_CONFIDENCE_MASK=0

methods=(
    "GC-MAPPO" "MAPPO" "IPPO" "CoLight" "PressLight" "Max-Pressure"
    "Double DQN" "Dueling Double DQN" "GAE Actor-Critic" "REINFORCE"
)

# 场景: tag | net-file | route-file | tls-min-links
scenarios=(
    "h1|nets/grid-3x3/3x3.net.xml|nets/grid-3x3/3x3-medium.rou.xml|4"
    "h2|nets/2x2-grid/2x2_oversaturated.net.xml|nets/2x2-grid/2x2_oversaturated.rou.xml|4"
)

train_seeds=(42 43 44 45 46 47 48 49 50 51)
eval_seeds=(1042 1043 1044 1045 1046 1047 1048 1049 1050 1051)

total=$(( ${#methods[@]} * ${#scenarios[@]} * ${#train_seeds[@]} ))
task_index=0
completed=0
skipped=0
failed=0
start_ts=$(date +%s)

ts()       { date '+%Y-%m-%d %H:%M:%S'; }
short_ts() { date '+%H:%M:%S'; }

# 读取 protocol_state.json 的 status 字段；文件不存在或解析失败返回空字符串
get_status() {
    python -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status', ''))
except Exception:
    print('')
" "$1" 2>/dev/null
}

# 丢弃不完整任务输出，避免旧实验数据混入新实验。
discard_dir() {
    local out_dir="$1"
    rm -rf -- "$out_dir"
}

echo "[$(ts)] Phase 3 batch start: $total tasks total"

for scen in "${scenarios[@]}"; do
    IFS='|' read -r tag net route tls <<< "$scen"
    for method in "${methods[@]}"; do
        method_dir="${method// /_}"
        for seed in "${train_seeds[@]}"; do
            task_index=$((task_index + 1))
            out_dir="results/phase3_${tag}_${method_dir}_seed${seed}"
            state_file="$out_dir/protocol_state.json"

            use_resume=false
            if [ -f "$state_file" ]; then
                status="$(get_status "$state_file")"
                if [ "$status" = "completed" ]; then
                    skipped=$((skipped + 1))
                    continue
                elif [ "$status" = "running" ]; then
                    resume_file="$out_dir/resume_state.pt"
                    if [ "$method" != "Max-Pressure" ] && [ -f "$resume_file" ]; then
                        use_resume=true
                        echo "[$(short_ts)] RESUME: $out_dir (running, resume_state.pt found) -> --resume mid-training"
                    else
                        discard_dir "$out_dir"
                        echo "[$(short_ts)] RESUME: discarded $out_dir (running, no resume_state.pt), will retrain from scratch"
                    fi
                else
                    discard_dir "$out_dir"
                    echo "[$(short_ts)] RESUME: discarded $out_dir (status=$status), will retrain"
                fi
            fi

            elapsed=$(( $(date +%s) - start_ts ))
            pct=$(awk "BEGIN{printf \"%.1f\", $task_index/$total*100}")
            resume_tag=""
            if $use_resume; then resume_tag=" [RESUME]"; fi
            echo "[$(short_ts)] Task $task_index/$total ($pct%) elapsed=${elapsed}s scenario=$tag method=$method seed=$seed outDir=$out_dir$resume_tag"

            args=(-u paper_protocol.py --method "$method" --train-seed "$seed" \
                  --eval-seeds "${eval_seeds[@]}" \
                  --net-file "$net" --route-file "$route" --tls-min-links "$tls" \
                  --output-dir "$out_dir")
            if $use_resume; then args+=(--resume); fi

            if python "${args[@]}"; then
                completed=$((completed + 1))
            else
                ec=$?
                failed=$((failed + 1))
                echo "[$(short_ts)] FAILED: $method seed=$seed scenario=$tag - exit code $ec"
            fi
        done
    done
done

duration=$(( $(date +%s) - start_ts ))
echo "[$(ts)] Phase 3 batch DONE: total=$total completed=$completed skipped=$skipped failed=$failed duration=${duration}s"
echo "Next: python paper_statistics.py --results-root results --output-dir results/phase3_statistics --bootstrap-samples 20000"
