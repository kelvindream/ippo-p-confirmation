#!/usr/bin/env python3
"""Phase 3 并行调度器（替代 run_phase3.sh 串行版）。

11 算法 x 2 场景(H1/H2) x 10 train seed = 220 任务，并行执行。
GPU 显存受限（~8GB 可用），默认并行 4，可用环境变量 PHASE3_WORKERS 覆盖。

特性：
- 串行预处理：跳过 completed、--resume 中断任务、归档失败任务（避免并行竞争）
- ThreadPoolExecutor 并行执行训练
- 每个任务独立日志 results/phase3_logs/<tag>_<method>_seed<seed>.log
- resume 失败容错：归档后无 --resume 重试一次（防 resume_state.pt 损坏导致永久失败）
"""
import os
import sys
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

VENV_PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(VENV_PY).exists():
    print(f"ERROR: venv python not found: {VENV_PY}", file=sys.stderr)
    sys.exit(1)

# GC-MAPPO C1 配置（aps=1.0, temp_end=0.30）
os.environ["GC_MAPPO_ACTION_PRIOR_SCALE"] = "1.0"
os.environ["GC_MAPPO_TEMP_END"] = "0.30"
os.environ["GC_MAPPO_PRESSURE_CONFIDENCE_MASK"] = "0"
# SUMO（activate 脚本已注入，这里兜底）
os.environ.setdefault("SUMO_HOME", "/home/zrt/sumo")
os.environ["PATH"] = f"{os.environ['SUMO_HOME']}/bin:" + os.environ.get("PATH", "")

MAX_WORKERS = int(os.environ.get("PHASE3_WORKERS", "4"))

METHODS = [
    "GC-MAPPO", "MAPPO", "IPPO", "CoLight", "PressLight", "Max-Pressure",
    "Fixed-Time", "Double DQN",
    "Dueling Double DQN", "GAE Actor-Critic",
    "REINFORCE",
]
SCENARIOS = [
    {"tag": "h1", "net": "nets/grid-3x3/3x3.net.xml",
     "route": "nets/grid-3x3/3x3-medium.rou.xml", "tls": 4},
    {"tag": "h2", "net": "nets/2x2-grid/2x2_oversaturated.net.xml",
     "route": "nets/2x2-grid/2x2_oversaturated.rou.xml", "tls": 4},
]
TRAIN_SEEDS = list(range(42, 52))
EVAL_SEEDS = list(range(1042, 1052))

ARCHIVE_ROOT = ROOT / "_interrupted_archive"
LOG_DIR = ROOT / "results" / "phase3_logs"


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def short_ts():
    return time.strftime("%H:%M:%S")


def method_dir(m):
    return m.replace(" ", "_")


def get_status(state_file):
    try:
        with open(state_file) as f:
            return json.load(f).get("status", "")
    except Exception:
        return ""


def archive_dir(out_dir):
    """归档输出目录到 _interrupted_archive/<basename>_<timestamp>"""
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_ROOT / f"{Path(out_dir).name}_{stamp}"
    shutil.move(str(out_dir), str(dest))
    return dest


def preprocess():
    """串行扫描所有任务，做 skip/resume/archive 判断，返回待执行任务列表。"""
    tasks = []
    skipped = 0
    for scen in SCENARIOS:
        for m in METHODS:
            md = method_dir(m)
            for seed in TRAIN_SEEDS:
                out_dir = ROOT / "results" / f"phase3_{scen['tag']}_{md}_seed{seed}"
                state_file = out_dir / "protocol_state.json"
                use_resume = False
                if state_file.exists():
                    status = get_status(state_file)
                    if status == "completed":
                        skipped += 1
                        continue
                    elif status == "running":
                        resume_file = out_dir / "resume_state.pt"
                        if m not in ("Max-Pressure", "Fixed-Time") and resume_file.exists():
                            use_resume = True
                            print(f"[{short_ts()}] RESUME: {out_dir.name} (running, resume_state.pt found) -> --resume")
                        else:
                            dest = archive_dir(out_dir)
                            print(f"[{short_ts()}] RESUME: archived {out_dir.name} -> {dest.name} (running, no resume_state.pt), retrain")
                    else:
                        dest = archive_dir(out_dir)
                        print(f"[{short_ts()}] RESUME: archived {out_dir.name} -> {dest.name} (status={status}), retrain")
                tasks.append({
                    "method": m, "seed": seed, "scen": scen,
                    "out_dir": out_dir, "use_resume": use_resume,
                })
    return tasks, skipped


def build_cmd(task, use_resume):
    m = task["method"]; seed = task["seed"]; scen = task["scen"]
    out_dir = task["out_dir"]
    cmd = [
        VENV_PY, "-u", "paper_protocol.py",
        "--method", m, "--train-seed", str(seed),
        "--eval-seeds", *[str(s) for s in EVAL_SEEDS],
        "--net-file", scen["net"], "--route-file", scen["route"],
        "--tls-min-links", str(scen["tls"]),
        "--output-dir", str(out_dir),
    ]
    if use_resume:
        cmd.append("--resume")
    return cmd


def run_task(task, idx, total):
    """执行单个训练任务。stdout/stderr 直接写独立日志文件（实时）。
    --resume 失败时归档后无 --resume 重试一次。"""
    m = task["method"]; seed = task["seed"]; scen = task["scen"]
    out_dir = task["out_dir"]
    md = method_dir(m)
    log_file = LOG_DIR / f"{scen['tag']}_{md}_seed{seed}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def run(use_resume):
        cmd = build_cmd(task, use_resume)
        with open(log_file, "a") as lf:
            tag = f"[{short_ts()}] Task {idx}/{total} {scen['tag']}/{m}/seed{seed}"
            lf.write(f"{tag} START{' [RESUME]' if use_resume else ''}\n")
            lf.flush()
            # stdout 直接写文件（实时），不占用主进程内存
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            return proc.returncode

    print(f"[{short_ts()}] Task {idx}/{total} START {scen['tag']}/{m}/seed{seed}{' [RESUME]' if task['use_resume'] else ''}")
    rc = run(task["use_resume"])
    if rc == 0:
        print(f"[{short_ts()}] Task {idx}/{total} OK    {scen['tag']}/{m}/seed{seed}")
        return True, None

    # resume 失败容错：归档 + 无 --resume 重试一次
    if task["use_resume"]:
        dest = archive_dir(out_dir)
        with open(log_file, "a") as lf:
            lf.write(f"[{short_ts()}] resume FAILED (rc={rc}), archived -> {dest.name}, retrain from scratch\n")
        print(f"[{short_ts()}] Task {idx}/{total} RESUME-FAIL -> retrain {scen['tag']}/{m}/seed{seed}")
        rc2 = run(False)
        if rc2 == 0:
            print(f"[{short_ts()}] Task {idx}/{total} OK(retrain) {scen['tag']}/{m}/seed{seed}")
            return True, None

    print(f"[{short_ts()}] Task {idx}/{total} FAIL  {scen['tag']}/{m}/seed{seed} rc={rc}")
    return False, f"rc={rc}"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"[{ts()}] Phase 3 PARALLEL batch start (max_workers={MAX_WORKERS})")
    tasks, skipped = preprocess()
    total = len(tasks)
    print(f"[{ts()}] Preprocess done: {total} to run, {skipped} skipped (already completed)")
    if not tasks:
        print(f"[{ts()}] Nothing to do. All tasks already completed.")
        return

    completed = 0
    failed = 0
    fail_details = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_task, t, i + 1, total): t for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            ok, err = fut.result()
            if ok:
                completed += 1
            else:
                failed += 1
                t = futures[fut]
                fail_details.append((t["scen"]["tag"], t["method"], t["seed"], err))

    duration = int(time.time() - start)
    print(f"[{ts()}] Phase 3 PARALLEL DONE: total={total} completed={completed} "
          f"skipped={skipped} failed={failed} duration={duration}s")
    if failed:
        print("Failed tasks:")
        for tag, m, seed, err in fail_details:
            print(f"  - {tag}/{m}/seed{seed} ({err})")
    print("Next: python paper_statistics.py --results-root results "
          "--output-dir results/phase3_statistics --bootstrap-samples 20000")


if __name__ == "__main__":
    main()
