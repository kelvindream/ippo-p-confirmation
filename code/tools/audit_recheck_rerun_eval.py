"""AUD-01 forensic recheck: re-run evaluation from a stored checkpoint.

Reproduces the exact evaluation pass of a formal_four_grid_v1 task
(method IPPO here) without any training: reload the stored checkpoint,
restore the environment snapshot from protocol_state.json, and call
paper_protocol._evaluate with the task's stored evaluation seeds.

Usage:
  .venv/bin/python tools/audit_recheck_rerun_eval.py \
      --task-dir results/formal_four_grid_v1/H2_IPPO_seed45 \
      --out-dir results/audit_recheck_v1/H2_IPPO_seed45_rerun
"""

import argparse
import json
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO')

import os
import sys

sys.path.insert(0, str(ROOT))

import paper_protocol
import test_algorithms as training

# Keys of the protocol environment snapshot that must be restored before
# evaluation (multi_agent is forced True and sumo_seed is passed per-episode
# by paper_protocol._evaluate, so they are excluded here).
ENV_KEYS = [
    'net_file', 'route_file', 'skip_range', 'simulation_time', 'yellow_time',
    'delta_rs_update_time', 'use_gui', 'tls_min_links',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    task_dir = (ROOT / args.task_dir) if not os.path.isabs(args.task_dir) else Path(args.task_dir)
    out_dir = (ROOT / args.out_dir) if not os.path.isabs(args.out_dir) else Path(args.out_dir)

    state = json.load(open(task_dir / 'protocol_state.json'))
    method = state['method']
    eval_seeds = list(state['evaluation_seeds'])
    for key in ENV_KEYS:
        if key in state['environment']:
            training.ENV_PARAMS[key] = state['environment'][key]
    checkpoint = task_dir / 'checkpoint.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint missing: {checkpoint}')

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[AUD-01] rerun eval: task={task_dir.name} method={method} '
          f'seeds={eval_seeds} ckpt_sha=see_report', flush=True)
    metrics = paper_protocol._evaluate(method, str(checkpoint), eval_seeds)

    stored = json.load(open(task_dir / 'evaluation_metrics.json'))
    rerun_u = metrics['demand_service_times']
    stored_u = stored['demand_service_times']
    max_abs_diff = max(abs(a - b) for a, b in zip(rerun_u, stored_u))

    record = {
        'purpose': 'AUD-01 forensic recheck of suspected file-copy incident',
        'source_task_dir': str(task_dir.relative_to(ROOT)),
        'method': method,
        'train_seed': state['train_seed'],
        'checkpoint_path': str(checkpoint.relative_to(ROOT)),
        'evaluation_seeds': eval_seeds,
        'evaluation_mode': state['evaluation_mode'],
        'environment_restored': {k: state['environment'].get(k) for k in ENV_KEYS},
        'rerun_evaluation_metrics': metrics,
        'stored_demand_service_times': stored_u,
        'rerun_demand_service_times': rerun_u,
        'max_abs_diff_rerun_vs_stored': max_abs_diff,
    }
    with open(out_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    print(f'[AUD-01] rerun U vector: {rerun_u!r}', flush=True)
    print(f'[AUD-01] stored U vector: {stored_u!r}', flush=True)
    print(f'[AUD-01] max |rerun - stored| = {max_abs_diff:.3e}', flush=True)
    verdict = 'reproduced' if max_abs_diff <= 1e-9 else 'NOT reproduced'
    print(f'[AUD-01] {task_dir.name}: {verdict}', flush=True)


if __name__ == '__main__':
    main()
