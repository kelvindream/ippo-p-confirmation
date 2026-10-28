"""O2 re-eval: reload gc_opt_v1 checkpoints with pure argmax (no execution prior).

Usage: ./.venv/bin/python tools/reeval_gc_o2.py --shard 0 --num-shards 3
Results are written to results/gc_o2_reeval/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO')
sys.path.insert(0, str(ROOT))

os.environ.setdefault('GC_MAPPO_EVAL_PRIOR_SCALE', '0')

import paper_protocol
import test_algorithms as training

ROOT = Path('/home/zrt/ai/GC-MAPPO')
sys.path.insert(0, str(ROOT))

os.environ.setdefault('GC_MAPPO_EVAL_PRIOR_SCALE', '0')

import paper_protocol
import test_algorithms as training

ROOT = Path('/home/zrt/ai/GC-MAPPO')
SRC = ROOT / 'results' / 'gc_opt_v1'
DST = ROOT / 'results' / 'gc_o2_reeval'
EVAL_SEEDS = [1042, 1043, 1044]
ENV_KEYS = [
    'net_file', 'route_file', 'skip_range', 'simulation_time', 'yellow_time',
    'delta_rs_update_time', 'use_gui', 'multi_agent', 'tls_min_links',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p for p in SRC.iterdir() if (p / 'checkpoint.pt').is_file())
    shard_tasks = [p for i, p in enumerate(tasks) if i % args.num_shards == args.shard]

    for task in shard_tasks:
        out_path = DST / f'{task.name}.json'
        if out_path.is_file():
            print(f'[SKIP] {task.name}')
            continue
        state = json.load(open(task / 'protocol_state.json'))
        env_cfg = state['environment']
        for key in ENV_KEYS:
            if key in env_cfg:
                training.ENV_PARAMS[key] = env_cfg[key]

        variant_is_ind = '_gc_ind_' in task.name
        os.environ['GC_MAPPO_INDEPENDENT_ACTOR'] = '1' if variant_is_ind else '0'
        os.environ['GC_MAPPO_EVAL_PRIOR_SCALE'] = '0'

        checkpoint = str(task / 'checkpoint.pt')
        train_seed = state['train_seed']
        print(f'[EVAL] {task.name} (train_seed={train_seed}, '
              f'independent={variant_is_ind}, prior_scale=0)', flush=True)
        metrics = paper_protocol._evaluate('GC-MAPPO', checkpoint, EVAL_SEEDS)
        record = {
            'source_task': task.name,
            'method': 'GC-MAPPO',
            'variant': 'gc_ind' if variant_is_ind else 'gc_base',
            'train_seed': train_seed,
            'evaluation_seeds': EVAL_SEEDS,
            'eval_prior_scale': 0.0,
            'eval_temperature': float(os.environ.get('GC_MAPPO_EVAL_POLICY_TEMPERATURE', '0.30')),
            'deterministic': True,
            'evaluation_metrics': metrics,
        }
        with open(out_path, 'w') as fh:
            json.dump(record, fh, indent=1)
        u = sum(metrics['demand_service_times']) / len(metrics['demand_service_times'])
        print(f'[DONE] {task.name}: U={u:.2f}', flush=True)


if __name__ == '__main__':
    main()
