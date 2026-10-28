"""IPPO+prior 对照重评：对 P1 50ep IPPO checkpoint 以与 GC 相同的执行先验重评。

用法：./.venv/bin/python tools/reeval_ippo_prior.py --shard 0 --num-shards 2
checkpoint 来源 results/p1_dev_v1/{h1,h2}_IPPO_seed{42,43,44}，结果写入 results/ippo_prior_eval/。
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO')
sys.path.insert(0, str(ROOT))

os.environ.setdefault('IPPO_EVAL_PRIOR_SCALE', '1.0')

import paper_protocol
import test_algorithms as training

SRC = ROOT / 'results' / 'p1_dev_v1'
PRIOR_SCALE = float(os.environ.get('IPPO_EVAL_PRIOR_SCALE', '1.0'))
DST = ROOT / 'results' / (
    'ippo_prior_eval' if PRIOR_SCALE == 1.0 else f'ippo_prior_eval_scale_{PRIOR_SCALE:g}'
)
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
    tasks = sorted(
        p for p in SRC.iterdir()
        if p.name.endswith(tuple(f'_IPPO_seed{s}' for s in (42, 43, 44)))
        and (p / 'checkpoint.pt').is_file()
    )
    shard_tasks = [p for i, p in enumerate(tasks) if i % args.num_shards == args.shard]

    for task in shard_tasks:
        out_path = DST / f'{task.name}_prior_scale_{PRIOR_SCALE:g}.json'
        if out_path.is_file():
            print(f'[SKIP] {task.name}')
            continue
        state = json.load(open(task / 'protocol_state.json'))
        env_cfg = state['environment']
        for key in ENV_KEYS:
            if key in env_cfg:
                training.ENV_PARAMS[key] = env_cfg[key]

        os.environ['IPPO_EVAL_PRIOR_SCALE'] = str(PRIOR_SCALE)
        checkpoint = str(task / 'checkpoint.pt')
        print(f'[EVAL] {task.name} (train_seed={state["train_seed"]}, prior_scale={PRIOR_SCALE:g})', flush=True)
        metrics = paper_protocol._evaluate('IPPO', checkpoint, EVAL_SEEDS)
        record = {
            'source_task': task.name,
            'method': 'IPPO',
            'variant': 'ippo_prior',
            'train_seed': state['train_seed'],
            'evaluation_seeds': EVAL_SEEDS,
            'eval_prior_scale': PRIOR_SCALE,
            'eval_policy_temperature': float(os.environ.get('IPPO_EVAL_POLICY_TEMPERATURE', '0.30')),
            'deterministic': True,
            'note': 'same raw execution-prior formula as GC; scale controlled explicitly; observation unchanged',
            'evaluation_metrics': metrics,
        }
        with open(out_path, 'w') as fh:
            json.dump(record, fh, indent=1)
        u = sum(metrics['demand_service_times']) / len(metrics['demand_service_times'])
        print(f'[DONE] {task.name}: U={u:.2f}', flush=True)


if __name__ == '__main__':
    main()
