"""AUD-01 forensic recheck: trace the full deterministic action sequence of one
evaluation episode for a stored IPPO checkpoint.

Purpose: decide whether two checkpoints with different weights (bit-different
tensors) produce identical evaluation output because their deterministic
policies coincide on the real evaluation-trajectory state distribution.

Usage:
  .venv/bin/python tools/audit_recheck_trace_actions.py \
      --task-dir results/formal_four_grid_v1/H2_IPPO_seed45 \
      --eval-seed 1042 --out results/audit_recheck_v1/trace_seed45_ep1042.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO')
sys.path.insert(0, str(ROOT))

import numpy as np

import paper_protocol
import test_algorithms as training

ENV_KEYS = [
    'net_file', 'route_file', 'skip_range', 'simulation_time', 'yellow_time',
    'delta_rs_update_time', 'use_gui', 'tls_min_links',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--eval-seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    task_dir = (ROOT / args.task_dir) if not os.path.isabs(args.task_dir) else Path(args.task_dir)
    state = json.load(open(task_dir / 'protocol_state.json'))
    for key in ENV_KEYS:
        if key in state['environment']:
            training.ENV_PARAMS[key] = state['environment'][key]

    seed = args.eval_seed
    paper_protocol._seed_everything(seed)
    agent = paper_protocol._build_evaluator('IPPO', seed, str(task_dir / 'checkpoint.pt'))
    paper_protocol._seed_everything(seed)
    env = paper_protocol._environment(seed)

    actions_seq = []
    states = env.reset()
    invalid_actions = [False] * env.num_agents
    done = False
    throughput = 0
    while not done:
        actions = paper_protocol._select_actions('IPPO', agent, env, states, invalid_actions)
        actions_seq.append(list(actions))
        next_states, _, done, info = env.step(actions)
        throughput += int(info.get('global_metrics', {}).get('arrived_number', 0))
        for index, next_state in enumerate(next_states):
            invalid_actions[index] = next_state is None and not done
        states = [
            training._normalize_state_dim(next_state, len(states[index]))
            if next_state is not None else states[index]
            for index, next_state in enumerate(next_states)
        ]
    episode_metrics = env._collect_global_metrics()
    env.close()

    blob = json.dumps(actions_seq, separators=(',', ':')).encode()
    record = {
        'source_task_dir': str(task_dir.relative_to(ROOT)),
        'eval_seed': seed,
        'num_decision_steps': len(actions_seq),
        'actions_sha256': hashlib.sha256(blob).hexdigest(),
        'demand_service_time': episode_metrics.get('demand_service_time'),
        'throughput': throughput,
        'first_20_actions': actions_seq[:20],
    }
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=1)[:800], flush=True)


if __name__ == '__main__':
    main()
