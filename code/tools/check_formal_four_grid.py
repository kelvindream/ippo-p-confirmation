#!/usr/bin/env python3
"""Summarize the formal four-grid task matrix without changing task state."""

import argparse
import json
from collections import Counter
from pathlib import Path

METHODS = ('IPPO', 'IPPO-P', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P')
SCENARIOS = ('H1', 'H2', 'H3')
SEEDS = (42, 43, 44, 45, 46)


def load_status(task_dir):
    state_path = task_dir / 'protocol_state.json'
    if not state_path.is_file():
        return 'not_started', None
    try:
        with state_path.open(encoding='utf-8') as handle:
            state = json.load(handle)
    except Exception as error:
        return 'invalid', {'error': str(error)}
    return state.get('status', 'unknown'), state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/formal_four_grid_v1')
    args = parser.parse_args()
    root = Path(args.root)
    counts = Counter()
    problems = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for seed in SEEDS:
                name = f'{scenario}_{method}_seed{seed}'
                status, state = load_status(root / name)
                counts[status] += 1
                if state and status in ('running', 'completed'):
                    variant = state.get('variant_config')
                    if variant is None:
                        problems.append(f'{name}: missing variant_config')
                    environment = state.get('environment', {})
                    route = environment.get('route_file', '')
                    expected = {
                        'H1': '3x3-medium.rou.xml',
                        'H2': '2x2-formal-h2.rou.xml',
                        'H3': '2x2-formal-h3.rou.xml',
                    }[scenario]
                    if not route.endswith(expected):
                        problems.append(f'{name}: route={route!r}, expected *{expected}')
    print('Formal four-grid status')
    print(' '.join(f'{key}={counts[key]}' for key in (
        'completed', 'running', 'not_started', 'invalid', 'unknown'
    )))
    if problems:
        print('Audit problems:')
        for problem in problems:
            print(f'- {problem}')
        raise SystemExit(2)
    print('Audit problems: none')


if __name__ == '__main__':
    main()
