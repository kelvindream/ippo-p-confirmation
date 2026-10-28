#!/usr/bin/env python3
"""Audit status and frozen bindings for IPPO prior confirmation tasks."""

import argparse
import json
from collections import Counter
from pathlib import Path

METHODS = ('IPPO', 'IPPO-P')
SCENARIOS = ('H1', 'H2', 'H3')
SEEDS = tuple(range(47, 57))
EXPECTED_VARIANTS = {
    'IPPO': {
        'family': 'IPPO', 'actor_type': 'independent_local',
        'prior_mode': 'none', 'action_prior_scale': 0.0,
        'pressure_observation': False, 'soft_q': False,
        'policy_temperature': 1.0, 'hold_prior_bonus': 0.0,
    },
    'IPPO-P': {
        'family': 'IPPO', 'actor_type': 'independent_local',
        'prior_mode': 'pressure_zscore_hold', 'action_prior_scale': 2.25,
        'pressure_observation': False, 'soft_q': False,
        'policy_temperature': 1.0, 'hold_prior_bonus': 0.0,
    },
}
EXPECTED_ROUTES = {
    'H1': '3x3-medium.rou.xml',
    'H2': '2x2-formal-h2.rou.xml',
    'H3': '2x2-formal-h3.rou.xml',
}


def load_state(task_dir):
    state_path = task_dir / 'protocol_state.json'
    if not state_path.is_file():
        return 'not_started', None
    try:
        return_state = json.loads(state_path.read_text())
    except Exception as error:
        return 'invalid', {'error': str(error)}
    return return_state.get('status', 'unknown'), return_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/ippo_prior_confirmation_v1')
    args = parser.parse_args()
    root = Path(args.root)
    counts = Counter()
    problems = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for seed in SEEDS:
                name = f'{scenario}_{method}_seed{seed}'
                status, state = load_state(root / name)
                counts[status] += 1
                if state and status in ('running', 'completed'):
                    if state.get('method') != method:
                        problems.append(f'{name}: method={state.get("method")!r}')
                    if state.get('train_seed') != seed or state.get('sumo_seed') != seed:
                        problems.append(f'{name}: seed mismatch')
                    if state.get('train_episodes') != 50:
                        problems.append(f'{name}: train_episodes mismatch')
                    if state.get('evaluation_seeds') != [1042, 1043, 1044, 1045, 1046]:
                        problems.append(f'{name}: evaluation_seeds mismatch')
                    if state.get('variant_config') != EXPECTED_VARIANTS[method]:
                        problems.append(f'{name}: variant_config mismatch')
                    route = state.get('environment', {}).get('route_file', '')
                    if not route.endswith(EXPECTED_ROUTES[scenario]):
                        problems.append(f'{name}: route={route!r}')
    print('IPPO prior confirmation status')
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
