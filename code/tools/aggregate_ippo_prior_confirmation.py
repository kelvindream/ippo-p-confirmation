#!/usr/bin/env python3
"""Frozen primary analysis for IPPO-P vs IPPO confirmation experiment."""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/zrt/ai/GC-MAPPO')
CONF_ROOT = ROOT / 'results/ippo_prior_confirmation_v1'
PILOT_ROOT = ROOT / 'results/formal_four_grid_v1'
METHODS = ('IPPO', 'IPPO-P')
SCENARIOS = ('H1', 'H2', 'H3')
CONF_SEEDS = tuple(range(47, 57))
PILOT_SEEDS = (42, 43, 44, 45, 46)
B = 20000
RNG_SEED = 20261024


def load_task(root, method, scenario, seed):
    task = root / f'{scenario}_{method}_seed{seed}'
    state = json.loads((task / 'protocol_state.json').read_text())
    metrics = json.loads((task / 'evaluation_metrics.json').read_text())
    if state.get('status') != 'completed':
        raise ValueError(f'{task}: task not completed')
    if state.get('evaluation_seeds') != [1042, 1043, 1044, 1045, 1046]:
        raise ValueError(f'{task}: evaluation seeds mismatch')
    u = np.asarray(metrics['demand_service_times'], dtype=np.float64)
    if u.size != 5 or not np.all(np.isfinite(u)) or np.any(u <= 0):
        raise ValueError(f'{task}: invalid U values {u}')
    if sum(metrics['teleport_counts']) != 0:
        raise ValueError(f'{task}: teleport detected')
    if not all(metrics['demand_conservation_ok']):
        raise ValueError(f'{task}: demand conservation failed')
    return float(u.mean())


def collect(root, seeds):
    return {
        (method, scenario, seed): load_task(root, method, scenario, seed)
        for method in METHODS for scenario in SCENARIOS for seed in seeds
    }


def seed_effects(cells, seeds):
    return np.asarray([
        np.mean([
            np.log(cells[('IPPO', scenario, seed)] /
                   cells[('IPPO-P', scenario, seed)])
            for scenario in SCENARIOS
        ])
        for seed in seeds
    ])


def scenario_effects(cells, seeds, scenario):
    return np.asarray([
        np.log(cells[('IPPO', scenario, seed)] /
               cells[('IPPO-P', scenario, seed)])
        for seed in seeds
    ])


def exact_sign_flip_p(values):
    n = values.size
    observed = abs(float(values.mean()))
    extreme = 0
    for mask in range(1 << n):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        statistic = abs(float((values * signs).mean()))
        extreme += statistic >= observed - 1e-15
    return float(extreme / (1 << n))


def summarize(values, rng, include_test):
    index = rng.integers(0, values.size, size=(B, values.size))
    boot = values[index].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = float(values.mean())
    result = {
        'mean_log_ratio': point,
        'improvement_pct': float((1.0 - np.exp(-point)) * 100.0),
        'bootstrap_ci95_pct': [
            float((1.0 - np.exp(-lo)) * 100.0),
            float((1.0 - np.exp(-hi)) * 100.0),
        ],
        'seed_log_ratios': values.tolist(),
        'seed_wins': int(np.sum(values > 0)),
        'n_seeds': int(values.size),
    }
    if include_test:
        result.update({
            'exact_two_sided_sign_flip_p': exact_sign_flip_p(values),
            'alpha': 0.05,
            'confirmed': bool(point > 0 and exact_sign_flip_p(values) < 0.05),
        })
    return result


def main():
    protocol = json.loads((CONF_ROOT / 'protocol_frozen.json').read_text())
    if not protocol.get('frozen_before_new_data'):
        raise ValueError('protocol is not frozen')

    confirmation = collect(CONF_ROOT, CONF_SEEDS)
    pilot = collect(PILOT_ROOT, PILOT_SEEDS)
    rng = np.random.default_rng(RNG_SEED)

    primary_values = seed_effects(confirmation, CONF_SEEDS)
    primary = summarize(primary_values, rng, include_test=True)

    per_scenario = {
        scenario: summarize(
            scenario_effects(confirmation, CONF_SEEDS, scenario),
            rng,
            include_test=False,
        )
        for scenario in SCENARIOS
    }

    pooled_cells = {**pilot, **confirmation}
    pooled_seeds = PILOT_SEEDS + CONF_SEEDS
    pooled = summarize(seed_effects(pooled_cells, pooled_seeds), rng, include_test=False)

    output = {
        'protocol_name': 'ippo_prior_confirmation_v1',
        'primary_population': 'new independent training seeds 47-56 only',
        'primary_contrast': 'IPPO-P vs IPPO',
        'primary_metric': 'demand_service_time',
        'primary': primary,
        'secondary_per_scenario_confirmation_seeds': per_scenario,
        'secondary_pooled_15_seeds_descriptive_only': pooled,
        'bootstrap_resamples': B,
        'bootstrap_seed': RNG_SEED,
        'warning': 'Pilot seeds 42-46 are excluded from the primary hypothesis test.',
    }
    path = CONF_ROOT / 'confirmation_summary.json'
    path.write_text(json.dumps(output, indent=2))

    print(f'Saved: {path}')
    print(
        f"PRIMARY IPPO-P vs IPPO: {primary['improvement_pct']:+.4f}% "
        f"CI=[{primary['bootstrap_ci95_pct'][0]:+.4f},"
        f"{primary['bootstrap_ci95_pct'][1]:+.4f}] "
        f"wins={primary['seed_wins']}/{primary['n_seeds']} "
        f"exact_p={primary['exact_two_sided_sign_flip_p']:.6f} "
        f"confirmed={primary['confirmed']}"
    )


if __name__ == '__main__':
    main()
