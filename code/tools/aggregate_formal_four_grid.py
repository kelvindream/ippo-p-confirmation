#!/usr/bin/env python3
"""Formal aggregation for the four-grid confirmation matrix.

Protocol 8.2:
1. per training seed, average U over fixed evaluation seeds;
2. per scenario, compute log(U_ref / U_new);
3. equal-weight mean over scenarios;
4. mean over training seeds;
5. improvement = 1 - exp(-mean_log_ratio).
Uncertainty: paired percentile bootstrap over training seeds (20000 resamples).
Inference: exact two-sided paired sign-flip test over training seeds,
with Holm correction across the contrast family.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/zrt/ai/GC-MAPPO/results/formal_four_grid_v1')
METHODS = ('IPPO', 'IPPO-P', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P')
SCENARIOS = ('H1', 'H2', 'H3')
SEEDS = (42, 43, 44, 45, 46)
B = 20000
rng = np.random.default_rng(20261024)

CONTRASTS = (
    ('IPPO-P', 'IPPO', 'prior effect within IPPO'),
    ('GC-MAPPO-ind-P', 'GC-MAPPO-ind-noP', 'prior effect within GC'),
    ('GC-MAPPO-ind-noP', 'IPPO', 'GC(no prior) vs IPPO'),
    ('GC-MAPPO-ind-P', 'IPPO', 'GC(prior) vs IPPO'),
    ('INTERACTION', None, '(GC prior effect) - (IPPO prior effect)'),
)


def load_u(method, scenario, seed):
    path = ROOT / f'{scenario}_{method}_seed{seed}' / 'evaluation_metrics.json'
    data = json.loads(path.read_text())
    return np.asarray(data['demand_service_times'], dtype=np.float64)


def load_sanity(method, scenario, seed):
    path = ROOT / f'{scenario}_{method}_seed{seed}' / 'evaluation_metrics.json'
    data = json.loads(path.read_text())
    return {
        'completion': float(np.mean(data['completion_rates'])),
        'planned_completion': float(np.mean(data['planned_completion_rates'])),
        'teleport': float(np.sum(data['teleport_counts'])),
        'conservation_ok': bool(all(data['demand_conservation_ok'])),
    }


def seed_log_ratios(new, ref):
    """new/ref: dict seed -> mean-U. Returns array over seeds of
    mean-over-scenarios log(U_ref/U_new)."""
    log_rows = []
    for seed in SEEDS:
        per_scenario = []
        for scenario in SCENARIOS:
            per_scenario.append(np.log(ref[scenario][seed] / new[scenario][seed]))
        log_rows.append(np.mean(per_scenario))
    return np.asarray(log_rows)


def exact_sign_flip_p(log_ratios):
    """Exact two-sided paired randomization test over training seeds."""
    n = log_ratios.size
    observed = abs(float(log_ratios.mean()))
    null_stats = []
    for mask in range(1 << n):
        signs = np.asarray([
            1.0 if mask & (1 << i) else -1.0 for i in range(n)
        ])
        null_stats.append(abs(float((log_ratios * signs).mean())))
    return float(np.mean(np.asarray(null_stats) >= observed - 1e-15))


def bootstrap(log_ratios):
    n = log_ratios.size
    idx = rng.integers(0, n, size=(B, n))
    stats = log_ratios[idx].mean(axis=1)
    point = log_ratios.mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        'mean_log_ratio': float(point),
        'improvement_pct': float((1.0 - np.exp(-point)) * 100.0),
        'ci95_pct': [float((1.0 - np.exp(-lo)) * 100.0),
                     float((1.0 - np.exp(-hi)) * 100.0)],
        'seed_log_ratios': log_ratios.tolist(),
        'seed_wins': int(np.sum(log_ratios > 0)),
        'p_value': exact_sign_flip_p(log_ratios),
        'p_method': 'exact two-sided paired sign-flip test',
    }


def holm(p_values):
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    means = {}
    sanity = {}
    for method in METHODS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                u = load_u(method, scenario, seed)
                if u.size != 5 or not np.all(np.isfinite(u)) or np.any(u <= 0):
                    raise ValueError(
                        f'invalid U values for {method}|{scenario}|seed{seed}: {u}'
                    )
                check = load_sanity(method, scenario, seed)
                if check['teleport'] != 0 or not check['conservation_ok']:
                    raise ValueError(
                        f'failed sanity check for {method}|{scenario}|seed{seed}: {check}'
                    )
                means.setdefault((method, scenario), {})[seed] = float(u.mean())
                sanity[(method, scenario, seed)] = check

    raw_p = []
    results = {}
    for new, ref, label in CONTRASTS:
        if new == 'INTERACTION':
            gc = seed_log_ratios(
                {s: means[('GC-MAPPO-ind-P', s)] for s in SCENARIOS},
                {s: means[('GC-MAPPO-ind-noP', s)] for s in SCENARIOS},
            )
            ip = seed_log_ratios(
                {s: means[('IPPO-P', s)] for s in SCENARIOS},
                {s: means[('IPPO', s)] for s in SCENARIOS},
            )
            ratios = gc - ip
        else:
            ratios = seed_log_ratios(
                {s: means[(new, s)] for s in SCENARIOS},
                {s: means[(ref, s)] for s in SCENARIOS},
            )
        entry = bootstrap(ratios)
        entry['label'] = label
        results[f'{new}_vs_{ref}'] = entry
        raw_p.append(entry['p_value'])

    adjusted = holm(np.asarray(raw_p))
    for (new, ref, _), p_h in zip(CONTRASTS, adjusted):
        results[f'{new}_vs_{ref}']['p_holm'] = float(p_h)

    summary = {
        'matrix': 'formal_four_grid_v1',
        'metric': 'demand_service_time (U, lower is better)',
        'aggregation': 'protocol 8.2: seed-mean over eval seeds, log-ratio, '
                       'equal-weight scenarios, mean over 5 training seeds',
        'bootstrap_resamples': B,
        'bootstrap_seed': 20261024,
        'inference': 'exact two-sided paired sign-flip test over 5 training seeds; Holm correction over 5 contrasts',
        'contrasts': results,
        'mean_U_by_cell': {
            f'{method}|{scenario}': {
                str(seed): means[(method, scenario)][seed] for seed in SEEDS
            }
            for method in METHODS for scenario in SCENARIOS
        },
        'sanity': {
            f'{method}|{scenario}': {
                'mean_completion': float(np.mean([
                    sanity[(method, scenario, s)]['completion'] for s in SEEDS])),
                'mean_planned_completion': float(np.mean([
                    sanity[(method, scenario, s)]['planned_completion'] for s in SEEDS])),
                'total_teleport': float(np.sum([
                    sanity[(method, scenario, s)]['teleport'] for s in SEEDS])),
                'conservation_all_ok': all(
                    sanity[(method, scenario, s)]['conservation_ok'] for s in SEEDS),
            }
            for method in METHODS for scenario in SCENARIOS
        },
    }

    out = ROOT / 'aggregation_summary.json'
    out.write_text(json.dumps(summary, indent=2))

    print(f'Saved: {out}\n')
    print(f"{'contrast':44s} {'improv%':>9s} {'95% CI':>18s} "
          f"{'wins':>5s} {'p':>8s} {'p_holm':>8s}")
    for new, ref, label in CONTRASTS:
        e = results[f'{new}_vs_{ref}']
        ci = e['ci95_pct']
        print(f"{f'{new} vs {ref}':44s} {e['improvement_pct']:+9.4f} "
              f"[{ci[0]:+8.4f},{ci[1]:+8.4f}] "
              f"{e['seed_wins']:d}/5 {e['p_value']:8.4f} {e['p_holm']:8.4f}")

    print('\nSanity (per method|scenario, mean over seeds):')
    for method in METHODS:
        for scenario in SCENARIOS:
            s = summary['sanity'][f'{method}|{scenario}']
            flag = '' if s['conservation_all_ok'] else ' CONSERVATION_FAIL'
            print(f"{f'{method}|{scenario}':40s} "
                  f"completion={s['mean_completion']*100:6.2f}% "
                  f"planned={s['mean_planned_completion']*100:6.2f}% "
                  f"teleport={s['total_teleport']:7.1f}{flag}")


if __name__ == '__main__':
    main()
