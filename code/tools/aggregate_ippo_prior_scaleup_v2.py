#!/usr/bin/env python3
"""Scale-up analysis (3x3, 5400 veh/h): same frozen estimator as confirmation v1.

Single scenario H1SAT; primary statistic is the per-seed log-ratio, tested with
the exact two-sided paired sign-flip test. Protocol copied, not re-registered.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/zrt/ai/GC-MAPPO')
SCALE_ROOT = ROOT / 'results/ippo_prior_scaleup_v2'
CONF_SUMMARY = ROOT / 'results/ippo_prior_confirmation_v1/confirmation_summary.json'
METHODS = ('IPPO', 'IPPO-P')
SEEDS = tuple(range(67, 72))
EVAL_SEEDS = [1042, 1043, 1044, 1045, 1046]
B = 20000
RNG_SEED = 20261026


def load_task(method, seed):
    task = SCALE_ROOT / f'H1SAT_{method}_seed{seed}'
    state = json.loads((task / 'protocol_state.json').read_text())
    metrics = json.loads((task / 'evaluation_metrics.json').read_text())
    if state.get('status') != 'completed':
        raise ValueError(f'{task}: task not completed')
    if state.get('evaluation_seeds') != EVAL_SEEDS:
        raise ValueError(f'{task}: evaluation seeds mismatch')
    u = np.asarray(metrics['demand_service_times'], dtype=np.float64)
    if u.size != 5 or not np.all(np.isfinite(u)) or np.any(u <= 0):
        raise ValueError(f'{task}: invalid U values {u}')
    if sum(metrics['teleport_counts']) != 0:
        raise ValueError(f'{task}: teleport detected')
    if not all(metrics['demand_conservation_ok']):
        raise ValueError(f'{task}: demand conservation failed')
    return {
        'u_mean': float(u.mean()),
        'u_values': u.tolist(),
        'completion_rates': [float(x) for x in metrics['completion_rates']],
        'planned_completion_rates': [float(x) for x in metrics['planned_completion_rates']],
    }


def exact_sign_flip_p(values):
    n = values.size
    observed = abs(float(values.mean()))
    extreme = 0
    for mask in range(1 << n):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        statistic = abs(float((values * signs).mean()))
        extreme += statistic >= observed - 1e-15
    return float(extreme / (1 << n))


def main():
    cells = {(m, s): load_task(m, s) for m in METHODS for s in SEEDS}
    rng = np.random.default_rng(RNG_SEED)

    log_ratios = np.asarray([
        np.log(cells[('IPPO', s)]['u_mean'] / cells[('IPPO-P', s)]['u_mean'])
        for s in SEEDS
    ])
    point = float(log_ratios.mean())
    index = rng.integers(0, log_ratios.size, size=(B, log_ratios.size))
    boot = log_ratios[index].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_value = exact_sign_flip_p(log_ratios)

    ippo_u = np.asarray([cells[('IPPO', s)]['u_mean'] for s in SEEDS])
    ippop_u = np.asarray([cells[('IPPO-P', s)]['u_mean'] for s in SEEDS])

    output = {
        'protocol_name': 'ippo_prior_scaleup_v2',
        'basis': 'frozen protocol of ippo_prior_confirmation_v1 copied, not re-registered',
        'scenario': 'H1SAT: 3x3 grid, 18 flows x 300 veh/h = 5400 veh/h (600/intersection)',
        'contrast': 'IPPO-P vs IPPO',
        'metric': 'demand_service_time',
        'seeds': list(SEEDS),
        'per_seed': {
            str(s): {
                'IPPO': cells[('IPPO', s)]['u_mean'],
                'IPPO-P': cells[('IPPO-P', s)]['u_mean'],
                'log_ratio': float(lr),
                'ippo_completion_rate_min': min(cells[('IPPO', s)]['completion_rates']),
                'ippop_completion_rate_min': min(cells[('IPPO-P', s)]['completion_rates']),
            }
            for s, lr in zip(SEEDS, log_ratios)
        },
        'primary': {
            'mean_log_ratio': point,
            'improvement_pct': float((1.0 - np.exp(-point)) * 100.0),
            'bootstrap_ci95_pct': [
                float((1.0 - np.exp(-lo)) * 100.0),
                float((1.0 - np.exp(-hi)) * 100.0),
            ],
            'seed_wins': int(np.sum(log_ratios > 0)),
            'n_seeds': int(log_ratios.size),
            'exact_two_sided_sign_flip_p': p_value,
            'alpha': 0.05,
            'confirmed': bool(point > 0 and p_value < 0.05),
        },
        'descriptive': {
            'ippo_u_mean_across_seeds': float(ippo_u.mean()),
            'ippo_u_seed_sd': float(ippo_u.std(ddof=1)),
            'ippo_p_u_mean_across_seeds': float(ippop_u.mean()),
            'ippo_p_u_seed_sd': float(ippop_u.std(ddof=1)),
            'ippo_completion_rate_overall_min': min(
                min(cells[('IPPO', s)]['completion_rates']) for s in SEEDS
            ),
            'ippo_p_completion_rate_overall_min': min(
                min(cells[('IPPO-P', s)]['completion_rates']) for s in SEEDS
            ),
        },
        'bootstrap_resamples': B,
        'bootstrap_seed': RNG_SEED,
    }
    path = SCALE_ROOT / 'scaleup_v2_summary.json'
    path.write_text(json.dumps(output, indent=2))

    print(f'Saved: {path}')
    pr = output['primary']
    print(
        f"H1SAT IPPO-P vs IPPO: {pr['improvement_pct']:+.4f}% "
        f"CI=[{pr['bootstrap_ci95_pct'][0]:+.4f},{pr['bootstrap_ci95_pct'][1]:+.4f}] "
        f"wins={pr['seed_wins']}/{pr['n_seeds']} "
        f"exact_p={pr['exact_two_sided_sign_flip_p']:.6f} "
        f"confirmed={pr['confirmed']}"
    )
    d = output['descriptive']
    print(
        f"IPPO U={d['ippo_u_mean_across_seeds']:.2f} (sd {d['ippo_u_seed_sd']:.2f}) "
        f"min-compl={d['ippo_completion_rate_overall_min']:.4f} | "
        f"IPPO-P U={d['ippo_p_u_mean_across_seeds']:.2f} "
        f"(sd {d['ippo_p_u_seed_sd']:.2f}) "
        f"min-compl={d['ippo_p_completion_rate_overall_min']:.4f}"
    )


if __name__ == '__main__':
    main()
