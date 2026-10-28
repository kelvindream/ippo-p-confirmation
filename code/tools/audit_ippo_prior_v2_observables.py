#!/usr/bin/env python3
"""Audit observable metrics for the IPPO-P scale-up scenarios.

This report deliberately does not infer per-intersection or phase behavior:
the existing evaluation artifacts do not record those granular metrics.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/zrt/ai/GC-MAPPO')
SCENARIOS = {
    'H1HIGH_2700': ROOT / 'results/ippo_prior_scaleup_v1',
    'H1SAT_5400': ROOT / 'results/ippo_prior_scaleup_v2',
}
SEEDS = {
    'H1HIGH_2700': range(62, 67),
    'H1SAT_5400': range(67, 72),
}


def load(root, prefix, method, seed):
    task = root / f'{prefix}_{method}_seed{seed}'
    return json.loads((task / 'evaluation_metrics.json').read_text())


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    report = {
        'scope': 'global evaluation observables only',
        'granular_metrics_available': False,
        'granular_metrics_missing': [
            'per_intersection input/output pressure',
            'per_direction pressure',
            'phase switch/hold frequencies for IPPO and IPPO-P',
            'per_intersection queue/waiting/speed time series',
        ],
        'scenarios': {},
    }
    for name, root in SCENARIOS.items():
        prefix = 'H1HIGH' if name.startswith('H1HIGH') else 'H1SAT'
        rows = {}
        for method in ('IPPO', 'IPPO-P'):
            task_rows = []
            for seed in SEEDS[name]:
                m = load(root, prefix, method, seed)
                keys = ('demand_service_times', 'travel_times', 'delays',
                        'queue_lengths', 'waiting_times', 'speeds',
                        'completion_rates', 'throughput_rates')
                row = {'seed': seed}
                for key in keys:
                    values = np.asarray(m[key], dtype=float)
                    row[key] = float(values.mean())
                    row[f'{key}_eval_sd'] = float(values.std(ddof=1))
                row['teleports'] = int(sum(m['teleport_counts']))
                row['conservation_ok'] = bool(all(m['demand_conservation_ok']))
                task_rows.append(row)
            rows[method] = task_rows
        pair = []
        for ippo, ippop in zip(rows['IPPO'], rows['IPPO-P']):
            item = {'seed': ippo['seed']}
            for key in ('demand_service_times', 'travel_times', 'delays',
                        'queue_lengths', 'waiting_times', 'speeds',
                        'completion_rates', 'throughput_rates'):
                a, b = ippo[key], ippop[key]
                item[f'{key}_delta_ippop_minus_ippo'] = b - a
                item[f'{key}_relative_ippop_minus_ippo_pct'] = (b - a) / a * 100.0 if a else None
            pair.append(item)
        report['scenarios'][name] = {'per_seed': rows, 'paired_deltas': pair}
    out = ROOT / 'results/ippo_prior_scaleup_v2' / 'global_observables_audit.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'Saved: {out}')
    for name, data in report['scenarios'].items():
        print(f'[{name}]')
        for method, rows in data['per_seed'].items():
            for key in ('demand_service_times', 'travel_times', 'delays', 'queue_lengths', 'waiting_times', 'speeds', 'completion_rates'):
                mean, sd = mean_sd([r[key] for r in rows])
                print(f'  {method:6s} {key:22s} mean={mean:.4f} seed_sd={sd:.4f}')
        for key in ('demand_service_times', 'travel_times', 'delays', 'queue_lengths', 'waiting_times', 'speeds', 'completion_rates'):
            values = [x[f'{key}_relative_ippop_minus_ippo_pct'] for x in data['paired_deltas']]
            mean, sd = mean_sd(values)
            print(f'  delta  {key:22s} IPPO-P minus IPPO pct mean={mean:+.4f} seed_sd={sd:.4f}')


if __name__ == '__main__':
    main()
