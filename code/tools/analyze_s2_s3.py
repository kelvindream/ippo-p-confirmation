#!/usr/bin/env python3
"""S2 (PressLight baseline) and S3 (scale probe) analyses per locked decision trees.

Locks: results/supplementary_v1/s2_presslight_lock.json, s3_scale_probe_lock.json.
Both are descriptive, post-confirmation supplementary analyses.
"""
import json
import math
from pathlib import Path

def _detect_results_root():
    """Support both the source layout (<repo>/results) and the deposit layout
    (<repo>/data/results)."""
    script_root = Path(__file__).resolve().parent.parent
    for candidate in (script_root / 'results',
                      script_root / 'data' / 'results',
                      script_root.parent / 'data' / 'results'):
        if candidate.is_dir():
            return candidate
    return script_root / 'results'


RESULTS_ROOT = _detect_results_root()
SUP = RESULTS_ROOT / 'supplementary_v1'
CONF = RESULTS_ROOT / 'ippo_prior_confirmation_v1'
SCENARIOS = ['H1', 'H2', 'H3']


def load_u(root, scenario, method, seed):
    values = json.loads((root / f'{scenario}_{method}_seed{seed}' /
                         'evaluation_metrics.json').read_text())['demand_service_times']
    assert len(values) == 5
    return sum(values) / len(values)


def archived_mean(scenario, method, seeds):
    values = [load_u(CONF, scenario, method, s) for s in seeds]
    return sum(values) / len(values)


def main():
    conf_seeds = list(range(47, 57))

    # ---- S3: scale probe (H1, scales 1.0/1.5/3.0, seeds 67/68) ----
    h1_ippo_mean = archived_mean('H1', 'IPPO', conf_seeds)
    h1_p_mean = archived_mean('H1', 'IPPO-P', conf_seeds)
    s3 = {'reference': {'H1_IPPO_mean_U': h1_ippo_mean, 'H1_IPPO-P_mean_U': h1_p_mean},
          'scales': {}}
    for scale in ['1.0', '1.5', '3.0']:
        us = {seed: load_u(SUP / 's3_scale_probe', 'H1', 'IPPO-P_s' + scale.replace('.', '.'), seed)
              for seed in (67, 68)}
        # task dirs use s1.0/s1.5/s3.0
        mean_u = sum(us.values()) / 2
        s3['scales'][scale] = {
            'per_seed_U': {str(k): v for k, v in us.items()},
            'mean_U': mean_u,
            'vs_IPPO_pct': (mean_u / h1_ippo_mean - 1) * 100,
            'vs_IPPO-P_pct': (mean_u / h1_p_mean - 1) * 100,
            'direction_vs_IPPO': 'better' if mean_u < h1_ippo_mean else 'worse',
        }
    stable = all(v['direction_vs_IPPO'] == 'better' for v in s3['scales'].values())
    s3['branch'] = ('all_directions_stable' if stable else
                    'direction_unstable' if any(v['direction_vs_IPPO'] == 'worse'
                                                for v in s3['scales'].values())
                    else 'mixed_or_weak')
    (SUP / 's3_scale_probe_analysis.json').write_text(json.dumps(s3, indent=2) + '\n')

    # ---- S2: PressLight baseline (descriptive, unpaired) ----
    s2 = {'per_scenario': {}}
    press_seeds = list(range(57, 67))
    for scenario in SCENARIOS:
        us = [load_u(SUP / 's2_presslight', scenario, 'PressLight', s) for s in press_seeds]
        mean_press = sum(us) / len(us)
        row = {'PressLight_mean_U': mean_press, 'PressLight_per_seed_U': us,
               'IPPO_mean_U': archived_mean(scenario, 'IPPO', conf_seeds),
               'IPPO-P_mean_U': archived_mean(scenario, 'IPPO-P', conf_seeds)}
        row['vs_IPPO_pct'] = (mean_press / row['IPPO_mean_U'] - 1) * 100
        row['vs_IPPO-P_pct'] = (mean_press / row['IPPO-P_mean_U'] - 1) * 100
        s2['per_scenario'][scenario] = row
    worse_everywhere = all(r['PressLight_mean_U'] > r['IPPO_mean_U'] for r in s2['per_scenario'].values())
    s2['branch'] = ('presslight_worse' if worse_everywhere else
                    'presslight_comparable_or_better' if any(
                        r['PressLight_mean_U'] <= r['IPPO-P_mean_U']
                        for r in s2['per_scenario'].values())
                    else 'mixed')
    (SUP / 's2_presslight_analysis.json').write_text(json.dumps(s2, indent=2) + '\n')

    print('S3 branch:', s3['branch'])
    for scale, v in s3['scales'].items():
        print(f"  scale {scale}: mean_U={v['mean_U']:.2f}  vs IPPO {v['vs_IPPO_pct']:+.3f}%"
              f"  vs IPPO-P {v['vs_IPPO-P_pct']:+.3f}%")
    print('S2 branch:', s2['branch'])
    for scenario, r in s2['per_scenario'].items():
        print(f"  {scenario}: PressLight={r['PressLight_mean_U']:.2f}  "
              f"IPPO={r['IPPO_mean_U']:.2f}  IPPO-P={r['IPPO-P_mean_U']:.2f}"
              f"  (PL vs IPPO {r['vs_IPPO_pct']:+.2f}%)")


if __name__ == '__main__':
    main()
