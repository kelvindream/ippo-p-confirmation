#!/usr/bin/env python3
"""S1 matched-arm analysis, strictly following the locked decision tree.

Lock: results/supplementary_v1/s1_matched_arm_lock.json
Pairs IPPO-State (supplementary_v1/s1_matched_arm) against the archived IPPO
results from the frozen confirmation (ippo_prior_confirmation_v1) on the same
seeds 47-56, same eval seeds 1042-1046, same metric U (demand service time).
"""
import json
import math
from pathlib import Path

def _detect_results_root():
    """Support both the source layout (<repo>/results) and the deposit layout
    (<repo>/data/results), mirroring analyze_eval_realization_sensitivity.py."""
    script_root = Path(__file__).resolve().parent.parent
    for candidate in (script_root / 'results',
                      script_root / 'data' / 'results',
                      script_root.parent / 'data' / 'results'):
        if candidate.is_dir():
            return candidate
    return script_root / 'results'


RESULTS_ROOT = _detect_results_root()
NEW = RESULTS_ROOT / 'supplementary_v1' / 's1_matched_arm'
OLD = RESULTS_ROOT / 'ippo_prior_confirmation_v1'
SCENARIOS = ['H1', 'H2', 'H3']
SEEDS = list(range(47, 57))


def load_u(root, scenario, method, seed):
    path = root / f'{scenario}_{method}_seed{seed}' / 'evaluation_metrics.json'
    values = json.loads(path.read_text())['demand_service_times']
    assert len(values) == 5, path
    return sum(values) / len(values)


def sign_test_p(pos, neg):
    n = pos + neg
    assert n > 0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1))
               / 2 ** n)


def main():
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            u_state = load_u(NEW, scenario, 'IPPO-State', seed)
            u_ippo = load_u(OLD, scenario, 'IPPO', seed)
            rows.append({
                'scenario': scenario, 'seed': seed,
                'U_IPPO_State': u_state, 'U_IPPO': u_ippo,
                'diff': u_state - u_ippo,
                'state_better': u_state < u_ippo,
            })
    summary = {}
    for scenario in SCENARIOS + ['ALL']:
        part = [r for r in rows if scenario == 'ALL' or r['scenario'] == scenario]
        pos = sum(1 for r in part if r['state_better'])
        neg = sum(1 for r in part if not r['state_better'])
        n = len(part)
        mean_old = sum(r['U_IPPO'] for r in part) / n
        mean_new = sum(r['U_IPPO_State'] for r in part) / n
        if scenario == 'ALL':
            log_ratio = [math.log(r['U_IPPO_State'] / r['U_IPPO']) for r in part]
            effect = (math.exp(sum(log_ratio) / n) - 1) * 100
        else:
            effect = (mean_new / mean_old - 1) * 100
        summary[scenario] = {
            'n': n, 'state_better': pos, 'ippo_better': neg,
            'p_sign_test': sign_test_p(pos, neg),
            'mean_U_IPPO': mean_old, 'mean_U_IPPO_State': mean_new,
            'pct_change': effect,
        }
    alls = summary['ALL']
    # Locked branch mapping (s1_matched_arm_lock.json):
    #   state_channel_advantage: IPPO-State beats IPPO in >= 8/10 with p<0.05
    #     -> the benefit is attributable to the extra halting-count state,
    #        not to the logit-prior integration.
    #   prior_advantage: IPPO-State does NOT beat IPPO (>= 8/10 the other way)
    #     -> the logit prior itself is the operative ingredient.
    if alls['state_better'] >= 8 and alls['p_sign_test'] < 0.05:
        branch = 'state_channel_advantage'
    elif alls['ippo_better'] >= 8:
        branch = 'prior_advantage'
    else:
        branch = 'indeterminate'
    out = {
        'lock': 'results/supplementary_v1/s1_matched_arm_lock.json',
        'branch': branch,
        'summary': summary,
        'rows': rows,
    }
    dest = RESULTS_ROOT / 'supplementary_v1' / 's1_matched_arm_analysis.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'branch': branch, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
