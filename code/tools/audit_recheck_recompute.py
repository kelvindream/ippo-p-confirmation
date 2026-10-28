"""AUD-01 forensic recheck: recompute pilot contrast and pooled 15-seed effect.

Recomputes, from raw per-task evaluation data:
  1. per-seed pilot log-ratios  (mean over 3 scenarios of log(U_IPPO/U_IPPO-P)),
  2. the pilot primary contrast (mean log-ratio, improvement pct, seed wins,
     exact two-sided sign test, n=5),
  3. the pooled 15-seed descriptive effect (10 confirmation + 5 pilot seeds),
plus the audit sensitivity reference excluding pilot seeds 45/46.

The corrupted stored H2 IPPO seed-46 U vector is replaced by the re-run
vector produced by tools/audit_recheck_rerun_eval.py.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO')
PILOT = ROOT / 'results' / 'formal_four_grid_v1'
CONF_SUMMARY = ROOT / 'results' / 'ippo_prior_confirmation_v1' / 'confirmation_summary.json'
SCENARIOS = ('H1', 'H2', 'H3')
PILOT_SEEDS = (42, 43, 44, 45, 46)


def task_u_mean(task_dir):
    data = json.load(open(task_dir / 'evaluation_metrics.json'))
    u = data['demand_service_times']
    return sum(u) / len(u), u


def mean_u_cell(method, scenario, seed, override=None):
    if override is not None:
        return override
    u, _ = task_u_mean(PILOT / f'{scenario}_{method}_seed{seed}')
    return u


def exact_two_sided_sign_p(wins, n):
    """Exact two-sided sign test: 2 * P(X <= min(wins, n-wins)) under Binomial(n, .5)."""
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun46', required=True,
                        help='path to audit recheck seed46 rerun evaluation_metrics.json')
    parser.add_argument('--rerun45', required=True,
                        help='path to audit recheck seed45 rerun evaluation_metrics.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rerun46 = json.load(open(args.rerun46))
    rerun45 = json.load(open(args.rerun45))
    recheck_u46 = rerun46['rerun_demand_service_times']
    recheck_u45 = rerun45['rerun_demand_service_times']

    # consistency gates
    stored45_u = json.load(open(PILOT / 'H2_IPPO_seed45' / 'evaluation_metrics.json'))['demand_service_times']
    stored46_u = json.load(open(PILOT / 'H2_IPPO_seed46' / 'evaluation_metrics.json'))['demand_service_times']
    d45 = max(abs(a - b) for a, b in zip(recheck_u45, stored45_u))
    d46_vs_stored46 = max(abs(a - b) for a, b in zip(recheck_u46, stored46_u))
    d46_vs_stored45 = max(abs(a - b) for a, b in zip(recheck_u46, stored45_u))

    agg = json.load(open(PILOT / 'aggregation_summary.json'))
    stored_seed_lr = agg['contrasts']['IPPO-P_vs_IPPO']['seed_log_ratios']
    conf = json.load(open(CONF_SUMMARY))
    conf_lr = conf['primary']['seed_log_ratios']
    pooled_stored = conf['secondary_pooled_15_seeds_descriptive_only']

    # per-seed pilot log-ratios; seed 46 H2 uses the re-run vector
    seed_lr, per_seed_detail = {}, {}
    for seed in PILOT_SEEDS:
        logs = []
        for scn in SCENARIOS:
            ippo = mean_u_cell('IPPO', scn, seed,
                               override=(sum(recheck_u46) / len(recheck_u46))
                               if (seed == 46 and scn == 'H2') else None)
            ippo_p = mean_u_cell('IPPO-P', scn, seed)
            logs.append(math.log(ippo / ippo_p))
        seed_lr[seed] = sum(logs) / len(logs)
        per_seed_detail[seed] = {
            'scenario_log_ratios': logs,
            'mean_log_ratio': seed_lr[seed],
            'H2_IPPO_u_mean': math.exp(logs[1]) * mean_u_cell('IPPO-P', 'H2', seed),
        }

    lr_list = [seed_lr[s] for s in PILOT_SEEDS]
    mean_lr = sum(lr_list) / len(lr_list)
    improvement_pct = 100.0 * (1.0 - math.exp(-mean_lr))
    wins = sum(1 for v in lr_list if v > 0)
    p_value = exact_two_sided_sign_p(wins, len(lr_list))

    # agreement of the 4 unaffected seeds vs stored values (42,43,44,45)
    unaffected = {s: {'recomputed': seed_lr[s], 'stored': stored_seed_lr[i],
                      'abs_diff': abs(seed_lr[s] - stored_seed_lr[i])}
                  for i, s in enumerate((42, 43, 44, 45))}

    # pooled 15 seeds (10 confirmation + 5 corrected pilot)
    pooled_lr = conf_lr + lr_list
    pooled_mean = sum(pooled_lr) / len(pooled_lr)
    pooled_pct = 100.0 * (1.0 - math.exp(-pooled_mean))
    pooled_wins = sum(1 for v in pooled_lr if v > 0)

    # sensitivity reference: drop pilot seeds 45 and 46 -> n = 13
    keep = conf_lr + [seed_lr[s] for s in (42, 43, 44)]
    sens_mean = sum(keep) / len(keep)
    sens_pct = 100.0 * (1.0 - math.exp(-sens_mean))
    sens_wins = sum(1 for v in keep if v > 0)

    result = {
        'verdict_inputs': {
            'rerun45_max_abs_diff_vs_stored45': d45,
            'rerun46_max_abs_diff_vs_stored46': d46_vs_stored46,
            'rerun46_max_abs_diff_vs_stored45': d46_vs_stored45,
        },
        'method': 'per-seed log-ratio = equal-weight mean over H1/H2/H3 of log(U_IPPO/U_IPPO-P); '
                  'contrast pct = 100*(1-exp(-mean log-ratio)); exact two-sided sign test',
        'pilot_seed_log_ratios_recomputed': {str(s): seed_lr[s] for s in PILOT_SEEDS},
        'pilot_seed_log_ratios_stored': {str(s): stored_seed_lr[i] for i, s in enumerate(PILOT_SEEDS)},
        'unaffected_seed_agreement': unaffected,
        'pilot_contrast': {
            'mean_log_ratio': mean_lr,
            'improvement_pct': improvement_pct,
            'seed_wins': wins,
            'n_seeds': 5,
            'p_value': p_value,
            'stored_reference': agg['contrasts']['IPPO-P_vs_IPPO'],
        },
        'pooled_15_seed': {
            'mean_log_ratio': pooled_mean,
            'improvement_pct': pooled_pct,
            'seed_wins': pooled_wins,
            'n_seeds': 15,
            'stored_reference': pooled_stored,
        },
        'sensitivity_drop_seeds_45_46_n13': {
            'mean_log_ratio': sens_mean,
            'improvement_pct': sens_pct,
            'seed_wins': sens_wins,
            'n_seeds': 13,
        },
        'per_seed_detail': per_seed_detail,
        'inputs': {
            'rerun46_u': recheck_u46,
            'rerun45_u': recheck_u45,
            'confirmation_seed_log_ratios': conf_lr,
        },
    }
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps({
        'pilot_contrast_pct': improvement_pct,
        'pilot_wins': wins,
        'pilot_p': p_value,
        'pooled_pct': pooled_pct,
        'sensitivity_n13_pct': sens_pct,
        'unaffected_max_abs_diff': max(v['abs_diff'] for v in unaffected.values()),
    }, indent=1))


if __name__ == '__main__':
    main()
