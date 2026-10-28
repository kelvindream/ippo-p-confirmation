"""Aggregate completed paper-protocol tasks without modifying their raw results."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


METHODS = (
    'GC-MAPPO', 'MAPPO', 'IPPO', 'CoLight', 'PressLight', 'Max-Pressure',
    'Fixed-Time', 'Double DQN',
    'Dueling Double DQN', 'GAE Actor-Critic',
    'REINFORCE',
)
# Primary cross-algorithm metrics defined by the experiment protocol.
METRICS = {
    'delay': ('delays', -1.0),
    'travel_time': ('travel_times', -1.0),
    'queue': ('queue_lengths', -1.0),
    'throughput': ('throughput', 1.0),
    'completion_rate': ('completion_rates', 1.0),
}
# Auxiliary metrics are exported separately and are not used for primary ranking.
AUXILIARY_METRICS = {
    'speed': ('speeds', 1.0),
    'waiting': ('waiting_times', -1.0),
    'evaluation_reward': ('evaluation_reward', 1.0),
}
# Phase 3 多场景支持：从 output-dir 路径中提取场景标签（如 phase3_h1_xxx → h1）；
# 无 --scenario-tag 时自动推断；推断失败用 'default' 保持原单场景行为。
_SCENARIO_RE = re.compile(r'(?:phase[23456]_|results[\\/])(h[12])(?:_|[\\/])')


def _scenario_from_path(state_path, override):
    if override:
        return override
    match = _SCENARIO_RE.search(state_path.replace('\\', '/'))
    return match.group(1) if match else 'default'


def _bootstrap_ci(values, rng, samples):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_bootstrap_pvalue(reference, baseline, direction, rng, samples):
    differences = direction * (np.asarray(reference, dtype=float) - np.asarray(baseline, dtype=float))
    draws = rng.choice(differences, size=(samples, differences.size), replace=True).mean(axis=1)
    return float(min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))))


def _holm(rows):
    # Holm 校正在传入的 rows 内部排序进行；多场景时按场景分组分别校正。
    ordered = sorted(enumerate(rows), key=lambda item: item[1]['p_value'])
    total = len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * row['p_value'])
        running = max(running, adjusted)
        rows[index]['holm_p_value'] = running


def main():
    parser = argparse.ArgumentParser(description='Summarize completed paper-protocol tasks.')
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=20000)
    # Phase 3 多场景：可选 --scenario-tag 显式指定单场景标签；
    # 不传则从每个 protocol_state.json 的 output-dir 路径推断（如 h1/h2）。
    parser.add_argument('--scenario-tag', default=None,
                        help='显式场景标签；不传则按路径推断 (h1/h2/default)')
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error('--bootstrap-samples must be at least 1000')

    records = []
    for state_path in Path(args.results_root).rglob('protocol_state.json'):
        # Only include Phase 3 results; skip diagnostic/smoke/calibration directories.
        if not any(part.startswith('phase3_') for part in state_path.parts):
            continue
        with state_path.open(encoding='utf-8') as source:
            state = json.load(source)
        if state.get('status') == 'completed':
            state['_path'] = str(state_path)
            state['_scenario'] = _scenario_from_path(str(state_path), args.scenario_tag)
            records.append(state)
    grouped = {}
    scenarios_found = set()
    for state in records:
        scenario = state['_scenario']
        scenarios_found.add(scenario)
        key = (scenario, state.get('method'), int(state.get('train_seed')))
        if key in grouped:
            raise RuntimeError(f'Duplicate completed task for {key}: {grouped[key]["_path"]} and {state["_path"]}')
        grouped[key] = state
    # 多场景完整性检查：每个场景都必须 10 算法 × 10 train seed 全部 completed
    missing = []
    for scenario in sorted(scenarios_found):
        for method in METHODS:
            for seed in range(42, 52):
                if (scenario, method, seed) not in grouped:
                    missing.append(f'{scenario}/{method}/seed_{seed}')
    if missing:
        raise RuntimeError(f'Incomplete protocol: {len(missing)} tasks missing; first: {", ".join(missing[:10])}')

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(20260922)
    summary_rows = []
    # per_seed 按 scenario -> method -> train_seed -> metric 组织，每个场景包含 10 个训练 seed 和 5 个核心指标
    per_seed = {scenario: {method: {} for method in METHODS} for scenario in scenarios_found}
    for scenario in sorted(scenarios_found):
        for method in METHODS:
            for metric, (field, _) in METRICS.items():
                values = []
                for seed in range(42, 52):
                    episode_values = grouped[(scenario, method, seed)]['evaluation_metrics'].get(field)
                    if not episode_values or any(value is None for value in episode_values):
                        raise RuntimeError(
                            f'Missing primary metric {metric} for {scenario}/{method}/seed_{seed}'
                        )
                    value = float(np.mean(episode_values))
                    values.append(value)
                    per_seed[scenario][method].setdefault(seed, {})[metric] = value
                low, high = _bootstrap_ci(values, rng, args.bootstrap_samples)
                summary_rows.append({
                    'scenario': scenario,
                    'method': method,
                    'metric': metric,
                    'n_train_seeds': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)),
                    'ci95_low': low,
                    'ci95_high': high,
                })

    significance_rows = []
    # 按 scenario 分组做 Holm 校正，不合并跨场景的检验。
    for scenario in sorted(scenarios_found):
        scenario_rows = []
        for method in METHODS:
            if method == 'GC-MAPPO':
                continue
            for metric, (_, direction) in METRICS.items():
                reference = [per_seed[scenario]['GC-MAPPO'][seed][metric] for seed in range(42, 52)]
                baseline = [per_seed[scenario][method][seed][metric] for seed in range(42, 52)]
                row = {
                    'scenario': scenario,
                    'reference': 'GC-MAPPO',
                    'baseline': method,
                    'metric': metric,
                    'mean_difference_oriented': float(direction * (np.mean(reference) - np.mean(baseline))),
                    'p_value': _paired_bootstrap_pvalue(reference, baseline, direction, rng, args.bootstrap_samples),
                }
                significance_rows.append(row)
                scenario_rows.append(row)
        _holm(scenario_rows)  # 场景内 10 基线 × 5 核心指标 = 50 检验 Holm 校正

    with open(os.path.join(args.output_dir, 'per_seed_evaluation.csv'), 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=('scenario', 'method', 'train_seed', *METRICS))
        writer.writeheader()
        for scenario in sorted(scenarios_found):
            for method in METHODS:
                for seed in range(42, 52):
                    writer.writerow({'scenario': scenario, 'method': method, 'train_seed': seed, **per_seed[scenario][method][seed]})
    with open(os.path.join(args.output_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    auxiliary_rows = []
    for scenario in sorted(scenarios_found):
        for method in METHODS:
            for metric, (field, _) in AUXILIARY_METRICS.items():
                values = []
                for seed in range(42, 52):
                    episode_values = grouped[(scenario, method, seed)]['evaluation_metrics'].get(field)
                    if not episode_values:
                        raise RuntimeError(
                            f'Missing auxiliary metric {metric} for {scenario}/{method}/seed_{seed}'
                        )
                    numeric_values = [value for value in episode_values if value is not None]
                    if not numeric_values:
                        raise RuntimeError(
                            f'Unavailable auxiliary metric {metric} for {scenario}/{method}/seed_{seed}'
                        )
                    values.append(float(np.mean(numeric_values)))
                low, high = _bootstrap_ci(values, rng, args.bootstrap_samples)
                auxiliary_rows.append({
                    'scenario': scenario,
                    'method': method,
                    'metric': metric,
                    'n_train_seeds': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)),
                    'ci95_low': low,
                    'ci95_high': high,
                })
    with open(os.path.join(args.output_dir, 'auxiliary_summary.csv'), 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=auxiliary_rows[0].keys())
        writer.writeheader()
        writer.writerows(auxiliary_rows)

    with open(os.path.join(args.output_dir, 'paired_bootstrap_holm.csv'), 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=significance_rows[0].keys())
        writer.writeheader()
        writer.writerows(significance_rows)
    with open(os.path.join(args.output_dir, 'paper_statistics.md'), 'w', encoding='utf-8') as output:
        output.write('# Fixed-Protocol Evaluation Statistics\n\n')
        output.write('Each row uses the mean of 10 deterministic evaluation environments per independently trained seed. '\
                     'Confidence intervals use 20,000 bootstrap resamples over 10 training seeds. '\
                     'GC-MAPPO comparisons are paired by training seed and Holm-adjusted within each scenario (50 tests per scenario: 10 baselines x 5 primary metrics).\n\n')
        for scenario in sorted(scenarios_found):
            output.write(f'## Scenario: {scenario}\n\n')
            output.write('| Method | Metric | Mean | SD | 95% CI |\n| --- | --- | ---: | ---: | --- |\n')
            for row in summary_rows:
                if row['scenario'] == scenario:
                    output.write(f"| {row['method']} | {row['metric']} | {row['mean']:.6f} | {row['std']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |\n")
            output.write('\n### Holm-adjusted paired bootstrap (GC-MAPPO vs baseline)\n\n')
            output.write('| Baseline | Metric | Mean diff (oriented) | p-value | Holm p |\n| --- | --- | ---: | ---: | ---: |\n')
            for row in significance_rows:
                if row['scenario'] == scenario:
                    output.write(f"| {row['baseline']} | {row['metric']} | {row['mean_difference_oriented']:.6f} | {row['p_value']:.6f} | {row.get('holm_p_value', float('nan')):.6f} |\n")
            output.write('\n')
    print(f'Statistics written to {args.output_dir} (scenarios: {sorted(scenarios_found)})')


if __name__ == '__main__':
    main()
