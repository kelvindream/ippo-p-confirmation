import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import time
import argparse
import json
import os
import csv
import traceback
import random
from datetime import datetime
import sys
import re
import pickle

# 导入环境和智能体
from environment.env import SumoEnv
from environment.metrics import collect_network_metrics
from replay import ReplayBuffer

# 基于值的算法
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_double_dqn import DuelingDoubleDQNAgent
from agents._colight_core import GraphAttentionDoubleDQN
from agents._presslight_core import PressureStateDoubleDQN
from agents.colight import CoLight
from agents.presslight import PressLight
from agents.fixed_time import FixedTimeController

# 基于策略的算法
from agents.reinforce import ReinforceAgent

# 基于Actor-Critic的算法
from agents.gae_actor_critic import GAEActorCritic
from agents.gc_mappo import GCMAPPO
from agents.dpg_mappo import DPGMAPPO, directional_phase_features
from agents.mappo import MAPPO
from agents.ippo import IPPO, ippo_parameter_count, matched_hidden_dim
from agents.gated_graph_residual_ippo import GatedGraphResidualIPPO
from agents.max_pressure import MaxPressureController
# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")
if device == "cuda":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU name: {torch.cuda.get_device_name(0)}")

# 环境参数配置
ENV_PARAMS = {
    'net_file': 'nets/2x2-grid/2x2.net.xml',
    'route_file': 'nets/2x2-grid/2x2-medium.rou.xml',
    'skip_range': 0,
    'simulation_time': 10000,
    'yellow_time': 2,
    'delta_rs_update_time': 10,
    'use_gui': False,
    'multi_agent': False,
    'tls_min_links': 4,
    'sumo_seed': 42,
}

METRIC_SAMPLE_INTERVAL = 10
GC_MAPPO_PRESSURE_CONFIDENCE_THRESHOLD = 4.0

SHARED_POLICY_GAE_NAME = 'GAE Actor-Critic'
SHARED_POLICY_REINFORCE_NAME = 'REINFORCE'
GRAPH_ATTENTION_DDQN_NAME = 'Graph-Attention Double DQN'
PRESSURE_STATE_DDQN_NAME = 'Pressure-State Double DQN'
COLIGHT_NAME = 'CoLight'
PRESSLIGHT_NAME = 'PressLight'
FIXED_TIME_NAME = 'Fixed-Time'
PARAMETER_SHARED_DDQN_NAME = 'Double DQN'
PARAMETER_SHARED_DUELING_DDQN_NAME = 'Dueling Double DQN'


def _seed_everything(seed):
    """Reset all local RNGs so each algorithm has an order-independent seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _save_episode_checkpoint(path, agent, episode, metrics, replay_buffer=None):
    checkpoint = {
        'episode': int(episode),
        'metrics': metrics,
        'rng_state': _capture_rng_state(),
        'agent': agent,
    }
    if replay_buffer is not None:
        checkpoint['replay_buffer'] = replay_buffer
    temp_path = f'{path}.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def _load_episode_checkpoint(path):
    if not path or not os.path.isfile(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def _has_active_vehicles(env):
    """判断当前采样时刻是否仍有活跃车辆，避免把空采样记成全 0 指标。"""
    signals = getattr(env, 'traffic_signals', None) or [env.traffic_signal]
    seen_lanes = set()
    total_vehicles = 0
    for ts in signals:
        for lane_id in getattr(ts, 'lanes_id', []):
            if lane_id in seen_lanes:
                continue
            seen_lanes.add(lane_id)
            total_vehicles += env.sumo.lane.getLastStepVehicleNumber(lane_id)
    return total_vehicles > 0


def _collect_network_metric_components(env):
    """Use the shared network metric implementation for every algorithm."""
    signals = getattr(env, 'traffic_signals', None) or [env.traffic_signal]
    metrics = collect_network_metrics(env.sumo, signals)
    return {
        'signal_count': len(signals),
        'lane_count': metrics['lane_count'],
        'vehicle_count': metrics['vehicle_count'],
        'speed_vehicle_count': metrics['speed_vehicle_count'],
        'total_speed': metrics['total_speed'],
        'total_queue_length': metrics['total_queue'],
        'total_waiting_time': metrics['total_waiting'],
    }


def _collect_env_metrics(env):
    """统一单/多智能体场景下的指标聚合口径。"""

    components = _collect_network_metric_components(env)
    if components['vehicle_count'] <= 0:
        return (None, None, None)

    result = (
        (
            float(components['total_speed'] / components['speed_vehicle_count'])
            if components['speed_vehicle_count'] > 0 else None
        ),
        (
            float(components['total_queue_length'] / components['lane_count'])
            if components['lane_count'] > 0 else None
        ),
        (
            float(components['total_waiting_time'] / components['vehicle_count'])
            if components['vehicle_count'] > 0 else None
        ),
    )
    return result


def _maybe_record_metrics(env, episode_speeds, episode_queue_lengths, episode_waiting_times, last_sample_time, done):
    """按统一频率采样，避免不同算法统计口径不一致。"""
    current_time = int(env.sumo.simulation.getTime())
    if last_sample_time is None:
        should_sample = True
    else:
        should_sample = bool(done) or (current_time - int(last_sample_time) >= int(METRIC_SAMPLE_INTERVAL))
    if not should_sample or current_time == last_sample_time:
        return last_sample_time

    speed, queue_length, waiting_time = _collect_env_metrics(env)
    if speed is not None and queue_length is not None and waiting_time is not None:
        episode_speeds.append(speed)
        episode_queue_lengths.append(queue_length)
        episode_waiting_times.append(waiting_time)
        return current_time

    return last_sample_time


def _episode_metric_means(episode_speeds, episode_queue_lengths, episode_waiting_times):
    """统一汇总单个 episode 的指标，避免空采样直接落成 0。"""
    return (
        _safe_mean(episode_speeds),
        _safe_mean(episode_queue_lengths),
        _safe_mean(episode_waiting_times),
    )


def _is_effective_transition(do_action, next_state, done):
    return bool(do_action is not None and do_action >= 0 and (next_state is not None or done))


def _append_effective_single_reward(episode_rewards, reward, do_action, next_state, done):
    if reward is None:
        return False
    if not _is_effective_transition(do_action, next_state, done):
        return False
    episode_rewards.append(float(reward))
    return True


def _append_effective_multi_rewards(episode_rewards, rewards, do_actions, next_states, done):
    numeric_rewards = []
    for reward in (rewards or []):
        if reward is None:
            continue
        try:
            numeric_reward = float(reward)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_reward):
            continue
        numeric_rewards.append(numeric_reward)
    if numeric_rewards:
        # 对多智能体指标统计来说，reward 应覆盖整个仿真过程，而不是只看有效 transition。
        # 否则 eval 中大量 do_action=None/-1 的拥堵时段会被跳过，导致 episode reward 虚高。
        episode_rewards.append(float(np.mean(numeric_rewards)))
        return True
    return False


def _safe_mean(values):
    """跳过 nan 计算均值；空结果保持 nan，避免错误回填成 0。"""
    if not values:
        return float('nan')
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return float('nan')
    return float(np.mean(valid))


def _accumulate_reward_components(accumulator, components):
    """按路口汇总环境 reward 分解项，供 episode 级诊断使用。"""
    for component in components or []:
        if not isinstance(component, dict):
            continue
        for name, value in component.items():
            if name in {'best_action', 'do_action'}:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            accumulator.setdefault(name, []).append(numeric_value)


def _finalize_reward_components(accumulator):
    return {name: _safe_mean(values) for name, values in accumulator.items()}


def _accumulate_global_reward(accumulator, info):
    value = info.get('global_reward') if isinstance(info, dict) else None
    if value is not None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric_value):
            accumulator.setdefault('global_reward', []).append(numeric_value)
    metrics = info.get('global_metrics', {}) if isinstance(info, dict) else {}
    for name in ('speed', 'queue', 'waiting'):
        try:
            numeric_value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            accumulator.setdefault(f'global_{name}', []).append(numeric_value)


def _finalize_global_reward(accumulator):
    return {name: _safe_mean(values) for name, values in accumulator.items()}


def _safe_corr(x_values, y_values):
    """计算相关系数，自动跳过 nan，并处理常量序列。"""
    if not x_values or not y_values:
        return float('nan')
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]
    if x_array.size < 2 or y_array.size < 2:
        return float('nan')
    if np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return float('nan')
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _compute_reward_acceptance(metrics):
    """
    计算 reward 趋势与业务指标一致性。
    验收逻辑遵循：
    1. reward 应以平滑上升为目标，因此用斜率和前后 20 集均值比较；
    2. reward 应满足原始业务指标的符号约束：
       与 speed 正相关，与 queue_length / waiting_time 负相关。
    """
    rewards = [float(v) for v in metrics.get('rewards', []) if v is not None and not math.isnan(float(v))]
    if not rewards:
        return {
            'reward_trend_slope': float('nan'),
            'reward_first20_mean': float('nan'),
            'reward_last20_mean': float('nan'),
            'reward_last20_delta': float('nan'),
            'corr_reward_speed': float('nan'),
            'corr_reward_queue_length': float('nan'),
            'corr_reward_waiting_time': float('nan'),
            'reward_trend_pass': False,
            'reward_metric_alignment_pass': False,
        }

    reward_series = np.asarray(metrics.get('rewards', []), dtype=float)
    x_axis = np.arange(len(reward_series), dtype=float)
    valid_mask = ~np.isnan(reward_series)
    valid_x = x_axis[valid_mask]
    valid_rewards = reward_series[valid_mask]
    if valid_rewards.size >= 2:
        reward_slope = float(np.polyfit(valid_x, valid_rewards, 1)[0])
    else:
        reward_slope = float('nan')

    first_window = rewards[: min(20, len(rewards))]
    last_window = rewards[-min(20, len(rewards)):]
    first20_mean = _safe_mean(first_window)
    last20_mean = _safe_mean(last_window)
    reward_last20_delta = (
        float(last20_mean - first20_mean)
        if not math.isnan(first20_mean) and not math.isnan(last20_mean)
        else float('nan')
    )

    speed_series = metrics.get('speeds', [])
    queue_length_series = metrics.get('queue_lengths', [])
    waiting_time_series = metrics.get('waiting_times', [])

    corr_reward_speed = _safe_corr(metrics.get('rewards', []), speed_series)
    corr_reward_queue_length = _safe_corr(metrics.get('rewards', []), queue_length_series)
    corr_reward_waiting_time = _safe_corr(metrics.get('rewards', []), waiting_time_series)

    trend_pass = (
        not math.isnan(reward_slope)
        and not math.isnan(reward_last20_delta)
        and reward_slope > 0.0
        and reward_last20_delta > 0.0
    )
    alignment_pass = (
        (not math.isnan(corr_reward_speed) and corr_reward_speed > 0.05)
        and (not math.isnan(corr_reward_queue_length) and corr_reward_queue_length < -0.05)
        and (not math.isnan(corr_reward_waiting_time) and corr_reward_waiting_time < -0.05)
    )

    return {
        'reward_trend_slope': reward_slope,
        'reward_first20_mean': first20_mean,
        'reward_last20_mean': last20_mean,
        'reward_last20_delta': reward_last20_delta,
        'corr_reward_speed': corr_reward_speed,
        'corr_reward_queue_length': corr_reward_queue_length,
        'corr_reward_waiting_time': corr_reward_waiting_time,
        'reward_trend_pass': bool(trend_pass),
        'reward_metric_alignment_pass': bool(alignment_pass),
    }


def _safe_metric_delta(current_value, previous_value):
    if current_value is None or previous_value is None:
        return float('nan')
    try:
        current_numeric = float(current_value)
        previous_numeric = float(previous_value)
    except (TypeError, ValueError):
        return float('nan')
    if math.isnan(current_numeric) or math.isnan(previous_numeric):
        return float('nan')
    return float(current_numeric - previous_numeric)


def _build_metric_snapshot(metrics):
    return {
        'avg_reward': _safe_mean(metrics.get('rewards', [])),
        'avg_speed': _safe_mean(metrics.get('speeds', [])),
        'avg_queue_length': _safe_mean(metrics.get('queue_lengths', [])),
        'avg_waiting_time': _safe_mean(metrics.get('waiting_times', [])),
    }


def _snapshot_business_gate_pass(snapshot):
    avg_speed = snapshot.get('avg_speed')
    avg_queue = snapshot.get('avg_queue_length')
    avg_waiting = snapshot.get('avg_waiting_time')
    if any(math.isnan(float(v)) for v in (avg_speed, avg_queue, avg_waiting)):
        return False
    return bool(
        float(avg_speed) >= 1.2
        and float(avg_queue) <= 100.0
        and float(avg_waiting) <= 150.0
    )


def _build_lessons_learned(analysis, snapshot):
    lessons = []
    if not analysis.get('reward_trend_pass'):
        lessons.append('reward 趋势未通过，仍需提升斜率和后段均值。')
    if not analysis.get('reward_metric_alignment_pass'):
        lessons.append('reward 与速度/队列/等待时间的业务对齐尚未全部通过。')
    if not _snapshot_business_gate_pass(snapshot):
        lessons.append('评估业务门槛未通过，当前模型仍存在 reward 与真实交通效果脱钩风险。')
    corr_speed = analysis.get('corr_reward_speed')
    if not math.isnan(corr_speed) and corr_speed <= 0.05:
        lessons.append('reward-speed 正相关不足，需要继续提高高 reward 对高速度的偏好。')
    corr_queue = analysis.get('corr_reward_queue_length')
    if not math.isnan(corr_queue) and corr_queue >= -0.05:
        lessons.append('reward-queue 负相关不足，需要继续压制高排队状态。')
    corr_waiting = analysis.get('corr_reward_waiting_time')
    if not math.isnan(corr_waiting) and corr_waiting >= -0.05:
        lessons.append('reward-waiting 负相关不足，需要继续压制高等待状态。')
    avg_waiting = snapshot.get('avg_waiting_time')
    if not math.isnan(avg_waiting) and avg_waiting > 30.0:
        lessons.append('平均等待时间偏高，说明评估场景下的策略泛化仍不稳定。')
    avg_queue = snapshot.get('avg_queue_length')
    if not math.isnan(avg_queue) and avg_queue > 50.0:
        lessons.append('平均排队长度偏高，需要进一步抑制拥堵累积。')
    return lessons


def _compare_with_previous_baseline(current_analysis, current_snapshot, previous_analysis, previous_snapshot):
    current_pass_count = int(bool(current_analysis.get('reward_trend_pass'))) + int(bool(current_analysis.get('reward_metric_alignment_pass')))
    previous_pass_count = int(bool(previous_analysis.get('reward_trend_pass'))) + int(bool(previous_analysis.get('reward_metric_alignment_pass')))
    if current_pass_count > previous_pass_count:
        return 'better_than_previous'
    if current_pass_count < previous_pass_count:
        return 'weaker_than_previous'

    better_axes = 0
    weaker_axes = 0

    speed_delta = _safe_metric_delta(current_snapshot.get('avg_speed'), previous_snapshot.get('avg_speed'))
    if not math.isnan(speed_delta):
        if speed_delta > 0.05:
            better_axes += 1
        elif speed_delta < -0.05:
            weaker_axes += 1

    queue_delta = _safe_metric_delta(current_snapshot.get('avg_queue_length'), previous_snapshot.get('avg_queue_length'))
    if not math.isnan(queue_delta):
        if queue_delta < -0.50:
            better_axes += 1
        elif queue_delta > 0.50:
            weaker_axes += 1

    waiting_delta = _safe_metric_delta(current_snapshot.get('avg_waiting_time'), previous_snapshot.get('avg_waiting_time'))
    if not math.isnan(waiting_delta):
        if waiting_delta < -0.50:
            better_axes += 1
        elif waiting_delta > 0.50:
            weaker_axes += 1

    reward_delta = _safe_metric_delta(current_snapshot.get('avg_reward'), previous_snapshot.get('avg_reward'))
    if not math.isnan(reward_delta):
        if reward_delta > 0.001:
            better_axes += 1
        elif reward_delta < -0.001:
            weaker_axes += 1

    corr_speed_delta = _safe_metric_delta(
        current_analysis.get('corr_reward_speed'),
        previous_analysis.get('corr_reward_speed'),
    )
    if not math.isnan(corr_speed_delta):
        if corr_speed_delta > 0.01:
            better_axes += 1
        elif corr_speed_delta < -0.01:
            weaker_axes += 1

    if better_axes >= weaker_axes + 2:
        return 'better_than_previous'
    if weaker_axes >= better_axes + 1:
        return 'weaker_than_previous'
    return 'mixed_vs_previous'


def _compare_primary_metrics(current_snapshot, previous_snapshot):
    """Apply the strict four-metric Pareto rule used for candidate retention."""
    metric_rules = {
        'avg_reward': 1.0,
        'avg_speed': 1.0,
        'avg_queue_length': -1.0,
        'avg_waiting_time': -1.0,
    }
    deltas = {
        metric_name: _safe_metric_delta(
            current_snapshot.get(metric_name), previous_snapshot.get(metric_name)
        )
        for metric_name in metric_rules
    }
    comparable = all(not math.isnan(delta) for delta in deltas.values())
    non_worse = comparable and all(
        delta * metric_rules[metric_name] >= 0.0
        for metric_name, delta in deltas.items()
    )
    strict_improvements = sum(
        delta * metric_rules[metric_name] > 0.0
        for metric_name, delta in deltas.items()
    ) if comparable else 0
    return {
        'deltas': deltas,
        'non_worse': bool(non_worse),
        'strict_improvement_count': int(strict_improvements),
        'strict_pareto_pass': bool(non_worse and strict_improvements >= 3),
    }


def _load_explicit_baseline_review(algorithm_name, baseline_dir, net_file):
    """Load the declared accepted baseline instead of inferring one from timestamps."""
    if not baseline_dir:
        return None
    baseline_dir = os.path.abspath(baseline_dir)
    state_paths = sorted(
        os.path.join(baseline_dir, filename)
        for filename in os.listdir(baseline_dir)
        if filename.startswith('run_state_') and filename.endswith('.json')
    ) if os.path.isdir(baseline_dir) else []
    protocol_state_path = os.path.join(baseline_dir, 'protocol_state.json')
    if not state_paths and os.path.isfile(protocol_state_path):
        with open(protocol_state_path, 'r', encoding='utf-8') as file:
            protocol_state = json.load(file)
        if protocol_state.get('status') != 'completed':
            raise ValueError(
                f'acceptance protocol baseline is not completed: {baseline_dir}'
            )
        environment = protocol_state.get('environment') or {}
        if environment.get('net_file') != net_file:
            raise ValueError('acceptance protocol baseline network does not match the current run')
        if int(protocol_state.get('train_episodes', -1)) != 50:
            raise ValueError('acceptance protocol baseline must contain 50 training episodes')
        metrics = protocol_state.get('training_metrics')
        required_series = ('rewards', 'speeds', 'queue_lengths', 'waiting_times')
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(key), list) or len(metrics[key]) != 50
            for key in required_series
        ):
            raise ValueError('acceptance protocol baseline has incomplete training metrics')
        return {
            'run_id': os.path.basename(baseline_dir),
            'finished_at': protocol_state.get('finished_at', ''),
            'analysis': _compute_reward_acceptance(metrics),
            'snapshot': _build_metric_snapshot(metrics),
            'source': 'explicit_completed_protocol_baseline',
        }
    # A protocol baseline may contain a legacy auxiliary state snapshot.  Use
    # the newest completed state while still rejecting ambiguous live states.
    if len(state_paths) > 1:
        completed_state_paths = []
        for state_path in state_paths:
            try:
                with open(state_path, 'r', encoding='utf-8') as file:
                    state_probe = json.load(file)
            except (OSError, ValueError):
                continue
            if state_probe.get('status') in (
                'completed', 'completed_with_failures', 'completed_with_reused_baselines'
            ):
                completed_state_paths.append(state_path)
        if len(completed_state_paths) == 1:
            state_paths = completed_state_paths
    if len(state_paths) != 1:
        raise ValueError(
            f'acceptance baseline must contain exactly one run_state_*.json: {baseline_dir}'
        )
    with open(state_paths[0], 'r', encoding='utf-8') as file:
        state = json.load(file)
    if (state.get('config') or {}).get('net_file') != net_file:
        raise ValueError('acceptance baseline network does not match the current run')
    metrics = (state.get('all_metrics') or {}).get(algorithm_name)
    if not isinstance(metrics, dict):
        raise ValueError(f'acceptance baseline does not contain metrics for {algorithm_name}')
    return {
        'run_id': state.get('run_id'),
        'finished_at': state.get('finished_at') or state.get('last_completed_at') or '',
        'analysis': _compute_reward_acceptance(metrics),
        'snapshot': _build_metric_snapshot(metrics),
        'source': 'explicit_accepted_baseline',
    }


def _find_previous_baseline_review(algorithm_name, current_run_id, net_file, results_root='./results'):
    best_candidate = None
    if not os.path.isdir(results_root):
        return None

    for root, _, files in os.walk(results_root):
        for filename in files:
            if not (filename.startswith('run_state_') and filename.endswith('.json')):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception:
                continue

            if not isinstance(state, dict):
                continue
            if state.get('run_id') == current_run_id:
                continue
            if state.get('status') not in ('completed', 'completed_with_failures', 'completed_with_reused_baselines'):
                continue
            config = state.get('config') or {}
            if config.get('net_file') != net_file:
                continue

            metrics = (state.get('all_metrics') or {}).get(algorithm_name)
            if not isinstance(metrics, dict):
                continue

            finished_at = state.get('finished_at') or state.get('last_completed_at') or state.get('started_at') or ''
            candidate = {
                'run_id': state.get('run_id'),
                'finished_at': finished_at,
                'analysis': _compute_reward_acceptance(metrics),
                'snapshot': _build_metric_snapshot(metrics),
            }
            if best_candidate is None or str(candidate['finished_at']) > str(best_candidate['finished_at']):
                best_candidate = candidate
    return best_candidate


def _build_algorithm_review(algorithm_name, metrics, run_info, output_dir='./results'):
    analysis = _compute_reward_acceptance(metrics)
    snapshot = _build_metric_snapshot(metrics)
    pass_count = int(bool(analysis.get('reward_trend_pass'))) + int(bool(analysis.get('reward_metric_alignment_pass')))
    business_gate_pass = _snapshot_business_gate_pass(snapshot)
    net_file = (run_info.get('config') or {}).get('net_file')
    previous = _load_explicit_baseline_review(
        algorithm_name, run_info.get('acceptance_baseline_dir'), net_file
    )
    if previous is None:
        previous = _find_previous_baseline_review(
            algorithm_name, run_info.get('run_id'), net_file, results_root=os.path.abspath('./results')
        )
    comparison = 'no_previous_baseline'
    primary_comparison = None
    if previous is not None:
        comparison = _compare_with_previous_baseline(
            analysis,
            snapshot,
            previous['analysis'],
            previous['snapshot'],
        )
        primary_comparison = _compare_primary_metrics(snapshot, previous['snapshot'])

    strict_pareto_pass = previous is None or primary_comparison['strict_pareto_pass']
    if not business_gate_pass or pass_count < 2:
        acceptance_status = 'failed'
    elif strict_pareto_pass:
        acceptance_status = 'accepted'
    else:
        acceptance_status = 'candidate'

    if not business_gate_pass or pass_count < 2:
        decision = 'rollback_to_previous_stronger_version' if previous is not None else 'continue_optimization_from_current_baseline'
    elif strict_pareto_pass:
        decision = 'keep'
    else:
        decision = 'rollback_to_previous_stronger_version'

    review = {
        'acceptance_status': acceptance_status,
        'comparison_to_previous': comparison,
        'keep_or_rollback_decision': decision,
        'analysis': analysis,
        'metric_snapshot': snapshot,
        'lessons_learned': _build_lessons_learned(analysis, snapshot),
    }
    if previous is not None:
        review['previous_baseline'] = {
            'run_id': previous.get('run_id'),
            'finished_at': previous.get('finished_at'),
            'analysis': previous.get('analysis'),
            'metric_snapshot': previous.get('snapshot'),
        }
        review['primary_metric_comparison'] = primary_comparison
    return review


def _format_algorithm_review_for_log(algorithm_name, review):
    analysis = review.get('analysis') or {}
    snapshot = review.get('metric_snapshot') or {}
    return (
        f"算法评审: {algorithm_name} "
        f"acceptance={review.get('acceptance_status')} "
        f"comparison={review.get('comparison_to_previous')} "
        f"decision={review.get('keep_or_rollback_decision')} "
        f"trend_pass={analysis.get('reward_trend_pass')} "
        f"alignment_pass={analysis.get('reward_metric_alignment_pass')} "
        f"avg_reward={_serialize_metric_value(snapshot.get('avg_reward'))} "
        f"avg_speed={_serialize_metric_value(snapshot.get('avg_speed'))} "
        f"avg_queue={_serialize_metric_value(snapshot.get('avg_queue_length'))} "
        f"avg_waiting={_serialize_metric_value(snapshot.get('avg_waiting_time'))}"
    )


def _print_algorithm_reviews(run_info):
    reviews = (run_info or {}).get('algorithm_reviews') or {}
    if not reviews:
        return
    print("\n" + "=" * 80)
    print("奖励验收与保留/回退建议")
    print("=" * 80)
    for algorithm_name, review in reviews.items():
        print(
            f"{algorithm_name}: acceptance={review.get('acceptance_status')}, "
            f"comparison={review.get('comparison_to_previous')}, "
            f"decision={review.get('keep_or_rollback_decision')}"
        )

def _infer_action_dim(env):
    if getattr(env, 'multi_agent', False):
        try:
            spaces = env.get_agent_spaces()
            if spaces:
                return int(max(space[1].n for space in spaces))
        except Exception:
            pass
        try:
            signals = getattr(env, 'traffic_signals', None)
            if signals:
                return int(max(len(ts.all_green_phases) for ts in signals))
        except Exception:
            pass
    return int(env.action_space.n)


def _enforce_execution_standard(net_file, route_file, effective_tls_min_links, sumo_seed):
    python_ok = (sys.version_info.major == 3 and sys.version_info.minor == 9)
    cuda_ok = bool(torch.cuda.is_available()) and device == 'cuda'
    is_standard_2x2 = '2x2-grid' in str(net_file)
    scenario_ok = True
    expected_agents = None
    if is_standard_2x2:
        expected_agents = 4
        scenario_ok = (
            int(ENV_PARAMS.get('simulation_time', 0)) == 10000
            and int(effective_tls_min_links) >= 4
        )

    standard_info = {
        'profile_name': 'paper_2x2_medium_py39_gpu_core4_10000s',
        'python_required': '3.9',
        'python_actual': sys.version.split()[0],
        'python_ok': bool(python_ok),
        'device_required': 'cuda',
        'device_actual': device,
        'cuda_ok': bool(cuda_ok),
        'net_file': net_file,
        'route_file': route_file,
        'simulation_time': int(ENV_PARAMS.get('simulation_time', 0)),
        'tls_min_links': int(effective_tls_min_links),
        'sumo_seed': int(sumo_seed),
        'is_standard_2x2': bool(is_standard_2x2),
        'expected_core_agents': expected_agents,
        'scenario_ok': bool(scenario_ok),
        'standard_ok': bool(python_ok and cuda_ok and scenario_ok),
    }

    if not python_ok or not cuda_ok:
        raise RuntimeError(
            "标准环境要求 Python 3.9 + GPU(CUDA)。请使用 `py -3.9` 启动，并确保当前 Torch 可用 CUDA。"
        )
    return standard_info


def _normalize_state_dim(state, target_dim):
    """将状态向量对齐到固定维度，不足补 0，超出截断。"""
    array = np.asarray(state, dtype=np.float32).reshape(-1)
    if array.size == int(target_dim):
        return array
    normalized = np.zeros(int(target_dim), dtype=np.float32)
    copy_size = min(array.size, int(target_dim))
    if copy_size > 0:
        normalized[:copy_size] = array[:copy_size]
    return normalized


def _pressure_confidence_action_masks(raw_pressure, decision_slots, threshold=None):
    pressure = np.asarray(raw_pressure, dtype=np.float32)
    masks = np.ones_like(pressure, dtype=bool)
    slots = np.asarray(decision_slots, dtype=bool)
    threshold = (
        GC_MAPPO_PRESSURE_CONFIDENCE_THRESHOLD
        if threshold is None else float(threshold)
    )
    masked_count = 0
    for agent_index in np.flatnonzero(slots):
        margin = float(np.max(pressure[agent_index]) - np.min(pressure[agent_index]))
        if margin < threshold:
            continue
        masks[agent_index, :] = False
        masks[agent_index, int(np.argmax(pressure[agent_index]))] = True
        masked_count += 1
    return masks, masked_count


def _add_current_phase_hold_prior(action_priors, env, decision_slots, bonus):
    priors = np.asarray(action_priors, dtype=np.float32).copy()
    bonus = float(bonus)
    if bonus == 0.0:
        return priors
    for agent_index in np.flatnonzero(np.asarray(decision_slots, dtype=bool)):
        signal = env.traffic_signals[agent_index]
        current_phase_index = next(
            (
                phase_index
                for phase_index, phase in enumerate(signal.all_green_phases)
                if phase.state == signal.green_phase.state
            ),
            None,
        )
        if current_phase_index is not None:
            priors[agent_index, current_phase_index] += bonus
    return priors


def run_double_dqn(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行 parameter-shared Double DQN 算法。"""
    print(f"[{PARAMETER_SHARED_DDQN_NAME}] Running...")
    sys.stdout.flush()
    
    env = SumoEnv(**ENV_PARAMS)
    replay_buffer = ReplayBuffer(capacity=20000)
    
    # 获取维度信息用于初始化智能体
    temp_env = SumoEnv(**ENV_PARAMS)
    temp_env.reset()
    input_dim = temp_env.observation_space.shape[0]
    output_dim = _infer_action_dim(temp_env)
    temp_env.close()
    
    agent = DoubleDQNAgent(
        mode='train',
        replay_buffer=replay_buffer,
        target_update=3000,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=83000,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=32
    )
    
    return run_single_agent(env, agent, replay_buffer, num_episodes, PARAMETER_SHARED_DDQN_NAME, checkpoint_path,
                            resume_state, on_episode_complete)

def run_dueling_dqn(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行 parameter-shared dueling Double DQN 算法。"""
    print(f"[{PARAMETER_SHARED_DUELING_DDQN_NAME}] Running...")
    sys.stdout.flush()
    
    env = SumoEnv(**ENV_PARAMS)
    replay_buffer = ReplayBuffer(capacity=20000)
    
    temp_env = SumoEnv(**ENV_PARAMS)
    temp_env.reset()
    input_dim = temp_env.observation_space.shape[0]
    output_dim = _infer_action_dim(temp_env)
    temp_env.close()
    
    agent = DuelingDoubleDQNAgent(
        mode='train',
        replay_buffer=replay_buffer,
        target_update=8000,
        gamma=0.92,
        eps_start=1.0,
        eps_end=0.24,
        eps_decay=120000,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=128,
        learning_rate=0.00008,
        warmup_transitions=768,
        train_freq=4,
    )
    
    return run_single_agent(env, agent, replay_buffer, num_episodes, PARAMETER_SHARED_DUELING_DDQN_NAME, checkpoint_path,
                            resume_state, on_episode_complete)


def run_reinforce(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行共享策略、带学习价值基线的 REINFORCE。"""
    print(f"[{SHARED_POLICY_REINFORCE_NAME}] Running...")
    sys.stdout.flush()
    
    env = SumoEnv(**ENV_PARAMS)
    
    temp_env = SumoEnv(**ENV_PARAMS)
    temp_env.reset()
    input_dim = temp_env.observation_space.shape[0]
    output_dim = _infer_action_dim(temp_env)
    temp_env.close()
    
    agent = ReinforceAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        gamma=0.99,
        lr=0.001
    )
    
    return run_reinforce_agent(env, agent, num_episodes, SHARED_POLICY_REINFORCE_NAME, checkpoint_path,
                               resume_state, on_episode_complete)

def run_shared_policy_gae_actor_critic(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行 GAE Actor-Critic 算法。"""
    print(f"[{SHARED_POLICY_GAE_NAME}] Running...")
    sys.stdout.flush()
    
    env = SumoEnv(**ENV_PARAMS)
    
    temp_env = SumoEnv(**ENV_PARAMS)
    temp_env.reset()
    input_dim = temp_env.observation_space.shape[0]
    output_dim = _infer_action_dim(temp_env)
    temp_env.close()
    
    agent = GAEActorCritic(
        input_dim=input_dim,
        output_dim=output_dim,
        gamma=0.99,
        learning_rate=0.0003,
        gae_lambda=0.95,
    )
    
    return run_shared_policy_gae_agent(env, agent, num_episodes, SHARED_POLICY_GAE_NAME, checkpoint_path,
                                       resume_state, on_episode_complete)

def run_gc_mappo(num_episodes=10, checkpoint_path=None, resume_state=None,
                 on_episode_complete=None, agent_class=GCMAPPO,
                 method_name='GC-MAPPO', independent_actor=None,
                 action_prior_scale=None, hold_prior_bonus=None,
                 policy_temperature_end=None, soft_q_execution=None):
    """运行 GC-MAPPO；显式 variant 参数优先，缺省保持旧环境变量兼容。"""
    log_progress(f"[{method_name}] Running.")
    env_params = {**ENV_PARAMS, 'multi_agent': True}
    env = SumoEnv(**env_params)
    initial_states = env.reset()
    num_agents = env.num_agents
    spaces = env.get_agent_spaces()
    raw_obs_dim = spaces[0][0].shape[0]
    action_dim = spaces[0][1].n
    obs_dim = raw_obs_dim + action_dim
    action_prior_scale = float(
        os.environ.get('GC_MAPPO_ACTION_PRIOR_SCALE', '2.25')
        if action_prior_scale is None else action_prior_scale
    )
    hold_prior_bonus = float(
        os.environ.get('GC_MAPPO_HOLD_PRIOR_BONUS', '0.0')
        if hold_prior_bonus is None else hold_prior_bonus
    )
    independent_actor = (
        os.environ.get('GC_MAPPO_INDEPENDENT_ACTOR', '0') == '1'
        if independent_actor is None else bool(independent_actor)
    )
    soft_q_execution = (
        os.environ.get('GC_MAPPO_SOFT_Q_EXECUTION', '1') == '1'
        if soft_q_execution is None else bool(soft_q_execution)
    )
    agent = agent_class(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        adjacency_matrix=env.adjacency_matrix,
        distance_matrix=env.distance_matrix,
        agent_embedding_dim=0,
        independent_actor=independent_actor,
        action_prior_scale=action_prior_scale,
        family='GC-MAPPO',
        prior_mode='pressure_zscore_hold',
        pressure_observation=True,
        relative_value_coef=0.10,
        action_value_head=True,
        action_value_correction_scale=0.10,
        action_value_consensus=True,
        soft_q_execution=soft_q_execution,
        soft_q_policy_agreement=True,
        soft_q_policy_agreement_scale=0.25,
        soft_q_policy_support_only=(
            os.environ.get('GC_MAPPO_SOFT_Q_POLICY_SUPPORT_ONLY', '0') == '1'
        ),
        soft_q_value_disagreement_gate=False,
        soft_q_margin_calibration=False,
        action_value_margin_calibration=False,
        action_value_per_agent_normalization=False,
        action_value_unselected_anchor=False,
        action_value_unselected_conservative_anchor=False,
        action_value_unselected_behavior_anchor=False,
        dense_critic_exclude_decision_states=(
            os.environ.get('GC_MAPPO_DENSE_CRITIC_EXCLUDE_DECISION_STATES', '0') == '1'
        ),
        residual_global_dominant_sharpening=False,
        residual_critic=True,
        adaptive_residual_consensus=True,
        adaptive_residual_energy_mix=True,
        residual_energy_power=2.0,
        max_grad_norm=0.5,
        update_epochs=5,
        clip_ratio=0.20,
        value_coef=0.50,
    )
    entropy_start = 0.010
    entropy_end = 0.001
    temperature_start = 1.00
    temperature_end = float(
        os.environ.get('GC_MAPPO_TEMP_END', '0.30')
        if policy_temperature_end is None else policy_temperature_end
    )
    actor_lr_start = 3e-4
    actor_lr_end = 3e-4
    critic_lr_start = 1e-3
    critic_lr_end = 5e-4
    q_lr_start = 1e-3
    q_lr_end = 5e-4
    clip_ratio_start = 0.20
    clip_ratio_end = 0.20
    metrics = (resume_state or {}).get(
        'metrics', {'rewards': [], 'speeds': [], 'queue_lengths': [], 'waiting_times': []}
    )
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        _restore_rng_state(resume_state.get('rng_state'))
    anneal_episodes = min(num_episodes, 50)
    log_progress(
        f"GC-MAPPO entropy schedule: {entropy_start:.4f} -> {entropy_end:.4f} "
        f"over {anneal_episodes} episodes, then hold."
    )
    log_progress(
        f"GC-MAPPO policy-temperature schedule: {temperature_start:.2f} -> {temperature_end:.2f} "
        f"over {anneal_episodes} episodes, then hold."
    )
    log_progress(
        f"GC-MAPPO action_prior_scale={agent.action_prior_scale:.3f}, "
        f"hold_prior_bonus={hold_prior_bonus:.3f} "
        f"(env GC_MAPPO_ACTION_PRIOR_SCALE={'set' if os.environ.get('GC_MAPPO_ACTION_PRIOR_SCALE') else 'default'}, "
        f"GC_MAPPO_HOLD_PRIOR_BONUS={'set' if os.environ.get('GC_MAPPO_HOLD_PRIOR_BONUS') else 'default'}, "
        f"GC_MAPPO_TEMP_END={'set' if os.environ.get('GC_MAPPO_TEMP_END') else 'default'})"
    )
    log_progress(
        f"GC-MAPPO late learning-rate schedule: actor fixed at {actor_lr_start:.1e}, "
        f"critic {critic_lr_start:.1e} -> {critic_lr_end:.1e} from episode 26 onward."
    )
    log_progress(
        "GC-MAPPO critic relative-value consistency coefficient: 0.10."
    )
    log_progress(
        "GC-MAPPO C46 uses semi-Markov actor targets with a full-density C37 vector-critic loss."
    )
    log_progress(
        "GC-MAPPO C157 sharpens the C111 residual handoff only where the global "
        "residual already dominates, preserving local correction where it does not."
    )
    log_progress(
        "GC-MAPPO C71 keeps the shared C60 critic and projects only dense-critic gradients that "
        "directly conflict with executable-decision critic gradients."
    )
    log_progress(
        "GC-MAPPO C74 keeps C71 and projects only entropy gradients that oppose the PPO policy objective."
    )
    log_progress(
        "GC-MAPPO C85 freezes the rollout Q bias inside each PPO batch while retaining epoch-wise Q learning."
    )
    log_progress(
        "GC-MAPPO C89 uses a zero-initialized residual centralized critic to learn only baseline correction."
    )
    log_progress(
        "GC-MAPPO C99 adaptively shares residual value only when its global component "
        "dominates local residual disagreement; no episode-index switch is used."
    )
    log_progress(
        "GC-MAPPO C111 uses the residual energy share to sharpen C99's state-dependent "
        "handoff between network-wide and junction-specific correction."
    )
    log_progress(
        "GC-MAPPO C159 keeps C111 execution unchanged and anchors only unselected "
        "Q outputs to the frozen primary critic value, preventing unsupervised "
        "alternative-action rankings from driving the soft-Q bias."
    )
    log_progress(
        "GC-MAPPO C160 keeps C111 execution unchanged and applies a conservative "
        "hinge only to unselected Q values above the critic baseline, preserving "
        "negative action evidence while suppressing unsupported optimistic rankings."
    )
    log_progress(
        "GC-MAPPO C161 narrows the conservative hinge to unselected Q values that "
        "outrank the executed action, preserving valid alternatives below the "
        "behavior action while suppressing unsupported reversals."
    )
    log_progress(
        "GC-MAPPO keeps C111's 0.25 policy-Q concordance amplifier: both lower "
        "(C123) and higher (C125) strengths regressed all primary metrics."
    )
    log_progress(
        "GC-MAPPO C152 restores C111's accepted schedules and keeps soft-Q guidance "
        "only when both Q heads positively support the actor's own preferred action. "
        "This removes execution-time ranking of unexecuted alternatives, which have "
        "no direct return target in the selected-action Q regression."
    )
    for episode in range(start_episode, num_episodes):
        progress = min(episode / max(1, anneal_episodes - 1), 1.0)
        late_progress = max(0.0, (episode - 24) / max(1, anneal_episodes - 25))
        actor_lr = actor_lr_start + (actor_lr_end - actor_lr_start) * late_progress
        critic_lr = critic_lr_start + (critic_lr_end - critic_lr_start) * late_progress
        q_lr = q_lr_start + (q_lr_end - q_lr_start) * late_progress
        agent.clip_ratio = (
            clip_ratio_start + (clip_ratio_end - clip_ratio_start) * late_progress
        )
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr
        if agent.action_value_optimizer is not None:
            for param_group in agent.action_value_optimizer.param_groups:
                param_group['lr'] = q_lr
        if agent.action_value_reference_optimizer is not None:
            for param_group in agent.action_value_reference_optimizer.param_groups:
                param_group['lr'] = q_lr
        agent.entropy_coef = entropy_end + (entropy_start - entropy_end) * (1.0 - progress)
        policy_temperature = temperature_end + (temperature_start - temperature_end) * (1.0 - progress)
        states = env.reset()
        done = False
        episode_rewards = []
        episode_reward_components = {}
        dense_observations = []
        dense_rewards = []
        dense_values = []
        dense_dones = []
        dense_critic_decision_rows = []
        decision_observations = []
        decision_actions = []
        decision_action_masks = []
        decision_log_probs = []
        decision_rewards = []
        decision_action_priors = []
        decision_directional_features = []
        decision_policy_temperatures = []
        decision_values = []
        decision_dones = []
        decision_masks = []
        decision_bootstrap_discounts = []
        decision_gae_discounts = []
        pending_transition_rows = [None] * num_agents
        pending_rewards = np.zeros(num_agents, dtype=np.float32)
        pending_durations = np.zeros(num_agents, dtype=np.int32)
        pending_decision_kinds = [None] * num_agents
        execution_diagnostics = {
            'decision_slots': 0,
            'hold_count': 0,
            'switch_count': 0,
            'hold_margin_sum': 0.0,
            'switch_margin_sum': 0.0,
            'hold_prior_strength_sum': 0.0,
            'switch_prior_strength_sum': 0.0,
            'hold_prior_saturated_count': 0,
            'switch_prior_saturated_count': 0,
            'hold_reward_sum': 0.0,
            'switch_reward_sum': 0.0,
            'hold_duration_sum': 0.0,
            'switch_duration_sum': 0.0,
            'hold_duration_count': 0,
            'switch_duration_count': 0,
            'execution_mismatches': 0,
            'nominal_yellow_advance_count': 0,
            'pressure_confidence_masked_count': 0,
        }

        def record_completed_decision(agent_index):
            decision_kind = pending_decision_kinds[agent_index]
            duration = int(pending_durations[agent_index])
            if decision_kind is None or duration <= 0:
                return
            execution_diagnostics[f'{decision_kind}_duration_sum'] += duration
            execution_diagnostics[f'{decision_kind}_duration_count'] += 1

        episode_speeds = []
        episode_queues = []
        episode_waiting = []
        last_metric_sample_time = None
        update_metrics = {}

        while not done:
            local_states = np.asarray([
                _normalize_state_dim(state, raw_obs_dim) for state in states
            ], dtype=np.float32)
            raw_pressure = np.asarray([
                signal.pressure_state() for signal in env.traffic_signals
            ], dtype=np.float32)
            # Standardize within each junction so the prior encodes the relative
            # pressure ordering rather than an arbitrary traffic-volume scale.
            pressure_center = raw_pressure.mean(axis=1, keepdims=True)
            pressure_scale = raw_pressure.std(axis=1, keepdims=True)
            pressure_prior = np.clip(
                (raw_pressure - pressure_center) / np.maximum(pressure_scale, 1.0),
                -3.0,
                3.0,
            ).astype(np.float32)
            normalized_states = np.concatenate([local_states, pressure_prior], axis=1)
            valid_masks = np.ones((num_agents, action_dim), dtype=bool)
            decision_slots = np.asarray([
                signal.yellow_phase is None
                and env.sumo.simulation.getTime() >= signal.end_time
                for signal in env.traffic_signals
            ], dtype=bool)
            if os.environ.get('GC_MAPPO_PRESSURE_CONFIDENCE_MASK') == '1':
                valid_masks, pressure_masked_count = _pressure_confidence_action_masks(
                    raw_pressure, decision_slots,
                )
                execution_diagnostics['pressure_confidence_masked_count'] += pressure_masked_count
            execution_prior = _add_current_phase_hold_prior(
                pressure_prior, env, decision_slots, hold_prior_bonus,
            )
            directional_features = directional_phase_features(env.traffic_signals) if method_name == 'DPG-MAPPO' else None
            selected_actions, log_probs, value = agent.select_actions(
                normalized_states,
                action_masks=valid_masks,
                action_priors=execution_prior,
                policy_temperature=policy_temperature,
                **({'directional_features': directional_features} if directional_features is not None else {}),
            )

            # Only phase choices that can take effect train the actor. The dense
            # critic stream below still observes every original control interval.
            decision_row_indices = np.full(num_agents, -1, dtype=np.int32)
            decision_kinds = [None] * num_agents
            if np.any(decision_slots):
                for agent_index in np.flatnonzero(decision_slots):
                    prior_row = pending_transition_rows[agent_index]
                    if prior_row is not None:
                        record_completed_decision(agent_index)
                        decision_rewards[prior_row][agent_index] = pending_rewards[agent_index]
                        decision_bootstrap_discounts[prior_row][agent_index] = (
                            agent.gamma ** pending_durations[agent_index]
                        )
                        decision_gae_discounts[prior_row][agent_index] = (
                            (agent.gamma * agent.gae_lambda) ** pending_durations[agent_index]
                        )

                transition_row = len(decision_observations)
                decision_observations.append(normalized_states.copy())
                decision_actions.append(np.asarray(selected_actions, dtype=np.int64))
                decision_action_masks.append(valid_masks.copy())
                decision_log_probs.append(np.asarray(log_probs, dtype=np.float32))
                decision_rewards.append(np.zeros(num_agents, dtype=np.float32))
                decision_action_priors.append(execution_prior.copy())
                if directional_features is not None:
                    decision_directional_features.append(directional_features.copy())
                decision_policy_temperatures.append(policy_temperature)
                decision_values.append(np.asarray(value, dtype=np.float32))
                decision_dones.append(np.zeros(num_agents, dtype=np.float32))
                decision_masks.append(decision_slots.copy())
                decision_bootstrap_discounts.append(np.ones(num_agents, dtype=np.float32))
                decision_gae_discounts.append(np.ones(num_agents, dtype=np.float32))
                for agent_index in np.flatnonzero(decision_slots):
                    signal = env.traffic_signals[agent_index]
                    current_phase_index = next(
                        (
                            phase_index
                            for phase_index, phase in enumerate(signal.all_green_phases)
                            if phase.state == signal.green_phase.state
                        ),
                        None,
                    )
                    decision_kind = (
                        'hold'
                        if int(selected_actions[agent_index]) == current_phase_index
                        else 'switch'
                    )
                    pressure_margin = float(
                        np.max(raw_pressure[agent_index]) - np.min(raw_pressure[agent_index])
                    )
                    prior_strength = float(np.max(np.abs(pressure_prior[agent_index])))
                    prior_saturated = bool(np.std(raw_pressure[agent_index]) >= 1.0)
                    decision_kinds[agent_index] = decision_kind
                    execution_diagnostics['decision_slots'] += 1
                    execution_diagnostics[f'{decision_kind}_count'] += 1
                    execution_diagnostics[f'{decision_kind}_margin_sum'] += pressure_margin
                    execution_diagnostics[f'{decision_kind}_prior_strength_sum'] += prior_strength
                    execution_diagnostics[f'{decision_kind}_prior_saturated_count'] += int(prior_saturated)
                    pending_transition_rows[agent_index] = transition_row
                    pending_rewards[agent_index] = 0.0
                    pending_durations[agent_index] = 0
                    pending_decision_kinds[agent_index] = decision_kind
                    decision_row_indices[agent_index] = transition_row

            next_states, rewards, done, info = env.step(selected_actions)
            reward_array = np.asarray(
                [0.0 if reward is None else float(reward) for reward in rewards],
                dtype=np.float32,
            )
            reward_value = float(reward_array.mean())
            _accumulate_reward_components(episode_reward_components, info.get('reward_components'))
            execution_diagnostics['nominal_yellow_advance_count'] += int(
                info.get('nominal_yellow_advance_count', 0)
            )
            actual_actions = info.get('do_actions', [])
            for agent_index, decision_kind in enumerate(decision_kinds):
                if decision_kind is None:
                    continue
                execution_diagnostics[f'{decision_kind}_reward_sum'] += reward_array[agent_index]
                actual_action = (
                    actual_actions[agent_index]
                    if agent_index < len(actual_actions) else None
                )
                expected_action = None if decision_kind == 'hold' else -1
                if actual_action != expected_action:
                    execution_diagnostics['execution_mismatches'] += 1

            dense_observations.append(normalized_states)
            dense_rewards.append(reward_array)
            dense_values.append(value)
            dense_dones.append(bool(done))
            dense_critic_decision_rows.append(decision_row_indices)
            episode_rewards.append(reward_value)
            for agent_index, transition_row in enumerate(pending_transition_rows):
                if transition_row is not None:
                    pending_rewards[agent_index] += (
                        agent.gamma ** pending_durations[agent_index]
                    ) * reward_array[agent_index]
                    pending_durations[agent_index] += 1
                    if done:
                        record_completed_decision(agent_index)
                        decision_rewards[transition_row][agent_index] = pending_rewards[agent_index]
                        decision_dones[transition_row][agent_index] = 1.0
                        decision_bootstrap_discounts[transition_row][agent_index] = 0.0
                        decision_gae_discounts[transition_row][agent_index] = 0.0

            states = [
                _normalize_state_dim(next_state, raw_obs_dim) if next_state is not None else local_states[i]
                for i, next_state in enumerate(next_states)
            ]
            try:
                last_metric_sample_time = _maybe_record_metrics(
                    env, episode_speeds, episode_queues, episode_waiting,
                    last_metric_sample_time, done,
                )
            except Exception:
                pass

        if decision_observations:
            update_metrics = agent.update(
                observations=np.asarray(decision_observations, dtype=np.float32),
                actions=np.asarray(decision_actions, dtype=np.int64),
                old_log_probs=np.asarray(decision_log_probs, dtype=np.float32),
                rewards=np.asarray(decision_rewards, dtype=np.float32),
                values=np.asarray(decision_values, dtype=np.float32),
                dones=np.asarray(decision_dones, dtype=np.float32),
                action_masks=np.asarray(decision_action_masks, dtype=bool),
                action_priors=np.asarray(decision_action_priors, dtype=np.float32),
                policy_temperatures=np.asarray(decision_policy_temperatures, dtype=np.float32),
                transition_mask=np.asarray(decision_masks, dtype=bool),
                transition_discounts=np.asarray(decision_bootstrap_discounts, dtype=np.float32),
                gae_discounts=np.asarray(decision_gae_discounts, dtype=np.float32),
                dense_critic_observations=np.asarray(dense_observations, dtype=np.float32),
                dense_critic_rewards=np.asarray(dense_rewards, dtype=np.float32),
                dense_critic_values=np.asarray(dense_values, dtype=np.float32),
                dense_critic_dones=np.asarray(dense_dones, dtype=np.float32),
                dense_critic_decision_rows=np.asarray(dense_critic_decision_rows, dtype=np.int32),
                **({'directional_features': np.asarray(decision_directional_features, dtype=np.float32)} if decision_directional_features else {}),
            )

        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queues, episode_waiting
        )
        metrics['rewards'].append(_safe_mean(episode_rewards))
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        metrics.setdefault('reward_components', []).append(
            _finalize_reward_components(episode_reward_components)
        )
        decision_slot_count = max(1, execution_diagnostics['decision_slots'])
        hold_count = execution_diagnostics['hold_count']
        switch_count = execution_diagnostics['switch_count']
        hold_duration_count = max(1, execution_diagnostics['hold_duration_count'])
        switch_duration_count = max(1, execution_diagnostics['switch_duration_count'])
        log_progress(
            f"GC-MAPPO - Episode {episode + 1}/{num_episodes} done. "
            f"Avg Reward: {metrics['rewards'][-1]:.4f}, "
            f"Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}, "
            f"EntropyCoef: {agent.entropy_coef:.4f}, PolicyTemp: {policy_temperature:.3f}, "
            f"ActorLR: {actor_lr:.1e}, CriticLR: {critic_lr:.1e}, QLR: {q_lr:.1e}, "
            f"PriorScale: {agent.action_prior_scale:.3f}, "
            f"DecisionTransitions: {len(decision_observations)}/"
            f"{int(np.asarray(decision_masks, dtype=bool).sum()) if decision_masks else 0}, "
            f"PressureMaskRate: {execution_diagnostics.get('pressure_confidence_masked_count', 0) / decision_slot_count:.4f}, "
            f"DenseCriticSteps: {len(dense_observations)}, "
            f"DenseDecisionExcl: {update_metrics.get('dense_critic_decision_exclusion_rate', 0.0):.4f}, "
            f"DenseTargetGap: {update_metrics.get('dense_critic_decision_target_gap', 0.0):.4f}, "
            f"Qloss: {update_metrics.get('action_value_loss', float('nan')):.4f}, "
            f"Qanchor: {update_metrics.get('action_value_unselected_anchor_loss', float('nan')):.4f}, "
            f"Qconf: {update_metrics.get('action_value_confidence', float('nan')):.4f}, "
            f"QcorrStd: {update_metrics.get('action_value_correction_std', float('nan')):.4f}, "
            f"Qagree: {update_metrics.get('action_value_consensus_rate', float('nan')):.4f}, "
            f"QBiasFrozen: {update_metrics.get('q_bias_frozen', 0.0):.0f}, "
            f"ResidualCritic: {update_metrics.get('residual_critic', 0.0):.0f}, "
            f"ResidualAdaptive: {update_metrics.get('adaptive_residual_consensus', 0.0):.0f}, "
            f"ResidualEnergyMix: {update_metrics.get('adaptive_residual_energy_mix', 0.0):.0f}, "
            f"ResidualGlobalMix: {update_metrics.get('residual_global_mix', 0.0):.4f}, "
            f"ResidualGlobalSharpen: {agent.last_residual_global_sharpen_rate:.4f}, "
            f"ResidualGradCos: {update_metrics.get('residual_critic_gradient_cosine', float('nan')):.4f}, "
            f"ResidualGradProj: {update_metrics.get('residual_critic_gradient_projected', 0.0):.0f}, "
            f"ActorGradCos: {update_metrics.get('actor_gradient_cosine', float('nan')):.4f}, "
            f"ActorGradProj: {update_metrics.get('actor_gradient_projected', float('nan')):.0f}, "
            f"CriticGradCos: {update_metrics.get('critic_gradient_cosine', float('nan')):.4f}, "
            f"CriticGradProj: {update_metrics.get('critic_gradient_projected', float('nan')):.0f}, "
            f"Qsoft: {agent.last_soft_q_execution_rate:.4f}, "
            f"Qpolicy: {agent.last_soft_q_policy_agreement:.4f}, "
            f"Qsupport: {agent.last_soft_q_policy_support_rate:.4f}, "
            f"QvalueAgree: {agent.last_soft_q_value_agreement:.4f}, "
            f"QmarginScale: {agent.last_soft_q_margin_scale:.4f}, "
            f"QupdateScale: {agent.last_action_value_update_scale:.4f}, "
            f"QagentNorm: {float(agent.action_value_per_agent_normalization):.0f}, "
            f"QagentScaleSpread: {agent.last_action_value_agent_scale_spread:.4f}, "
            f"ExecHoldRate: {hold_count / decision_slot_count:.4f}, "
            f"ExecSwitchRate: {switch_count / decision_slot_count:.4f}, "
            f"HoldMargin: {execution_diagnostics['hold_margin_sum'] / max(1, hold_count):.4f}, "
            f"SwitchMargin: {execution_diagnostics['switch_margin_sum'] / max(1, switch_count):.4f}, "
            f"HoldPriorAbs: {execution_diagnostics['hold_prior_strength_sum'] / max(1, hold_count):.4f}, "
            f"SwitchPriorAbs: {execution_diagnostics['switch_prior_strength_sum'] / max(1, switch_count):.4f}, "
            f"HoldPriorSat: {execution_diagnostics['hold_prior_saturated_count'] / max(1, hold_count):.4f}, "
            f"SwitchPriorSat: {execution_diagnostics['switch_prior_saturated_count'] / max(1, switch_count):.4f}, "
            f"HoldDur: {execution_diagnostics['hold_duration_sum'] / hold_duration_count:.2f}, "
            f"SwitchDur: {execution_diagnostics['switch_duration_sum'] / switch_duration_count:.2f}, "
            f"HoldStepReward: {execution_diagnostics['hold_reward_sum'] / max(1, hold_count):.4f}, "
            f"SwitchStepReward: {execution_diagnostics['switch_reward_sum'] / max(1, switch_count):.4f}, "
            f"ExecMismatch: {execution_diagnostics['execution_mismatches']}, "
            f"YellowAdvancePerStep: {execution_diagnostics['nominal_yellow_advance_count'] / max(1, len(dense_observations)):.4f}, "
            f"YellowAdvancePerDecision: {execution_diagnostics['nominal_yellow_advance_count'] / decision_slot_count:.4f}"
        )
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, agent)
        sys.stdout.flush()

    env.close()
    if checkpoint_path:
        agent.save(checkpoint_path)
    log_progress(f"[{method_name}] Completed.")
    return {method_name: metrics}


def run_gc_mappo_ind_no_prior(num_episodes=10, checkpoint_path=None, resume_state=None,
                              on_episode_complete=None):
    return run_gc_mappo(
        num_episodes, checkpoint_path, resume_state, on_episode_complete,
        method_name='GC-MAPPO-ind-noP', independent_actor=True,
        action_prior_scale=0.0, hold_prior_bonus=0.0,
        policy_temperature_end=0.30, soft_q_execution=True,
    )


def run_gc_mappo_ind_prior(num_episodes=10, checkpoint_path=None, resume_state=None,
                           on_episode_complete=None):
    return run_gc_mappo(
        num_episodes, checkpoint_path, resume_state, on_episode_complete,
        method_name='GC-MAPPO-ind-P', independent_actor=True,
        action_prior_scale=2.25, hold_prior_bonus=0.0,
        policy_temperature_end=0.30, soft_q_execution=True,
    )


def run_dpg_mappo(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    return run_gc_mappo(
        num_episodes, checkpoint_path, resume_state, on_episode_complete,
        agent_class=DPGMAPPO, method_name='DPG-MAPPO',
    )


def _multi_agent_spec():
    env = SumoEnv(**{**ENV_PARAMS, 'multi_agent': True})
    env.reset()
    spaces = env.get_agent_spaces()
    obs_dim = spaces[0][0].shape[0]
    action_dim = spaces[0][1].n
    if any(space[0].shape[0] != obs_dim or space[1].n != action_dim for space in spaces):
        env.close()
        raise ValueError('All intersections must expose identical observation and action dimensions.')
    return env, env.num_agents, obs_dim, action_dim


def _run_on_policy_multi_agent(env, agent, num_episodes, algo_name, checkpoint_path=None,
                              resume_state=None, on_episode_complete=None,
                              observation_transform=None, action_prior_builder=None,
                              policy_temperature=1.0):
    metrics = (resume_state or {}).get(
        'metrics', {'rewards': [], 'speeds': [], 'queue_lengths': [], 'waiting_times': []}
    )
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        _restore_rng_state(resume_state.get('rng_state'))
    for episode in range(start_episode, num_episodes):
        states = env.reset()
        observations, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []
        action_priors, policy_temperatures = [], []
        episode_rewards, episode_speeds, episode_queues, episode_waiting = [], [], [], []
        episode_reward_components = {}
        last_metric_sample_time = None
        done = False

        while not done:
            state_array = np.asarray(states, dtype=np.float32)
            if observation_transform is not None:
                state_array = observation_transform(env, state_array)
            step_priors = None if action_prior_builder is None else action_prior_builder(env)
            select_kwargs = {}
            if isinstance(agent, IPPO):
                select_kwargs = {
                    'action_priors': step_priors,
                    'policy_temperature': policy_temperature,
                }
            selected_actions, log_probs, value = agent.select_actions(
                state_array, **select_kwargs
            )
            next_states, step_rewards, done, info = env.step(selected_actions)
            reward_array = np.asarray([0.0 if reward is None else float(reward) for reward in step_rewards], dtype=np.float32)
            _accumulate_reward_components(episode_reward_components, info.get('reward_components'))
            _append_effective_multi_rewards(
                episode_rewards, step_rewards, info.get('do_actions', []), next_states, done,
            )
            observations.append(state_array)
            actions.append(selected_actions)
            old_log_probs.append(log_probs)
            rewards.append(reward_array)
            values.append(value)
            dones.append(bool(done))
            if step_priors is not None:
                action_priors.append(np.asarray(step_priors, dtype=np.float32))
            policy_temperatures.append(float(policy_temperature))
            states = [
                _normalize_state_dim(next_state, env.traffic_signals[index].observation_space.shape[0])
                if next_state is not None else states[index]
                for index, next_state in enumerate(next_states)
            ]
            last_metric_sample_time = _maybe_record_metrics(
                env, episode_speeds, episode_queues, episode_waiting, last_metric_sample_time, done,
            )

        if observations:
            if isinstance(agent, MAPPO):
                agent.update(
                    np.asarray(observations), np.asarray(actions), np.asarray(old_log_probs),
                    np.mean(np.asarray(rewards), axis=1), np.asarray(values), np.asarray(dones),
                )
            else:
                update_kwargs = {}
                if action_prior_builder is not None:
                    update_kwargs['action_priors'] = np.asarray(action_priors, dtype=np.float32)
                    update_kwargs['policy_temperatures'] = np.asarray(
                        policy_temperatures, dtype=np.float32
                    )
                agent.update(
                    np.asarray(observations), np.asarray(actions), np.asarray(old_log_probs),
                    np.asarray(rewards), np.asarray(values), np.asarray(dones),
                    **update_kwargs,
                )
        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queues, episode_waiting,
        )
        metrics['rewards'].append(_safe_mean(episode_rewards))
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        metrics.setdefault('reward_components', []).append(
            _finalize_reward_components(episode_reward_components)
        )
        log_progress(
            f"{algo_name} - Episode {episode + 1}/{num_episodes} done. "
            f"Avg Reward: {metrics['rewards'][-1]:.4f}, Speed: {avg_speed:.2f}, "
            f"Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
        )
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, agent)
    env.close()
    if checkpoint_path:
        agent.save(checkpoint_path)
    return {algo_name: metrics}


def run_mappo(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run the graph-free MAPPO ablation under the same CTDE budget as GC-MAPPO."""
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    agent = MAPPO(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim)
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'MAPPO', checkpoint_path, resume_state, on_episode_complete
    )


def _neighbor_augmented_observations(env, states):
    observations = np.asarray(states, dtype=np.float32)
    adjacency = np.asarray(env.adjacency_matrix, dtype=np.float32)
    neighbor_sum = adjacency @ observations
    degree = adjacency.sum(axis=1, keepdims=True)
    neighbor_mean = neighbor_sum / np.maximum(degree, 1.0)
    return np.concatenate([observations, neighbor_mean], axis=1)


def _pressure_state_observations(env, states):
    """Append the clipped within-intersection z-scored pressure vector as state.

    Uses the identical pressure computation as the IPPO-P logit prior
    (population SD, SD floor 1.0, clip [-3, 3]) but supplies it through the
    observation channel instead of the policy logits.
    """
    observations = np.asarray(states, dtype=np.float32)
    pressure = np.asarray([
        signal.pressure_state() for signal in env.traffic_signals
    ], dtype=np.float32)
    pressure_center = pressure.mean(axis=1, keepdims=True)
    pressure_scale = pressure.std(axis=1, keepdims=True)
    pressure_zscore = np.clip(
        (pressure - pressure_center) / np.maximum(pressure_scale, 1.0),
        -3.0, 3.0,
    ).astype(np.float32)
    return np.concatenate([observations, pressure_zscore], axis=1)


def _pressure_hold_action_priors(env, hold_prior_bonus=0.0):
    raw_pressure = np.asarray([
        signal.pressure_state() for signal in env.traffic_signals
    ], dtype=np.float32)
    pressure_center = raw_pressure.mean(axis=1, keepdims=True)
    pressure_scale = raw_pressure.std(axis=1, keepdims=True)
    pressure_prior = np.clip(
        (raw_pressure - pressure_center) / np.maximum(pressure_scale, 1.0),
        -3.0, 3.0,
    ).astype(np.float32)
    decision_slots = np.asarray([
        signal.yellow_phase is None
        and env.sumo.simulation.getTime() >= signal.end_time
        for signal in env.traffic_signals
    ], dtype=bool)
    return _add_current_phase_hold_prior(
        pressure_prior, env, decision_slots, hold_prior_bonus
    )


def run_ippo(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run independent PPO with no graph communication or centralized critic."""
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    agent = IPPO(
        num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim,
        action_prior_scale=0.0, prior_mode='none', policy_temperature=1.0,
    )
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'IPPO', checkpoint_path, resume_state, on_episode_complete,
        policy_temperature=1.0,
    )


def run_ippo_prior(num_episodes=10, checkpoint_path=None, resume_state=None,
                   on_episode_complete=None, action_prior_scale=2.25,
                   hold_prior_bonus=None):
    """Run IPPO with the GC pressure/hold prior and unchanged local observations."""
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    hold_bonus = float(
        os.environ.get('GC_MAPPO_HOLD_PRIOR_BONUS', '0.0')
        if hold_prior_bonus is None else hold_prior_bonus
    )
    # S3 scale probe only: task-wide override of the frozen prior scale.
    scale_override = os.environ.get('GC_MAPPO_IPPO_PRIOR_SCALE')
    if scale_override:
        action_prior_scale = float(scale_override)
    agent = IPPO(
        num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim,
        action_prior_scale=action_prior_scale,
        prior_mode='pressure_zscore_hold', policy_temperature=1.0,
    )
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'IPPO-P', checkpoint_path, resume_state,
        on_episode_complete,
        action_prior_builder=lambda current_env: _pressure_hold_action_priors(
            current_env, hold_bonus
        ),
        policy_temperature=1.0,
    )


def run_gated_graph_residual_ippo(num_episodes=10, checkpoint_path=None, resume_state=None,
                                  on_episode_complete=None, freeze_actor=False,
                                  initialization_checkpoint=None):
    """Exploratory BR-IPPO or GR-GMAPPO-L initialized from a matching IPPO actor."""
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    source = initialization_checkpoint or os.environ.get('GC_MAPPO_IPPO_INIT_CHECKPOINT')
    if resume_state is None and (not source or not os.path.isfile(source)):
        env.close()
        raise ValueError('Set GC_MAPPO_IPPO_INIT_CHECKPOINT to a matching-seed IPPO checkpoint.')
    agent = GatedGraphResidualIPPO(num_agents, obs_dim, action_dim, env.adjacency_matrix,
                                   freeze_actor=freeze_actor)
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
    else:
        agent.initialize_from_ippo(source)
    name = 'BR-IPPO' if freeze_actor else 'GR-GMAPPO-L'
    return _run_on_policy_multi_agent(env, agent, num_episodes, name, checkpoint_path,
                                      resume_state, on_episode_complete)


def run_br_ippo(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    return run_gated_graph_residual_ippo(num_episodes, checkpoint_path, resume_state,
                                         on_episode_complete, freeze_actor=True)


def run_gr_gmappo_l(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    return run_gated_graph_residual_ippo(num_episodes, checkpoint_path, resume_state,
                                         on_episode_complete, freeze_actor=False)


def run_ippo_plus(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run IPPO with fixed one-hop neighbor mean information concatenated to each observation."""
    env, num_agents, raw_obs_dim, action_dim = _multi_agent_spec()
    obs_dim = raw_obs_dim * 2
    agent = IPPO(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim)
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'IPPO+', checkpoint_path, resume_state, on_episode_complete,
        observation_transform=_neighbor_augmented_observations,
    )


def run_ippo_state(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Matched arm: IPPO with the pressure vector as actor state and no logit prior."""
    env, num_agents, raw_obs_dim, action_dim = _multi_agent_spec()
    pressure_dim = len(env.traffic_signals[0].pressure_state())
    agent = IPPO(
        num_agents=num_agents, obs_dim=raw_obs_dim + pressure_dim, action_dim=action_dim,
        action_prior_scale=0.0, prior_mode='none', policy_temperature=1.0,
    )
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'IPPO-State', checkpoint_path, resume_state, on_episode_complete,
        observation_transform=_pressure_state_observations,
    )


def run_ippo_cap(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run local IPPO with hidden width matched to IPPO+ parameter count."""
    env, num_agents, raw_obs_dim, action_dim = _multi_agent_spec()
    plus_parameters = ippo_parameter_count(raw_obs_dim * 2, action_dim, 128, num_agents)
    hidden_dim = matched_hidden_dim(raw_obs_dim, action_dim, plus_parameters, num_agents)
    agent = IPPO(
        num_agents=num_agents, obs_dim=raw_obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
    )
    return _run_on_policy_multi_agent(
        env, agent, num_episodes, 'IPPO-Cap', checkpoint_path, resume_state, on_episode_complete,
    )


def _pressure_features(env):
    return np.asarray([signal.pressure_state() for signal in env.traffic_signals], dtype=np.float32)


def _run_value_based_multi_agent(
    env, agent, num_episodes, algo_name, feature_fn, reward_fn=None, checkpoint_path=None,
    resume_state=None, on_episode_complete=None,
):
    metrics = (resume_state or {}).get(
        'metrics', {'rewards': [], 'speeds': [], 'queue_lengths': [], 'waiting_times': []}
    )
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        _restore_rng_state(resume_state.get('rng_state'))
    for episode in range(start_episode, num_episodes):
        env.reset()
        features = feature_fn(env)
        episode_rewards, episode_speeds, episode_queues, episode_waiting = [], [], [], []
        last_metric_sample_time = None
        done = False
        while not done:
            selected_actions = agent.select_actions(features)
            next_states, step_rewards, done, info = env.step(selected_actions)
            next_features = feature_fn(env)
            training_rewards = reward_fn(next_features) if reward_fn else np.asarray(
                [0.0 if reward is None else float(reward) for reward in step_rewards], dtype=np.float32,
            )
            _append_effective_multi_rewards(
                episode_rewards, step_rewards, info.get('do_actions', []), next_states, done,
            )
            if getattr(agent, 'formal_name', None) == PRESSLIGHT_NAME or isinstance(agent, PressureStateDoubleDQN):
                agent.store(features, selected_actions, training_rewards, next_features, [done] * env.num_agents)
            else:
                agent.store(features, selected_actions, training_rewards, next_features, done)
            agent.learn()
            features = next_features
            last_metric_sample_time = _maybe_record_metrics(
                env, episode_speeds, episode_queues, episode_waiting, last_metric_sample_time, done,
            )
        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queues, episode_waiting,
        )
        metrics['rewards'].append(_safe_mean(episode_rewards))
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        log_progress(
            f"{algo_name} - Episode {episode + 1}/{num_episodes} done. "
            f"Avg Reward: {metrics['rewards'][-1]:.4f}, Speed: {avg_speed:.2f}, "
            f"Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
        )
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, agent)
    env.close()
    if checkpoint_path:
        agent.save(checkpoint_path)
    return {algo_name: metrics}


def run_graph_attention_double_dqn(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run graph-attention Double DQN with one network shared across intersections."""
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    agent = GraphAttentionDoubleDQN(num_agents, obs_dim, action_dim, env.adjacency_matrix)
    return _run_value_based_multi_agent(
        env, agent, num_episodes, GRAPH_ATTENTION_DDQN_NAME,
        lambda current_env: np.asarray(current_env._compute_states(), dtype=np.float32),
        checkpoint_path=checkpoint_path, resume_state=resume_state,
        on_episode_complete=on_episode_complete,
    )


def run_pressure_state_double_dqn(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """Run the optional pressure-state mechanism baseline."""
    env, _, _, action_dim = _multi_agent_spec()
    agent = PressureStateDoubleDQN(input_dim=action_dim, action_dim=action_dim)
    return _run_value_based_multi_agent(
        env, agent, num_episodes, PRESSURE_STATE_DDQN_NAME, _pressure_features,
        reward_fn=lambda pressures: -np.sum(np.abs(pressures), axis=1),
        checkpoint_path=checkpoint_path, resume_state=resume_state,
        on_episode_complete=on_episode_complete,
    )


def run_presslight(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    env, _, _, action_dim = _multi_agent_spec()
    agent = PressLight(input_dim=action_dim, action_dim=action_dim)
    return _run_value_based_multi_agent(
        env, agent, num_episodes, PRESSLIGHT_NAME, _pressure_features,
        reward_fn=lambda pressures: -np.sum(np.abs(pressures), axis=1),
        checkpoint_path=checkpoint_path, resume_state=resume_state,
        on_episode_complete=on_episode_complete,
    )


def run_colight(num_episodes=10, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    env, num_agents, obs_dim, action_dim = _multi_agent_spec()
    agent = CoLight(num_agents, obs_dim, action_dim, env.adjacency_matrix)
    return _run_value_based_multi_agent(
        env, agent, num_episodes, COLIGHT_NAME,
        lambda current_env: np.asarray(current_env._compute_states(), dtype=np.float32),
        checkpoint_path=checkpoint_path, resume_state=resume_state,
        on_episode_complete=on_episode_complete,
    )


def _run_deterministic_controller(controller, name, num_episodes=10, resume_state=None, on_episode_complete=None):
    env, _, _, _ = _multi_agent_spec()
    metrics = (resume_state or {}).get(
        'metrics', {'rewards': [], 'speeds': [], 'queue_lengths': [], 'waiting_times': []}
    )
    start_episode = int((resume_state or {}).get('episode', 0))
    for episode in range(start_episode, num_episodes):
        env.reset()
        episode_rewards, episode_speeds, episode_queues, episode_waiting = [], [], [], []
        last_metric_sample_time = None
        done = False
        while not done:
            actions = controller.select_actions(env.traffic_signals)
            next_states, rewards, done, info = env.step(actions)
            _append_effective_multi_rewards(
                episode_rewards, rewards, info.get('do_actions', []), next_states, done,
            )
            last_metric_sample_time = _maybe_record_metrics(
                env, episode_speeds, episode_queues, episode_waiting, last_metric_sample_time, done,
            )
        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queues, episode_waiting,
        )
        metrics['rewards'].append(_safe_mean(episode_rewards))
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        log_progress(
            f"{name} - Episode {episode + 1}/{num_episodes} done. "
            f"Avg Reward: {metrics['rewards'][-1]:.4f}, Speed: {avg_speed:.2f}, "
            f"Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
        )
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, None)
    env.close()
    return {name: metrics}


def run_max_pressure(num_episodes=10, resume_state=None, on_episode_complete=None):
    """Evaluate deterministic max-pressure control without learning."""
    return _run_deterministic_controller(MaxPressureController, 'Max-Pressure', num_episodes, resume_state, on_episode_complete)


def run_fixed_time(num_episodes=10, resume_state=None, on_episode_complete=None):
    controller = FixedTimeController(
        green_duration=ENV_PARAMS['delta_rs_update_time'],
        yellow_time=ENV_PARAMS['yellow_time'],
        delta_rs_update_time=ENV_PARAMS['delta_rs_update_time'],
    )
    return _run_deterministic_controller(controller, FIXED_TIME_NAME, num_episodes, resume_state, on_episode_complete)


def _update_training_progress(
    progress_ctx,
    algorithm_name,
    episode_idx,
    num_episodes,
    train_reward,
    train_floor,
    eval_reward,
    eval_floor,
    extra_state=None,
    force=False,
):
    if not progress_ctx:
        return

    run_state_path = progress_ctx.get('run_state_path')
    run_log_path = progress_ctx.get('run_log_path')
    try:
        progress_every = int(progress_ctx.get('progress_every') or 10)
    except Exception:
        progress_every = 10

    should_log = bool(force) or (episode_idx == 1) or (episode_idx % progress_every == 0) or (episode_idx == num_episodes)
    if not should_log:
        return

    try:
        state = {}
        if run_state_path and os.path.exists(run_state_path):
            with open(run_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        state['current_algorithm'] = algorithm_name
        state['current_episode'] = int(episode_idx)
        state['current_total_episodes'] = int(num_episodes)
        if train_reward is not None:
            state['current_train_reward'] = _serialize_metric_value(train_reward)
        if train_floor is not None:
            state['current_train_floor_p20w10'] = _serialize_metric_value(train_floor)
        if eval_reward is not None:
            state['current_eval_reward'] = _serialize_metric_value(eval_reward)
        if eval_floor is not None:
            state['current_eval_floor_p20w10'] = _serialize_metric_value(eval_floor)
        if extra_state:
            for key, value in extra_state.items():
                state[key] = _serialize_metric_value(value)
        state['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if run_state_path:
            with open(run_state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

    try:
        if run_log_path:
            extra_suffix = ""
            if extra_state:
                extra_bits = [f"{k}={_serialize_metric_value(v)}" for k, v in extra_state.items()]
                extra_suffix = " " + " ".join(extra_bits)
            append_run_log(
                run_log_path,
                f"进度: {algorithm_name} {episode_idx}/{num_episodes} "
                f"train_reward={_serialize_metric_value(train_reward)} "
                f"train_floor_p20w10={_serialize_metric_value(train_floor)} "
                f"eval_floor_p20w10={_serialize_metric_value(eval_floor)}"
                f"{extra_suffix}"
            )
    except Exception:
        pass


def run_single_agent(env, agent, replay_buffer, num_episodes, algo_name, checkpoint_path=None,
                    resume_state=None, on_episode_complete=None):
    """通用单智能体或独立多智能体运行函数"""
    metrics = (resume_state or {}).get('metrics', {
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': []
    })
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        replay_state = resume_state.get('replay_buffer') or {}
        replay_buffer.memory = replay_state.get('memory', replay_buffer.memory)
        replay_buffer.steps_done = int(replay_state.get('steps_done', replay_buffer.steps_done))
        _restore_rng_state(resume_state.get('rng_state'))
    loop_error = None
    
    try:
        for episode in range(start_episode, num_episodes):
            episode_rewards = []
            episode_speeds = []
            episode_queue_lengths = []
            episode_waiting_times = []
            last_metric_sample_time = None
            
            env.reset()
            num_agents = env.num_agents
            is_multi = env.multi_agent
            
            done = False
            invalid_action = [False] * num_agents
            
            print(f"{algo_name} - Episode {episode+1}/{num_episodes}")
            sys.stdout.flush()
            
            step_count = 0
            while not done:
                step_count += 1
                if step_count % 50 == 0:
                    print(".", end="", flush=True)
                
                try:
                    if is_multi:
                        states = env._compute_states()
                        actions = []
                        for i in range(num_agents):
                            action = agent.select_action(states[i], replay_buffer.steps_done, invalid_action[i])
                            actions.append(int(action))
                        
                        next_states, rewards, done, info = env.step(actions)
                        
                        _append_effective_multi_rewards(
                            episode_rewards,
                            rewards,
                            info.get('do_actions', []),
                            next_states,
                            done,
                        )
                        
                        for i in range(num_agents):
                            if info['do_actions'][i] is not None and info['do_actions'][i] >= 0 and next_states[i] is not None:
                                replay_buffer.add(states[i], next_states[i], rewards[i], info['do_actions'][i], done)
                                invalid_action[i] = False
                            elif next_states[i] is None and done:
                                # 仿真结束，使用当前状态作为 next_state 存入
                                if info['do_actions'][i] is not None and info['do_actions'][i] >= 0:
                                    replay_buffer.add(states[i], states[i], rewards[i], info['do_actions'][i], done)
                                    invalid_action[i] = False
                                else:
                                    invalid_action[i] = True
                            else:
                                invalid_action[i] = True
                    else:
                        state = env._compute_state()
                        action = agent.select_action(state, replay_buffer.steps_done, invalid_action[0])
                        next_state, reward, done, info = env.step(int(action))
                        
                        _append_effective_single_reward(
                            episode_rewards,
                            reward,
                            info.get('do_action'),
                            next_state,
                            done,
                        )
                        
                        if info['do_action'] is not None and info['do_action'] >= 0 and next_state is not None:
                            replay_buffer.add(state, next_state, reward, info['do_action'], done)
                            invalid_action[0] = False
                        elif next_state is None and done:
                            if info['do_action'] is not None and info['do_action'] >= 0:
                                replay_buffer.add(state, state, reward, info['do_action'], done)
                                invalid_action[0] = False
                            else:
                                invalid_action[0] = True
                        else:
                            invalid_action[0] = True
                    
                    if len(replay_buffer) >= agent.batch_size:
                        agent.optimize_model()
                        if hasattr(agent, 'update_target_network'):
                            agent.update_target_network(replay_buffer.steps_done)
                    
                    try:
                        last_metric_sample_time = _maybe_record_metrics(
                            env,
                            episode_speeds,
                            episode_queue_lengths,
                            episode_waiting_times,
                            last_metric_sample_time,
                            done,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Error in episode {episode+1} loop: {e}")
                    import traceback
                    traceback.print_exc()
                    loop_error = RuntimeError(f"{algo_name} failed at episode {episode+1}: {e}")
                    done = True
                    break

            if loop_error is not None:
                break
            
            avg_reward = _safe_mean(episode_rewards)
            avg_speed, avg_queue, avg_waiting = _episode_metric_means(
                episode_speeds, episode_queue_lengths, episode_waiting_times
            )
            
            metrics['rewards'].append(avg_reward)
            metrics['speeds'].append(avg_speed)
            metrics['queue_lengths'].append(avg_queue)
            metrics['waiting_times'].append(avg_waiting)
            log_progress(
                f"{algo_name} - Episode {episode+1} done. Avg Reward: {avg_reward:.2f}, "
                f"Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
            )
            if on_episode_complete:
                on_episode_complete(episode + 1, metrics, agent, replay_buffer)
    finally:
        env.close()

    if loop_error is not None:
        raise loop_error

    if checkpoint_path:
        agent.save(checkpoint_path)

    print(f"[{algo_name}] Completed.")
    return metrics

def run_reinforce_agent(env, agent, num_episodes, algo_name, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行REINFORCE算法的专用函数 (支持独立多智能体)"""
    metrics = (resume_state or {}).get('metrics', {
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': []
    })
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        _restore_rng_state(resume_state.get('rng_state'))
    
    for episode in range(start_episode, num_episodes):
        episode_rewards = []
        episode_speeds = []
        episode_queue_lengths = []
        episode_waiting_times = []
        last_metric_sample_time = None
        
        env.reset()
        num_agents = env.num_agents
        is_multi = env.multi_agent
        
        done = False
        
        print(f"{algo_name} - Episode {episode+1}/{num_episodes}")
        
        while not done:
            if is_multi:
                states = env._compute_states()
                actions = [int(agent.select_action(s)) for s in states]
                next_states, rewards, done, info = env.step(actions)
                
                _append_effective_multi_rewards(
                    episode_rewards,
                    rewards,
                    info.get('do_actions', []),
                    next_states,
                    done,
                )
                
                for i in range(num_agents):
                    agent.store_transition(states[i], actions[i], rewards[i], done)
            else:
                state = env._compute_state()
                action = int(agent.select_action(state))
                next_state, reward, done, info = env.step(action)
                
                _append_effective_single_reward(
                    episode_rewards,
                    reward,
                    info.get('do_action'),
                    next_state,
                    done,
                )
                if reward is not None:
                    agent.store_transition(state, action, reward, done)

            try:
                last_metric_sample_time = _maybe_record_metrics(
                    env,
                    episode_speeds,
                    episode_queue_lengths,
                    episode_waiting_times,
                    last_metric_sample_time,
                    done,
                )
            except Exception:
                pass

        agent.optimize_model()
        
        avg_reward = _safe_mean(episode_rewards)
        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queue_lengths, episode_waiting_times
        )
        
        metrics['rewards'].append(avg_reward)
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        log_progress(
            f"{algo_name} - Episode {episode+1} done. Avg Reward: {avg_reward:.2f}, "
            f"Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
        )
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, agent)

    env.close()
    if checkpoint_path:
        agent.save(checkpoint_path)
    print(f"[{algo_name}] Completed.")
    return metrics

def run_shared_policy_gae_agent(env, agent, num_episodes, algo_name, checkpoint_path=None, resume_state=None, on_episode_complete=None):
    """运行共享策略 GAE Actor-Critic 的专用函数。"""
    metrics = (resume_state or {}).get('metrics', {
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': []
    })
    start_episode = int((resume_state or {}).get('episode', 0))
    if resume_state:
        agent.load_state_dict(resume_state['agent'])
        _restore_rng_state(resume_state.get('rng_state'))
    
    for episode in range(start_episode, num_episodes):
        episode_rewards = []
        episode_speeds = []
        episode_queue_lengths = []
        episode_waiting_times = []
        last_metric_sample_time = None
        
        env.reset()
        num_agents = env.num_agents
        is_multi = env.multi_agent
        
        done = False
        
        print(f"{algo_name} - Episode {episode+1}/{num_episodes}")
        
        while not done:
            if is_multi:
                states = env._compute_states()
                actions = [int(agent.select_action(s)) for s in states]
                next_states, rewards, done, info = env.step(actions)
                
                _append_effective_multi_rewards(
                    episode_rewards,
                    rewards,
                    info.get('do_actions', []),
                    next_states,
                    done,
                )
                
                for i in range(num_agents):
                    if next_states[i] is not None and info['do_actions'][i] is not None and info['do_actions'][i] >= 0:
                        agent.store_transition(states[i], actions[i], next_states[i], rewards[i], done)
                    elif done and info['do_actions'][i] is not None and info['do_actions'][i] >= 0:
                        agent.store_transition(states[i], actions[i], states[i], rewards[i], done)
            else:
                state = env._compute_state()
                action = int(agent.select_action(state))
                next_state, reward, done, info = env.step(action)
                
                _append_effective_single_reward(
                    episode_rewards,
                    reward,
                    info.get('do_action'),
                    next_state,
                    done,
                )
                if next_state is not None and info['do_action'] is not None and info['do_action'] >= 0:
                    agent.store_transition(state, action, next_state, reward, done)
                elif done and info['do_action'] is not None and info['do_action'] >= 0:
                    agent.store_transition(state, action, state, reward, done)
            
            try:
                last_metric_sample_time = _maybe_record_metrics(
                    env,
                    episode_speeds,
                    episode_queue_lengths,
                    episode_waiting_times,
                    last_metric_sample_time,
                    done,
                )
            except Exception:
                pass

        agent.optimize_model()
        avg_reward = _safe_mean(episode_rewards)
        avg_speed, avg_queue, avg_waiting = _episode_metric_means(
            episode_speeds, episode_queue_lengths, episode_waiting_times
        )
        
        metrics['rewards'].append(avg_reward)
        metrics['speeds'].append(avg_speed)
        metrics['queue_lengths'].append(avg_queue)
        metrics['waiting_times'].append(avg_waiting)
        log_progress(
            f"{algo_name} - Episode {episode+1} done. Avg Reward: {avg_reward:.2f}, "
            f"Speed: {avg_speed:.2f}, Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}"
        )
    
        if on_episode_complete:
            on_episode_complete(episode + 1, metrics, agent)

    env.close()
    if checkpoint_path:
        agent.save(checkpoint_path)
    print(f"[{algo_name}] Completed.")
    return metrics

def smooth_data(data, weight=0.95):
    """指数移动平均平滑处理"""
    if not data:
        return []
    last = data[0]
    smoothed = []
    for point in data:
        if np.isnan(point):
            smoothed.append(np.nan)
            continue
        if np.isnan(last):
            last = point
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metrics(all_metrics=None, output_dir='./results'):
    """绘制所有算法的性能指标比较图（增强专业版）"""
    print("Plotting metrics with enhanced style...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载历史数据以进行全面对比
    metrics_to_plot = {}
    json_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metrics_to_plot = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics.json for plotting: {e}")
            
    # 合并当前运行的数据
    if all_metrics:
        for algo, data in all_metrics.items():
            metrics_to_plot[algo] = data
            
    if not metrics_to_plot:
        print("No metrics to plot.")
        return

    algorithms = list(metrics_to_plot.keys())
    # 过滤掉没有任何数据的算法（防止全 NaN 导致绘图异常）
    valid_algorithms = []
    for algo in algorithms:
        rewards = metrics_to_plot[algo]['rewards']
        if any(x is not None and not np.isnan(float(x)) for x in rewards):
            valid_algorithms.append(algo)
    
    if not valid_algorithms:
        print("No valid metrics with data to plot.")
        return
        
    algorithms = valid_algorithms
    max_len = max(len(metrics_to_plot[algo]['rewards']) for algo in algorithms)
    
    # 使用更专业的配色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    metric_specs = [
        ('rewards', 'Training Reward (Smoothed)', 'Reward'),
        ('speeds', 'Average Speed (m/s)', 'Speed (m/s)'),
        ('queue_lengths', 'Average Queue Length (m)', 'Queue Length (m)'),
        ('waiting_times', 'Average Waiting Time (s)', 'Waiting Time (s)'),
    ]

    episodes = list(range(1, max_len + 1))

    for ax, (metric_key, title, ylabel) in zip(axes.flatten(), metric_specs):
        has_data = False
        for i, name in enumerate(algorithms):
            display_name = name
            raw_data = metrics_to_plot[name][metric_key]
            data = [float(x) if x is not None else np.nan for x in raw_data]
            if all(np.isnan(x) for x in data):
                continue
            
            has_data = True
            # 对数据进行平滑处理
            smoothed_values = smooth_data(data, weight=0.95 if max_len >= 100 else (0.90 if max_len >= 50 else 0.80))
            
            linewidth = 1.8
            alpha = 0.7
            
            # 添加标记以便在数据点较少时也能看到
            marker = 'o' if max_len < 20 else None
            markersize = 4 if max_len < 20 else 0
            
            ax.plot(
                episodes[:len(smoothed_values)],
                smoothed_values,
                label=display_name,
                color=colors[i],
                linewidth=linewidth,
                alpha=alpha,
                marker=marker,
                markersize=markersize
            )

        ax.set_title(title, fontsize=16, fontweight='bold')
        # 自动调整 X 轴刻度
        if max_len > 20:
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        elif max_len > 0:
            ax.set_xticks(episodes)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.6)
        if has_data:
            ax.legend(loc='best', fontsize=10)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.suptitle(
        f"Multi-Agent Traffic Signal Control: Algorithm Performance Comparison ({generated_at})",
        fontsize=22,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"final_comparison_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Enhanced plot saved as '{filepath}'")
    plt.close(fig)

def save_metrics_to_file(metrics_dict, output_dir='./results', filename='metrics.json'):
    """保存所有算法的metrics到JSON文件，支持增量更新"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    existing_metrics = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_metrics = json.load(f)
        except Exception:
            pass
            
    # 合并新旧指标
    for algo, metrics in metrics_dict.items():
        existing_metrics[algo] = metrics
        
    print(f"Saving metrics to {filepath}...")
    try:
        serialized_metrics = _serialize_metrics(existing_metrics)
        with open(filepath, 'w') as f:
            json.dump(serialized_metrics, f, indent=4)
        print(f"Metrics saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")
        import traceback
        traceback.print_exc()

def save_metrics_to_csv(metrics_dict, timestamp, output_dir='./results'):
    """将每个算法的指标保存到单独的CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    for algorithm, metrics in metrics_dict.items():
        algorithm_name = algorithm.replace(' ', '_')
        filename = f"{algorithm_name}_metrics_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        if not isinstance(metrics, dict) or 'rewards' not in metrics:
            print(f"Skipping malformed metrics for {algorithm}: missing rewards")
            continue
        keys = ['episode'] + list(metrics.keys())
        num_episodes = len(metrics['rewards'])
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            for episode in range(num_episodes):
                row = [episode + 1]
                for key in metrics.keys():
                    series = metrics.get(key)
                    if isinstance(series, list):
                        row.append(series[episode] if episode < len(series) else float('nan'))
                    else:
                        row.append(series)
                writer.writerow(row)
        
        print(f"{algorithm} metrics saved to {filepath}")

def _normalize_training_metrics(metrics):
    """Attach one canonical training-return field without changing legacy curves."""
    if not isinstance(metrics, dict):
        return metrics
    rewards = metrics.get('rewards')
    if isinstance(rewards, list):
        metrics.setdefault('training_return', list(rewards))
    return metrics


def run_algorithm_wrapper(algorithm_name, num_episodes, resume_state=None, on_episode_complete=None):
    """Run one algorithm implemented in the local agents package."""
    log_progress(f"[{algorithm_name}] Starting with {num_episodes} episodes.")
    runners = {
        'DPG-MAPPO': run_dpg_mappo,
        PARAMETER_SHARED_DDQN_NAME: run_double_dqn,
        PARAMETER_SHARED_DUELING_DDQN_NAME: run_dueling_dqn,
        'GC-MAPPO': run_gc_mappo,
        'GC-MAPPO-ind-noP': run_gc_mappo_ind_no_prior,
        'GC-MAPPO-ind-P': run_gc_mappo_ind_prior,
        'MAPPO': run_mappo,
        'IPPO': run_ippo,
        'IPPO-P': run_ippo_prior,
        'IPPO+': run_ippo_plus,
        'IPPO-Cap': run_ippo_cap,
        'BR-IPPO': run_br_ippo,
        'GR-GMAPPO-L': run_gr_gmappo_l,
        GRAPH_ATTENTION_DDQN_NAME: run_graph_attention_double_dqn,
        PRESSURE_STATE_DDQN_NAME: run_pressure_state_double_dqn,
        COLIGHT_NAME: run_colight,
        PRESSLIGHT_NAME: run_presslight,
        'Max-Pressure': run_max_pressure,
        FIXED_TIME_NAME: run_fixed_time,
        SHARED_POLICY_REINFORCE_NAME: run_reinforce,
        SHARED_POLICY_GAE_NAME: run_shared_policy_gae_actor_critic,
    }
    try:
        runner = runners[algorithm_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}") from exc
    result = runner(
        num_episodes=num_episodes,
        resume_state=resume_state,
        on_episode_complete=on_episode_complete,
    )
    if isinstance(result, dict) and 'rewards' not in result:
        nested_result = result.get(algorithm_name)
        if isinstance(nested_result, dict):
            result = nested_result
    return algorithm_name, _normalize_training_metrics(result)

def _serialize_metric_value(value):
    """将指标值序列化为适合 JSON/CSV 的格式。"""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return numeric_value


def _serialize_metrics(metrics_dict):
    """将所有算法指标转换为可落盘格式。"""
    serialized_metrics = {}
    for algorithm, metrics in metrics_dict.items():
        serialized_metrics[algorithm] = {}
        for metric_name, values in metrics.items():
            serialized_metrics[algorithm][metric_name] = [
                _serialize_metric_value(v) for v in values
            ]
    return serialized_metrics


def _collect_reward_component_summary(env):
    # #region debug-point B:reward-component-summary
    signals = getattr(env, 'traffic_signals', None) or [getattr(env, 'traffic_signal', None)]
    component_keys = [
        'reward',
        'avg_speed',
        'avg_queue_length',
        'avg_waiting_time',
        'action_reward',
        'linear_score',
        'delta_score',
        'balanced_state_score',
        'aligned_progress_bonus',
        'aligned_speed_bonus',
        'overload_penalty',
        'barrier_penalty',
        'anti_counterexample_penalty',
        'regression_penalty',
        'speed_gate',
        'queue_gate',
        'waiting_gate',
        'congestion_gate',
        'joint_gate',
        'weakest_gate',
        'speed_alignment_gate',
    ]
    summary = {'signal_count': 0}
    component_values = {key: [] for key in component_keys}
    do_actions = []
    best_actions = []
    for ts in signals:
        if ts is None:
            continue
        components = getattr(ts, 'last_reward_components', None) or {}
        if not components:
            continue
        summary['signal_count'] += 1
        for key in component_keys:
            value = components.get(key)
            if value is not None:
                component_values[key].append(float(value))
        do_action = components.get('do_action')
        if do_action is not None:
            do_actions.append(int(do_action))
        best_action = components.get('best_action')
        if best_action is not None:
            best_actions.append(int(best_action))
    for key, values in component_values.items():
        if values:
            summary[key] = float(np.mean(values))
    if do_actions:
        summary['do_actions_preview'] = do_actions[:8]
    if best_actions:
        summary['best_actions_preview'] = best_actions[:8]
    return summary
    # #endregion


def append_run_log(log_path, message):
    """追加写入运行日志，保留每次算法完成的时间点。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


ACTIVE_RUN_LOG_PATH = None


def log_progress(message):
    """Print progress immediately and mirror timestamped milestones to the run log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
    if ACTIVE_RUN_LOG_PATH:
        append_run_log(ACTIVE_RUN_LOG_PATH, message)


class TimestampedLogMirror:
    """将标准输出按行镜像到运行日志，便于实时追踪训练进度。"""

    def __init__(self, stream, log_path):
        self._stream = stream
        self._log_path = log_path
        self._buffer = ''

    def write(self, data):
        if not data:
            return 0
        written = self._stream.write(data)
        self._buffer += data
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            text = line.rstrip('\r')
            if text.strip():
                append_run_log(self._log_path, text)
        return written

    def flush(self):
        self._stream.flush()
        if self._buffer.strip():
            append_run_log(self._log_path, self._buffer.rstrip('\r'))
            self._buffer = ''

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False


def save_summary_to_csv(metrics_dict, timestamp, output_dir='./results'):
    """保存当前已完成算法的汇总指标。"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'algo_summary_{timestamp}.csv')
    metrics_to_summarize = {}
    json_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metrics_to_summarize = json.load(f) or {}
        except Exception:
            metrics_to_summarize = {}
    if metrics_dict:
        for algorithm, metrics in metrics_dict.items():
            metrics_to_summarize[algorithm] = metrics
    if not metrics_to_summarize:
        metrics_to_summarize = metrics_dict or {}

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'algorithm',
            'avg_reward',
            'avg_speed',
            'avg_queue_length',
            'avg_waiting_time',
        ])
        for algorithm, metrics in metrics_to_summarize.items():
            writer.writerow([
                algorithm,
                _serialize_metric_value(_safe_mean(metrics['rewards'])),
                _serialize_metric_value(_safe_mean(metrics['speeds'])),
                _serialize_metric_value(_safe_mean(metrics['queue_lengths'])),
                _serialize_metric_value(_safe_mean(metrics['waiting_times'])),
            ])

    print(f"Summary saved to {filepath}")


def save_reward_analysis_to_csv(metrics_dict, timestamp, output_dir='./results'):
    """保存 reward 趋势与业务指标一致性分析，作为固定验收项。"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'reward_analysis_{timestamp}.csv')
    metrics_to_analyze = {}
    json_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metrics_to_analyze = json.load(f) or {}
        except Exception:
            metrics_to_analyze = {}
    if metrics_dict:
        for algorithm, metrics in metrics_dict.items():
            metrics_to_analyze[algorithm] = metrics
    if not metrics_to_analyze:
        metrics_to_analyze = metrics_dict or {}

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'algorithm',
            'reward_trend_slope',
            'reward_first20_mean',
            'reward_last20_mean',
            'reward_last20_delta',
            'corr_reward_speed',
            'corr_reward_queue_length',
            'corr_reward_waiting_time',
            'reward_trend_pass',
            'reward_metric_alignment_pass',
        ])
        for algorithm, metrics in metrics_to_analyze.items():
            analysis = _compute_reward_acceptance(metrics)
            writer.writerow([
                algorithm,
                _serialize_metric_value(analysis['reward_trend_slope']),
                _serialize_metric_value(analysis['reward_first20_mean']),
                _serialize_metric_value(analysis['reward_last20_mean']),
                _serialize_metric_value(analysis['reward_last20_delta']),
                _serialize_metric_value(analysis['corr_reward_speed']),
                _serialize_metric_value(analysis['corr_reward_queue_length']),
                _serialize_metric_value(analysis['corr_reward_waiting_time']),
                analysis['reward_trend_pass'],
                analysis['reward_metric_alignment_pass'],
            ])

    print(f"Reward analysis saved to {filepath}")


def save_result_analysis(metrics_dict, run_id, output_dir='./results'):
    """Write a concise, reproducible analysis of the completed comparison."""
    snapshots = {
        name: _build_metric_snapshot(metrics)
        for name, metrics in metrics_dict.items()
    }
    analyses = {
        name: _compute_reward_acceptance(metrics)
        for name, metrics in metrics_dict.items()
    }

    def _rank(metric_name, reverse):
        return sorted(
            snapshots,
            key=lambda name: snapshots[name][metric_name],
            reverse=reverse,
        )

    reward_rank = _rank('avg_reward', True)
    speed_rank = _rank('avg_speed', True)
    queue_rank = _rank('avg_queue_length', False)
    waiting_rank = _rank('avg_waiting_time', False)
    path = os.path.join(output_dir, f'analysis_{run_id}.md')

    with open(path, 'w', encoding='utf-8') as file:
        file.write(f'# Agent Algorithm Comparison Analysis ({run_id})\n\n')
        file.write('All values use training episodes under the same scenario.\n\n')
        file.write(f'- Best reward: {reward_rank[0]} ({snapshots[reward_rank[0]]["avg_reward"]:.6f})\n')
        file.write(f'- Best speed: {speed_rank[0]} ({snapshots[speed_rank[0]]["avg_speed"]:.4f} m/s)\n')
        file.write(f'- Lowest queue: {queue_rank[0]} ({snapshots[queue_rank[0]]["avg_queue_length"]:.4f})\n')
        file.write(f'- Lowest waiting time: {waiting_rank[0]} ({snapshots[waiting_rank[0]]["avg_waiting_time"]:.4f} s)\n\n')
        file.write('| Algorithm | Reward | Speed (m/s) | Queue | Waiting (s) | Trend | Alignment |\n')
        file.write('| --- | ---: | ---: | ---: | ---: | --- | --- |\n')
        for name in reward_rank:
            snapshot = snapshots[name]
            analysis = analyses[name]
            file.write(
                f'| {name} | {snapshot["avg_reward"]:.6f} | '
                f'{snapshot["avg_speed"]:.4f} | {snapshot["avg_queue_length"]:.4f} | '
                f'{snapshot["avg_waiting_time"]:.4f} | '
                f'{analysis["reward_trend_pass"]} | '
                f'{analysis["reward_metric_alignment_pass"]} |\n'
            )

    print(f"Result analysis saved to {path}")
    return path


def save_run_state(run_id, run_info, output_dir='./results'):
    """保存本次运行的状态快照，便于中断后追踪最近完成时间。"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'run_state_{run_id}.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)
    return filepath


def persist_run_outputs(all_metrics, run_id, output_dir='./results', plot=True):
    """Persist completed algorithm metrics and optionally refresh the figure."""
    if not all_metrics:
        return
    save_metrics_to_file(all_metrics, output_dir=output_dir)
    save_metrics_to_file(all_metrics, output_dir=output_dir, filename=f'metrics_{run_id}.json')
    save_metrics_to_csv(all_metrics, run_id, output_dir=output_dir)
    save_summary_to_csv(all_metrics, run_id, output_dir=output_dir)
    save_reward_analysis_to_csv(all_metrics, run_id, output_dir=output_dir)
    if plot:
        plot_metrics(all_metrics, output_dir=output_dir)


def update_run_state(run_info, all_metrics, algorithms_to_run, output_dir='./results', final_status=None):
    """更新运行状态文件，并写入当前已完成算法列表。"""
    completed_algorithms = list(all_metrics.keys())
    failed_algorithms = list((run_info.get('failed_algorithms') or {}).keys())
    run_info['completed_algorithms'] = completed_algorithms
    run_info['completed_count'] = len(completed_algorithms)
    run_info['all_metrics'] = _serialize_metrics(all_metrics)
    run_info['algorithm_reviews'] = {
        algorithm_name: _build_algorithm_review(
            algorithm_name,
            metrics,
            run_info,
            output_dir=output_dir,
        )
        for algorithm_name, metrics in all_metrics.items()
    }
    if completed_algorithms:
        run_info['last_completed_algorithm'] = completed_algorithms[-1]
        run_info['last_completed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if final_status is not None:
        run_info['status'] = final_status
        run_info['finished_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_info['pending_algorithms'] = [
        algo
        for algo in algorithms_to_run
        if algo not in completed_algorithms and algo not in failed_algorithms
    ]
    save_run_state(run_info['run_id'], run_info, output_dir=output_dir)

def print_summary(all_metrics):
    """打印所有算法的性能汇总"""
    print("\n" + "="*80)
    print("算法性能对比汇总")
    print("="*80)
    
    headers = ["算法", "平均奖励", "平均速度", "平均排队长度", "平均等待时间"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<15} {headers[4]:<15}")
    print("-"*80)
    
    valid_metrics = {}
    for name, metrics in all_metrics.items():
        if not isinstance(metrics, dict) or 'rewards' not in metrics:
            print(f"{name:<15} 失败：指标缺少 rewards，已从汇总中跳过")
            continue
        avg_reward = _safe_mean(metrics['rewards'])
        avg_speed = _safe_mean(metrics['speeds'])
        avg_queue = _safe_mean(metrics['queue_lengths'])
        avg_waiting = _safe_mean(metrics['waiting_times'])
        valid_metrics[name] = metrics
        
        print(f"{name:<15} {avg_reward:<12.2f} {avg_speed:<12.2f} {avg_queue:<15.2f} {avg_waiting:<15.2f}")
    
    print("="*80)
    if not valid_metrics:
        print("\n未匹配到包含完整指标的可运行算法。")
        return
    
    # 找出最优算法
    best_reward = max((_safe_mean(m['rewards']), n) for n, m in valid_metrics.items())
    best_speed = max((_safe_mean(m['speeds']), n) for n, m in valid_metrics.items())
    best_queue = min((_safe_mean(m['queue_lengths']), n) for n, m in valid_metrics.items())
    best_waiting = min((_safe_mean(m['waiting_times']), n) for n, m in valid_metrics.items())
    
    print(f"\n最优奖励: {best_reward[1]} ({best_reward[0]:.2f})")
    print(f"最优速度: {best_speed[1]} ({best_speed[0]:.2f} m/s)")
    print(f"最优排队: {best_queue[1]} ({best_queue[0]:.2f} m)")
    print(f"最优等待: {best_waiting[1]} ({best_waiting[0]:.2f} s)")


def _normalize_algorithm_token(name):
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


SUPPORTED_ALGORITHMS = (
    'DPG-MAPPO',
    'GC-MAPPO',
    'GC-MAPPO-ind-noP',
    'GC-MAPPO-ind-P',
    'MAPPO',
    'IPPO',
    'IPPO-P',
    'IPPO+',
    'IPPO-Cap',
    'BR-IPPO',
    'GR-GMAPPO-L',
    COLIGHT_NAME,
    PRESSLIGHT_NAME,
    'Max-Pressure',
    FIXED_TIME_NAME,
    PARAMETER_SHARED_DDQN_NAME,
    PARAMETER_SHARED_DUELING_DDQN_NAME,
    SHARED_POLICY_GAE_NAME,
    SHARED_POLICY_REINFORCE_NAME,
)


def _resolve_algorithms(requested_algorithms):
    aliases = {
        _normalize_algorithm_token(name): name
        for name in SUPPORTED_ALGORITHMS
    }

    resolved = []
    for name in requested_algorithms:
        canonical_name = aliases.get(_normalize_algorithm_token(name))
        if canonical_name is None:
            allowed = ', '.join(SUPPORTED_ALGORITHMS)
            raise ValueError(f"Unsupported algorithm '{name}'. Allowed: {allowed}")
        if canonical_name not in resolved:
            resolved.append(canonical_name)
    return resolved


def _load_reused_baseline_metrics(reuse_results_dir, expected_config, episodes):
    """Load a completed baseline package for comparison without retraining it."""
    source_dir = os.path.abspath(reuse_results_dir)
    if not os.path.isdir(source_dir):
        raise ValueError(f'Reuse results directory does not exist: {source_dir}')

    state_paths = [
        os.path.join(source_dir, name)
        for name in os.listdir(source_dir)
        if name.startswith('run_state_') and name.endswith('.json')
    ]
    if not state_paths:
        raise ValueError(f'No run_state_*.json found in reuse directory: {source_dir}')
    state_path = max(state_paths, key=os.path.getmtime)
    with open(state_path, 'r', encoding='utf-8') as file:
        source_state = json.load(file)

    if source_state.get('status') not in ('completed', 'completed_with_failures', 'completed_with_reused_baselines'):
        raise ValueError(f'Reuse source is not a completed run: {source_state.get("status")}')
    if int(source_state.get('episodes', -1)) != int(episodes):
        raise ValueError(
            f'Reuse source episode count ({source_state.get("episodes")}) does not match requested {episodes}.'
        )

    source_config = source_state.get('config') or {}
    config_keys = (
        'net_file', 'route_file', 'simulation_time', 'tls_min_links',
        'sumo_seed', 'seed', 'multi_agent',
    )
    mismatches = [
        key for key in config_keys
        if source_config.get(key) != expected_config.get(key)
    ]
    if mismatches:
        raise ValueError(
            'Reuse source configuration does not match this standard run: '
            + ', '.join(mismatches)
        )
    if not (source_state.get('standard_profile') or {}).get('standard_ok'):
        raise ValueError('Reuse source did not pass the Python 3.9 + CUDA standard check.')

    source_run_id = source_state.get('run_id')
    metric_paths = [
        os.path.join(source_dir, f'metrics_{source_run_id}.json') if source_run_id else None,
        os.path.join(source_dir, 'metrics.json'),
    ]
    source_metrics = None
    for metric_path in metric_paths:
        if metric_path and os.path.isfile(metric_path):
            with open(metric_path, 'r', encoding='utf-8') as file:
                source_metrics = json.load(file)
            break
    if not isinstance(source_metrics, dict):
        source_metrics = source_state.get('all_metrics')
    if not isinstance(source_metrics, dict) or not source_metrics:
        raise ValueError(f'No usable metrics found in reuse directory: {source_dir}')

    valid_metrics = {}
    for name, metrics in source_metrics.items():
        if not isinstance(metrics, dict):
            continue
        required_series = ('rewards', 'speeds', 'queue_lengths', 'waiting_times')
        if not all(isinstance(metrics.get(key), list) for key in required_series):
            continue
        if any(len(metrics[key]) != int(episodes) for key in required_series):
            continue
        valid_metrics[name] = metrics
    if not valid_metrics:
        raise ValueError('Reuse source has no complete metric series matching the requested episodes.')

    return valid_metrics, {
        'source_directory': source_dir,
        'source_state_path': state_path,
        'source_run_id': source_run_id,
        'source_status': source_state.get('status'),
        'source_started_at': source_state.get('started_at'),
        'source_finished_at': source_state.get('finished_at'),
        'source_requested_algorithms': source_state.get('requested_algorithms') or [],
        'source_standard_profile': source_state.get('standard_profile'),
        'validated_config_keys': list(config_keys),
    }


def _save_data_provenance(run_id, output_dir, executed_algorithms, reused_algorithms, reuse_source=None):
    """Record whether every plotted series was newly executed or reused."""
    provenance = {
        'run_id': run_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'executed_algorithms': list(executed_algorithms),
        'reused_algorithms': list(reused_algorithms),
        'series_origin': {
            name: {'origin': 'executed_this_run'}
            for name in executed_algorithms
        },
    }
    for name in reused_algorithms:
        provenance['series_origin'][name] = {
            'origin': 'reused_baseline',
            'source_run_id': (reuse_source or {}).get('source_run_id'),
            'source_directory': (reuse_source or {}).get('source_directory'),
        }
    if reuse_source:
        provenance['reuse_source'] = reuse_source

    path = os.path.join(output_dir, f'data_provenance_{run_id}.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(provenance, file, indent=4, ensure_ascii=False)
    print(f'Data provenance saved to {path}')
    return provenance


def main():
    global ACTIVE_RUN_LOG_PATH

    parser = argparse.ArgumentParser(
        description='Compare the algorithms implemented in the local agents package.'
    )
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--algorithms', nargs='+', default=list(SUPPORTED_ALGORITHMS))
    parser.add_argument('--net-file', default=ENV_PARAMS['net_file'])
    parser.add_argument('--route-file', default=ENV_PARAMS['route_file'])
    parser.add_argument('--simulation-time', type=int, default=ENV_PARAMS['simulation_time'])
    parser.add_argument('--tls-min-links', type=int, default=ENV_PARAMS['tls_min_links'])
    parser.add_argument('--sumo-seed', type=int, default=ENV_PARAMS['sumo_seed'])
    parser.add_argument('--output-dir')
    parser.add_argument('--run-id')
    parser.add_argument(
        '--acceptance-baseline-dir',
        help=(
            'Directory of the accepted run used for strict candidate retention. '
            'A candidate must not worsen any primary metric and must strictly improve at least three.'
        ),
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--reuse-results-dir',
        help=(
            'Completed standard-run result directory. Metrics for algorithms omitted from '
            '--algorithms are reused for the final comparison, without retraining them.'
        ),
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error('--episodes must be greater than zero')
    if args.simulation_time <= 0:
        parser.error('--simulation-time must be greater than zero')
    if args.tls_min_links <= 0:
        parser.error('--tls-min-links must be greater than zero')
    if args.acceptance_baseline_dir and not os.path.isdir(args.acceptance_baseline_dir):
        parser.error('--acceptance-baseline-dir must be an existing result directory')

    try:
        algorithms = _resolve_algorithms(args.algorithms)
    except ValueError as exc:
        parser.error(str(exc))

    _seed_everything(args.seed)

    ENV_PARAMS.update({
        'net_file': args.net_file,
        'route_file': args.route_file,
        'simulation_time': args.simulation_time,
        'tls_min_links': args.tls_min_links,
        'sumo_seed': args.sumo_seed,
        'multi_agent': True,
    })
    standard_profile = _enforce_execution_standard(
        args.net_file,
        args.route_file,
        args.tls_min_links,
        args.sumo_seed,
    )
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or os.path.join('./results', f'agents_50ep_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'run_{run_id}.log')
    all_metrics = {}
    failures = {}
    resume_state = None
    resume_checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(resume_checkpoint_dir, exist_ok=True)
    reuse_metrics = {}
    reuse_source = None
    if args.reuse_results_dir:
        expected_config = {
            'net_file': args.net_file,
            'route_file': args.route_file,
            'simulation_time': args.simulation_time,
            'tls_min_links': args.tls_min_links,
            'sumo_seed': args.sumo_seed,
            'seed': args.seed,
            'per_algorithm_seed_reset': True,
            'multi_agent': True,
        }
        try:
            reuse_metrics, reuse_source = _load_reused_baseline_metrics(
                args.reuse_results_dir,
                expected_config,
                args.episodes,
            )
        except ValueError as exc:
            parser.error(str(exc))
    run_info = {
        'run_id': run_id,
        'script': os.path.basename(__file__),
        'status': 'running',
        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'episodes': args.episodes,
        'requested_algorithms': algorithms,
        'failed_algorithms': failures,
        'config': {
            'net_file': args.net_file,
            'route_file': args.route_file,
            'simulation_time': args.simulation_time,
            'tls_min_links': args.tls_min_links,
            'sumo_seed': args.sumo_seed,
            'seed': args.seed,
            'multi_agent': True,
        },
        'standard_profile': standard_profile,
        'log_path': log_path,
        'acceptance_baseline_dir': (
            os.path.abspath(args.acceptance_baseline_dir)
            if args.acceptance_baseline_dir else None
        ),
    }
    if reuse_source:
        run_info['reuse_source'] = reuse_source
        run_info['reused_algorithms'] = [
            name for name in reuse_metrics
            if name not in algorithms
        ]
    existing_state_path = os.path.join(output_dir, f'run_state_{run_id}.json')
    previous_run_info = None
    if os.path.isfile(existing_state_path):
        with open(existing_state_path, 'r', encoding='utf-8') as state_file:
            previous_run_info = json.load(state_file)
        previous_config = previous_run_info.get('config') or {}
        if previous_run_info.get('episodes') != args.episodes or previous_config != run_info['config']:
            raise RuntimeError('Existing run directory belongs to a different episode budget or configuration.')
        run_info['started_at'] = previous_run_info.get('started_at', run_info['started_at'])
        saved_metrics = previous_run_info.get('all_metrics') or {}
        metrics_path = os.path.join(output_dir, 'metrics.json')
        if os.path.isfile(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as metrics_file:
                persisted_metrics = json.load(metrics_file)
            saved_metrics.update(persisted_metrics)
        for algorithm_name in algorithms:
            algorithm_metrics = saved_metrics.get(algorithm_name)
            if not isinstance(algorithm_metrics, dict):
                continue
            if all(
                isinstance(algorithm_metrics.get(key), list)
                and len(algorithm_metrics[key]) == args.episodes
                for key in ('rewards', 'speeds', 'queue_lengths', 'waiting_times')
            ):
                all_metrics[algorithm_name] = algorithm_metrics
    save_run_state(run_id, run_info, output_dir=output_dir)
    ACTIVE_RUN_LOG_PATH = log_path
    try:
        log_progress(f"Run {run_id} started: {len(algorithms)} algorithms x {args.episodes} episodes.")
        log_progress(f"Standard environment: Python {sys.version.split()[0]}, device={device}.")
        log_progress(f"Results directory: {os.path.abspath(output_dir)}")
        if reuse_source:
            reused_names = run_info['reused_algorithms']
            log_progress(
                f"Reuse mode: {len(reused_names)} baseline algorithms will be merged after this run "
                f"from {reuse_source['source_directory']}."
            )
        completed_on_disk = set()
        if previous_run_info and all_metrics:
            log_progress(f'Resuming run state: {len(all_metrics)} algorithms already complete.')
        for algorithm_name in algorithms:
            if algorithm_name in all_metrics:
                continue
            try:
                _seed_everything(args.seed)
                log_progress(
                    f"[{algorithm_name}] RNG reset to seed {args.seed} before initialization."
                )
                progress_path = os.path.join(
                    resume_checkpoint_dir, f'{_normalize_algorithm_token(algorithm_name)}.pt'
                )
                resume_state = _load_episode_checkpoint(progress_path)
                # The initial seed is reset above; a valid checkpoint restores
                # the exact RNG stream inside the runner before its next episode.
                if resume_state and resume_state.get('config') != run_info['config']:
                    raise RuntimeError(
                        f'Checkpoint configuration mismatch for {algorithm_name}; refusing resume.'
                    )
                if resume_state and resume_state.get('algorithm') != algorithm_name:
                    raise RuntimeError(
                        f'Checkpoint algorithm mismatch for {algorithm_name}; refusing resume.'
                    )
                if resume_state and resume_state.get('episode', 0) >= args.episodes:
                    all_metrics[algorithm_name] = resume_state['metrics']
                    os.remove(progress_path)
                    log_progress(f'[{algorithm_name}] Checkpoint already complete; skipping training.')
                    continue
                if resume_state:
                    log_progress(
                        f'[{algorithm_name}] Resuming from episode {resume_state["episode"]}/'
                        f'{args.episodes}.'
                    )

                def save_progress(episode, metrics, agent, replay_buffer=None):
                    checkpoint = {
                        'episode': episode,
                        'metrics': _serialize_metrics({algorithm_name: metrics})[algorithm_name],
                        'rng_state': _capture_rng_state(),
                        'agent': agent.state_dict() if agent is not None else None,
                        'config': run_info['config'],
                        'algorithm': algorithm_name,
                    }
                    if replay_buffer is not None:
                        checkpoint['replay_buffer'] = {
                            'memory': replay_buffer.memory,
                            'steps_done': replay_buffer.steps_done,
                        }
                    temp_path = f'{progress_path}.tmp'
                    torch.save(checkpoint, temp_path)
                    os.replace(temp_path, progress_path)
                    all_metrics[algorithm_name] = metrics
                    persist_run_outputs(all_metrics, run_id, output_dir=output_dir, plot=False)
                    run_info['current_algorithm'] = algorithm_name
                    run_info['current_episode'] = episode
                    run_info['completed_algorithms'] = list(all_metrics.keys())
                    save_run_state(run_id, run_info, output_dir=output_dir)

                name, metrics = run_algorithm_wrapper(
                    algorithm_name, args.episodes, resume_state=resume_state,
                    on_episode_complete=save_progress,
                )
                all_metrics[name] = metrics
                if os.path.isfile(progress_path):
                    os.remove(progress_path)
                persist_run_outputs(all_metrics, run_id, output_dir=output_dir, plot=False)
                update_run_state(run_info, all_metrics, algorithms, output_dir=output_dir)
                log_progress(f"[{name}] completed and persisted.")
            except Exception as exc:
                failures[algorithm_name] = repr(exc)
                print(f"[{algorithm_name}] failed: {exc}", file=sys.stderr)
                traceback.print_exc()
                append_run_log(log_path, f"[{algorithm_name}] failed: {exc}")
                update_run_state(run_info, all_metrics, algorithms, output_dir=output_dir)

        if not all_metrics:
            raise RuntimeError('No algorithms completed successfully.')

        comparison_metrics = dict(all_metrics)
        reused_algorithms = []
        if reuse_source:
            for algorithm_name in reuse_source.get('source_requested_algorithms', []):
                if algorithm_name not in algorithms and algorithm_name in reuse_metrics:
                    comparison_metrics.setdefault(algorithm_name, reuse_metrics[algorithm_name])
            for algorithm_name, metrics in reuse_metrics.items():
                if algorithm_name not in algorithms:
                    comparison_metrics.setdefault(algorithm_name, metrics)
            reused_algorithms = [
                name for name in comparison_metrics
                if name not in all_metrics
            ]
            log_progress(
                f"Merged {len(reused_algorithms)} reused baseline series; they are marked in data provenance."
            )

        persist_run_outputs(comparison_metrics, run_id, output_dir=output_dir, plot=True)
        analysis_path = save_result_analysis(comparison_metrics, run_id, output_dir=output_dir)
        _save_data_provenance(
            run_id,
            output_dir,
            all_metrics.keys(),
            reused_algorithms,
            reuse_source=reuse_source,
        )
        update_run_state(
            run_info,
            comparison_metrics,
            algorithms,
            output_dir=output_dir,
            final_status=(
                'completed_with_failures'
                if failures else ('completed_with_reused_baselines' if reused_algorithms else 'completed')
            ),
        )
        print_summary(comparison_metrics)
        _print_algorithm_reviews(run_info)
        log_progress(f"Run completed. Analysis: {os.path.abspath(analysis_path)}")

        if failures:
            failed_names = ', '.join(failures)
            raise RuntimeError(f'Comparison completed with failures: {failed_names}')
    finally:
        ACTIVE_RUN_LOG_PATH = None


if __name__ == '__main__':
    main()









