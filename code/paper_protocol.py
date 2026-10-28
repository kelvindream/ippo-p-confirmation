"""Run one auditable train/checkpoint/evaluate task for the repaired C168 protocol."""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

import test_algorithms as training
from agents.baselines_descriptive import (
    DESCRIPTIVE_METHODS,
    FixedTimeProgramController,
    MaxPressureDescriptiveController,
)
from environment.env import SumoEnv
from replay import ReplayBuffer


FORMAL_VARIANTS = {
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
    'GC-MAPPO-ind-noP': {
        'family': 'GC-MAPPO', 'actor_type': 'independent',
        'prior_mode': 'pressure_zscore_hold', 'action_prior_scale': 0.0,
        'pressure_observation': True, 'soft_q': True,
        'policy_temperature': 0.30, 'hold_prior_bonus': 0.0,
    },
    'GC-MAPPO-ind-P': {
        'family': 'GC-MAPPO', 'actor_type': 'independent',
        'prior_mode': 'pressure_zscore_hold', 'action_prior_scale': 2.25,
        'pressure_observation': True, 'soft_q': True,
        'policy_temperature': 0.30, 'hold_prior_bonus': 0.0,
    },
}

METHODS = (
    'DPG-MAPPO', 'GC-MAPPO', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P',
    'MAPPO', 'IPPO', 'IPPO-P', 'IPPO+', 'IPPO-Cap', 'BR-IPPO',
    'GR-GMAPPO-L', 'CoLight', 'PressLight', 'Max-Pressure', 'Fixed-Time',
    'Fixed-Time-Program', 'Max-Pressure-Descriptive',
    'Double DQN', 'Dueling Double DQN', 'GAE Actor-Critic', 'REINFORCE',
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure(train_seed, sumo_seed, net_file=None, route_file=None, tls_min_links=None):
    # Phase 2 P2-3 扩展：场景参数化以支持 H1（3x3-medium）/H2（2x2-oversaturated）评测；
    # 不传参时保持原 2x2-medium 默认行为不变（不破坏冻结协议）。
    training.ENV_PARAMS.update({
        'net_file': net_file or 'nets/2x2-grid/2x2.net.xml',
        'route_file': route_file or 'nets/2x2-grid/2x2-medium.rou.xml',
        'skip_range': 0,
        'simulation_time': 10000,
        'yellow_time': 2,
        'delta_rs_update_time': 10,
        'use_gui': False,
        'multi_agent': True,
        'tls_min_links': 4 if tls_min_links is None else int(tls_min_links),
        'sumo_seed': int(sumo_seed),
    })
    _seed_everything(train_seed)


def _environment(sumo_seed):
    return SumoEnv(**{**training.ENV_PARAMS, 'multi_agent': True, 'sumo_seed': int(sumo_seed)})


def _multi_agent_dimensions(sumo_seed):
    env = _environment(sumo_seed)
    try:
        env.reset()
        spaces = env.get_agent_spaces()
        return env.num_agents, spaces[0][0].shape[0], spaces[0][1].n, env.adjacency_matrix, env.distance_matrix
    finally:
        env.close()


def _build_gc_mappo(num_agents, raw_obs_dim, action_dim, adjacency, distance,
                    independent_actor=None, action_prior_scale=None,
                    soft_q_execution=None):
    return training.GCMAPPO(
        num_agents=num_agents,
        obs_dim=raw_obs_dim + action_dim,
        action_dim=action_dim,
        adjacency_matrix=adjacency,
        distance_matrix=distance,
        agent_embedding_dim=0,
        independent_actor=(
            os.environ.get('GC_MAPPO_INDEPENDENT_ACTOR', '0') == '1'
            if independent_actor is None else bool(independent_actor)
        ),
        action_prior_scale=float(
            os.environ.get('GC_MAPPO_ACTION_PRIOR_SCALE', '2.25')
            if action_prior_scale is None else action_prior_scale
        ),
        family='GC-MAPPO', prior_mode='pressure_zscore_hold',
        pressure_observation=True,
        relative_value_coef=0.10,
        action_value_head=True,
        action_value_correction_scale=0.10,
        action_value_consensus=True,
        soft_q_execution=(
            os.environ.get('GC_MAPPO_SOFT_Q_EXECUTION', '1') == '1'
            if soft_q_execution is None else bool(soft_q_execution)
        ),
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


def _build_evaluator(method, sumo_seed, checkpoint_path):
    if method in ('Max-Pressure', 'Fixed-Time') or method in DESCRIPTIVE_METHODS:
        return None
    num_agents, raw_obs_dim, action_dim, adjacency, distance = _multi_agent_dimensions(sumo_seed)
    if method == 'DPG-MAPPO':
        agent = training.DPGMAPPO(
            num_agents, raw_obs_dim + action_dim, action_dim,
            adjacency_matrix=adjacency, distance_matrix=distance,
            action_prior_scale=float(os.environ.get('GC_MAPPO_ACTION_PRIOR_SCALE', '2.25')),
            residual_critic=True, adaptive_residual_consensus=True,
            adaptive_residual_energy_mix=True,
        )
    elif method == 'GC-MAPPO':
        agent = _build_gc_mappo(num_agents, raw_obs_dim, action_dim, adjacency, distance)
    elif method in ('GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P'):
        variant = FORMAL_VARIANTS[method]
        agent = _build_gc_mappo(
            num_agents, raw_obs_dim, action_dim, adjacency, distance,
            independent_actor=True,
            action_prior_scale=variant['action_prior_scale'],
            soft_q_execution=variant['soft_q'],
        )
    elif method == 'MAPPO':
        agent = training.MAPPO(num_agents, raw_obs_dim, action_dim)
    elif method in ('IPPO', 'IPPO-P'):
        variant = FORMAL_VARIANTS[method]
        agent = training.IPPO(
            num_agents, raw_obs_dim, action_dim,
            action_prior_scale=variant['action_prior_scale'],
            prior_mode=variant['prior_mode'],
            policy_temperature=variant['policy_temperature'],
        )
    elif method == 'IPPO+':
        agent = training.IPPO(num_agents, raw_obs_dim * 2, action_dim)
    elif method in ('BR-IPPO', 'GR-GMAPPO-L'):
        initialization_checkpoint = os.environ.get('GC_MAPPO_IPPO_INIT_CHECKPOINT')
        if not initialization_checkpoint or not os.path.isfile(initialization_checkpoint):
            raise ValueError('Set GC_MAPPO_IPPO_INIT_CHECKPOINT to the matching-train-seed IPPO checkpoint.')
        agent = training.GatedGraphResidualIPPO(
            num_agents, raw_obs_dim, action_dim, adjacency,
            freeze_actor=(method == 'BR-IPPO'),
        )
        agent.initialize_from_ippo(initialization_checkpoint)
    elif method == 'IPPO-Cap':
        target_parameters = training.ippo_parameter_count(
            raw_obs_dim * 2, action_dim, 128, num_agents
        )
        hidden_dim = training.matched_hidden_dim(
            raw_obs_dim, action_dim, target_parameters, num_agents
        )
        agent = training.IPPO(num_agents, raw_obs_dim, action_dim, hidden_dim=hidden_dim)
    elif method in (training.COLIGHT_NAME, training.GRAPH_ATTENTION_DDQN_NAME):
        agent_cls = training.CoLight if method == training.COLIGHT_NAME else training.GraphAttentionDoubleDQN
        agent = agent_cls(num_agents, raw_obs_dim, action_dim, adjacency)
    elif method in (training.PRESSLIGHT_NAME, training.PRESSURE_STATE_DDQN_NAME):
        agent_cls = training.PressLight if method == training.PRESSLIGHT_NAME else training.PressureStateDoubleDQN
        agent = agent_cls(input_dim=action_dim, action_dim=action_dim)
    elif method == training.PARAMETER_SHARED_DDQN_NAME:
        agent = training.DoubleDQNAgent(
            mode='eval', replay_buffer=ReplayBuffer(capacity=1), target_update=3000,
            gamma=0.95, eps_start=1.0, eps_end=0.1, eps_decay=83000,
            input_dim=raw_obs_dim, output_dim=action_dim, batch_size=32,
            network_file=checkpoint_path,
        )
        return agent
    elif method == training.PARAMETER_SHARED_DUELING_DDQN_NAME:
        agent = training.DuelingDoubleDQNAgent(
            mode='eval', replay_buffer=ReplayBuffer(capacity=1), target_update=8000,
            gamma=0.92, eps_start=1.0, eps_end=0.24, eps_decay=120000,
            input_dim=raw_obs_dim, output_dim=action_dim, batch_size=128,
            learning_rate=0.00008, warmup_transitions=768, train_freq=4,
            network_file=checkpoint_path,
        )
        return agent
    elif method == training.SHARED_POLICY_GAE_NAME:
        agent = training.GAEActorCritic(
            input_dim=raw_obs_dim, output_dim=action_dim, mode='eval', gamma=0.99,
            learning_rate=0.0003, gae_lambda=0.95,
        )
    elif method == training.SHARED_POLICY_REINFORCE_NAME:
        agent = training.ReinforceAgent(
            input_dim=raw_obs_dim, output_dim=action_dim, mode='eval', gamma=0.99, lr=0.001,
        )
    else:
        raise ValueError(f'Unsupported method: {method}')
    agent.load(checkpoint_path)
    return agent


def _gc_actions(agent, env, states, directional=False, hold_prior_bonus=None,
                policy_temperature=None, deterministic=None, formal=False):
    raw_obs_dim = len(states[0])
    action_dim = len(env.traffic_signals[0].all_green_phases)
    local_states = np.asarray([
        training._normalize_state_dim(state, raw_obs_dim) for state in states
    ], dtype=np.float32)
    raw_pressure = np.asarray([signal.pressure_state() for signal in env.traffic_signals], dtype=np.float32)
    pressure_center = raw_pressure.mean(axis=1, keepdims=True)
    pressure_scale = raw_pressure.std(axis=1, keepdims=True)
    pressure_prior = np.clip(
        (raw_pressure - pressure_center) / np.maximum(pressure_scale, 1.0), -3.0, 3.0,
    ).astype(np.float32)
    observations = np.concatenate([local_states, pressure_prior], axis=1)
    decision_slots = np.asarray([
        signal.yellow_phase is None
        and env.sumo.simulation.getTime() >= signal.end_time
        for signal in env.traffic_signals
    ], dtype=bool)
    action_masks = np.ones((len(states), action_dim), dtype=bool)
    if os.environ.get('GC_MAPPO_PRESSURE_CONFIDENCE_MASK') == '1':
        action_masks, _ = training._pressure_confidence_action_masks(
            raw_pressure, decision_slots,
        )
    execution_prior = training._add_current_phase_hold_prior(
        pressure_prior,
        env,
        decision_slots,
        float(
            os.environ.get('GC_MAPPO_HOLD_PRIOR_BONUS', '0.0')
            if hold_prior_bonus is None else hold_prior_bonus
        ),
    )
    if not formal:
        prior_scale = float(os.environ.get('GC_MAPPO_EVAL_PRIOR_SCALE', '1.0'))
        if prior_scale != 1.0:
            execution_prior = execution_prior * prior_scale
    select_kwargs = {}
    if directional:
        select_kwargs['directional_features'] = training.directional_phase_features(
            env.traffic_signals
        )
    actions, _, _ = agent.select_actions(
        observations,
        action_masks=action_masks,
        action_priors=execution_prior,
        policy_temperature=float(
            os.environ.get('GC_MAPPO_EVAL_POLICY_TEMPERATURE', os.environ.get('GC_MAPPO_TEMP_END', '0.30'))
            if policy_temperature is None else policy_temperature
        ),
        deterministic=(
            os.environ.get('GC_MAPPO_EVAL_DETERMINISTIC', '1') == '1'
            if deterministic is None else bool(deterministic)
        ),
        **select_kwargs,
    )
    return actions


def _ippo_prior_actions(agent, env, states, hold_prior_bonus=0.0,
                        policy_temperature=1.0):
    """IPPO + 执行压力先验对照：先验构造与 _gc_actions 完全一致，
    仅将先验加到 IPPO 各 actor 的 logits 上（不修改策略观测输入）。"""
    raw_pressure = np.asarray([signal.pressure_state() for signal in env.traffic_signals], dtype=np.float32)
    pressure_center = raw_pressure.mean(axis=1, keepdims=True)
    pressure_scale = raw_pressure.std(axis=1, keepdims=True)
    pressure_prior = np.clip(
        (raw_pressure - pressure_center) / np.maximum(pressure_scale, 1.0), -3.0, 3.0,
    ).astype(np.float32)
    decision_slots = np.asarray([
        signal.yellow_phase is None
        and env.sumo.simulation.getTime() >= signal.end_time
        for signal in env.traffic_signals
    ], dtype=bool)
    execution_prior = training._add_current_phase_hold_prior(
        pressure_prior,
        env,
        decision_slots,
        float(hold_prior_bonus),
    )
    actions, _, _ = agent.select_actions(
        np.asarray(states, dtype=np.float32),
        deterministic=True,
        action_priors=execution_prior,
        policy_temperature=float(policy_temperature),
    )
    return actions


def _select_actions(method, agent, env, states, invalid_actions):
    if method == 'Max-Pressure':
        return training.MaxPressureController.select_actions(env.traffic_signals)
    if method == 'Fixed-Time':
        if not hasattr(env, '_fixed_time_controller'):
            env._fixed_time_controller = training.FixedTimeController(
                green_duration=training.ENV_PARAMS['delta_rs_update_time'],
                yellow_time=training.ENV_PARAMS['yellow_time'],
                delta_rs_update_time=training.ENV_PARAMS['delta_rs_update_time'],
            )
        return env._fixed_time_controller.select_actions(env.traffic_signals)
    if method == 'Max-Pressure-Descriptive':
        return MaxPressureDescriptiveController.select_actions(env.traffic_signals)
    if method == 'Fixed-Time-Program':
        if not hasattr(env, '_fixed_time_program_controller'):
            env._fixed_time_program_controller = FixedTimeProgramController()
        return env._fixed_time_program_controller.select_actions(env.traffic_signals)
    if method == 'DPG-MAPPO':
        return _gc_actions(agent, env, states, directional=True)
    if method == 'GC-MAPPO':
        return _gc_actions(agent, env, states)
    if method in ('GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P'):
        variant = FORMAL_VARIANTS[method]
        return _gc_actions(
            agent, env, states,
            hold_prior_bonus=variant['hold_prior_bonus'],
            policy_temperature=variant['policy_temperature'],
            deterministic=True, formal=True,
        )
    if method in ('BR-IPPO', 'GR-GMAPPO-L'):
        return agent.select_actions(np.asarray(states, dtype=np.float32), deterministic=True)[0]
    if method == 'IPPO+':
        observations = training._neighbor_augmented_observations(
            env, np.asarray(states, dtype=np.float32)
        )
        return agent.select_actions(observations, deterministic=True)[0]
    if method == 'IPPO-P':
        variant = FORMAL_VARIANTS[method]
        return _ippo_prior_actions(
            agent, env, states,
            hold_prior_bonus=variant['hold_prior_bonus'],
            policy_temperature=variant['policy_temperature'],
        )
    if method in ('MAPPO', 'IPPO', 'IPPO-Cap'):
        return agent.select_actions(
            np.asarray(states, dtype=np.float32), deterministic=True,
            **({'policy_temperature': 1.0} if method == 'IPPO' else {}),
        )[0]
    if method in (training.COLIGHT_NAME, training.GRAPH_ATTENTION_DDQN_NAME):
        return agent.select_actions(np.asarray(env._compute_states(), dtype=np.float32), deterministic=True)
    if method in (training.PRESSLIGHT_NAME, training.PRESSURE_STATE_DDQN_NAME):
        return agent.select_actions(np.asarray([signal.pressure_state() for signal in env.traffic_signals], dtype=np.float32), deterministic=True)
    if method in (training.PARAMETER_SHARED_DDQN_NAME, training.PARAMETER_SHARED_DUELING_DDQN_NAME):
        return [
            int(agent.select_action(np.asarray(state, dtype=np.float32), 0, invalid_actions[index]))
            for index, state in enumerate(states)
        ]
    return [int(agent.select_action(np.asarray(state, dtype=np.float32), 0)) for state in states]


def _evaluate(method, checkpoint_path, evaluation_seeds):
    metrics = {
        'evaluation_reward': [],
        'rewards': [],
        'speeds': [],
        'queue_lengths': [],
        'waiting_times': [],
        'travel_times': [],
        'demand_service_times': [],
        'delays': [],
        'throughput': [],
        'throughput_rates': [],
        'completion_rates': [],
        'planned_completion_rates': [],
        'normal_completed_vehicles': [],
        'planned_vehicles': [],
        'demand_conservation_ok': [],
        'incomplete_vehicles': [],
        'teleport_counts': [],
        'decision_slots': [],
        'executed_hold_rates': [],
        'executed_switch_rates': [],
        'yellow_advances_per_step': [],
        'yellow_advances_per_decision': [],
        'reward_components': [],
    }
    total_seeds = len(evaluation_seeds)
    for seed_idx, evaluation_seed in enumerate(evaluation_seeds, 1):
        _seed_everything(evaluation_seed)
        agent = _build_evaluator(method, evaluation_seed, checkpoint_path)
        _seed_everything(evaluation_seed)
        env = _environment(evaluation_seed)
        net_file = training.ENV_PARAMS.get('net_file', '?')
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"{method} eval seed {evaluation_seed} ({seed_idx}/{total_seeds}) "
            f"on {net_file} starting...",
            flush=True,
        )
        try:
            states = env.reset()
            if method == 'GC-MAPPO' and os.environ.get('GC_MAPPO_EVAL_DETERMINISTIC', '1') != '1':
                policy_seed_offset = int(os.environ.get('GC_MAPPO_EVAL_POLICY_SEED_OFFSET', '0'))
                _seed_everything(evaluation_seed + policy_seed_offset)
            invalid_actions = [False] * env.num_agents
            rewards, speeds, queues, waiting = [], [], [], []
            throughput = 0
            reward_components = {}
            last_sample_time = None
            decision_slot_count = 0
            hold_count = 0
            switch_count = 0
            nominal_yellow_advance_count = 0
            dense_step_count = 0
            done = False
            while not done:
                actions = _select_actions(method, agent, env, states, invalid_actions)
                if method in ('GC-MAPPO', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P'):
                    current_time = env.sumo.simulation.getTime()
                    for agent_index, signal in enumerate(env.traffic_signals):
                        is_decision_slot = (
                            signal.yellow_phase is None and current_time >= signal.end_time
                        )
                        if not is_decision_slot:
                            continue
                        decision_slot_count += 1
                        current_phase_index = next(
                            (
                                phase_index
                                for phase_index, phase in enumerate(signal.all_green_phases)
                                if phase.state == signal.green_phase.state
                            ),
                            None,
                        )
                        if current_phase_index == int(actions[agent_index]):
                            hold_count += 1
                        else:
                            switch_count += 1
                next_states, step_rewards, done, info = env.step(actions)
                dense_step_count += 1
                nominal_yellow_advance_count += int(info.get('nominal_yellow_advance_count', 0))
                training._accumulate_reward_components(
                    reward_components, info.get('reward_components')
                )
                global_metrics = info.get('global_metrics', {})
                throughput += int(global_metrics.get('arrived_number', 0))
                training._append_effective_multi_rewards(
                    rewards, step_rewards, info.get('do_actions', []), next_states, done,
                )
                for index, next_state in enumerate(next_states):
                    invalid_actions[index] = next_state is None and not done
                states = [
                    training._normalize_state_dim(next_state, len(states[index]))
                    if next_state is not None else states[index]
                    for index, next_state in enumerate(next_states)
                ]
                last_sample_time = training._maybe_record_metrics(
                    env, speeds, queues, waiting, last_sample_time, done,
                )
            avg_speed, avg_queue, avg_waiting = training._episode_metric_means(speeds, queues, waiting)
            evaluation_reward = training._safe_mean(rewards)
            metrics['evaluation_reward'].append(evaluation_reward)
            metrics['rewards'].append(evaluation_reward)
            metrics['speeds'].append(avg_speed)
            metrics['queue_lengths'].append(avg_queue)
            metrics['waiting_times'].append(avg_waiting)
            episode_metrics = env._collect_global_metrics()
            metrics['travel_times'].append(episode_metrics.get('travel_time'))
            metrics['demand_service_times'].append(episode_metrics.get('demand_service_time'))
            metrics['delays'].append(episode_metrics.get('delay'))
            metrics['throughput'].append(throughput)
            hours = float(training.ENV_PARAMS['simulation_time']) / 3600.0
            metrics['throughput_rates'].append(throughput / hours if hours else 0.0)
            metrics['completion_rates'].append(episode_metrics.get('completion_rate', 0.0))
            metrics['planned_completion_rates'].append(
                episode_metrics.get('planned_completion_rate', 0.0)
            )
            metrics['normal_completed_vehicles'].append(
                episode_metrics.get('normal_completed_vehicles', 0)
            )
            metrics['planned_vehicles'].append(episode_metrics.get('planned_vehicles', 0))
            metrics['demand_conservation_ok'].append(
                episode_metrics.get('demand_conservation_ok', False)
            )
            metrics['incomplete_vehicles'].append(episode_metrics.get('incomplete_vehicles', 0))
            metrics['teleport_counts'].append(episode_metrics.get('teleport_count', 0))
            gc_method = method in ('GC-MAPPO', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P')
            metrics['decision_slots'].append(decision_slot_count)
            metrics['executed_hold_rates'].append(
                hold_count / max(1, decision_slot_count) if gc_method else None
            )
            metrics['executed_switch_rates'].append(
                switch_count / max(1, decision_slot_count) if gc_method else None
            )
            metrics['yellow_advances_per_step'].append(
                nominal_yellow_advance_count / max(1, dense_step_count)
                if gc_method else None
            )
            metrics['yellow_advances_per_decision'].append(
                nominal_yellow_advance_count / max(1, decision_slot_count)
                if gc_method else None
            )
            metrics['reward_components'].append(
                training._finalize_reward_components(reward_components)
            )
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"{method} eval seed {evaluation_seed} done: "
                f"reward={metrics['rewards'][-1]:.5f} speed={avg_speed:.3f} "
                f"queue={avg_queue:.3f} waiting={avg_waiting:.3f}",
                flush=True,
            )
        finally:
            env.close()
    return metrics


def _runner(method):
    return {
        'DPG-MAPPO': training.run_dpg_mappo,
        'GC-MAPPO': training.run_gc_mappo,
        'GC-MAPPO-ind-noP': training.run_gc_mappo_ind_no_prior,
        'GC-MAPPO-ind-P': training.run_gc_mappo_ind_prior,
        'MAPPO': training.run_mappo,
        'IPPO': training.run_ippo,
        'IPPO-P': training.run_ippo_prior,
        'IPPO+': training.run_ippo_plus,
        'IPPO-Cap': training.run_ippo_cap,
        'BR-IPPO': training.run_br_ippo,
        'GR-GMAPPO-L': training.run_gr_gmappo_l,
        training.COLIGHT_NAME: training.run_colight,
        training.PRESSLIGHT_NAME: training.run_presslight,
        training.GRAPH_ATTENTION_DDQN_NAME: training.run_graph_attention_double_dqn,
        training.PRESSURE_STATE_DDQN_NAME: training.run_pressure_state_double_dqn,
        'Max-Pressure': training.run_max_pressure,
        'Fixed-Time': training.run_fixed_time,
        training.PARAMETER_SHARED_DDQN_NAME: training.run_double_dqn,
        training.PARAMETER_SHARED_DUELING_DDQN_NAME: training.run_dueling_dqn,
        training.SHARED_POLICY_GAE_NAME: training.run_shared_policy_gae_actor_critic,
        training.SHARED_POLICY_REINFORCE_NAME: training.run_reinforce,
    }.get(method)


def _source_hashes():
    paths = (
        'paper_protocol.py',
        'environment/env.py',
        'environment/traffic_signal.py',
        'environment/metrics.py',
        'test_algorithms.py',
        'agents/gc_mappo.py',
        'agents/dpg_mappo.py',
        'agents/mappo.py',
        'agents/ippo.py',
        'agents/gated_graph_residual_ippo.py',
        'agents/_colight_core.py',
        'agents/_presslight_core.py',
        'agents/colight.py',
        'agents/presslight.py',
        'agents/fixed_time.py',
        'agents/max_pressure.py',
        'agents/baselines_descriptive.py',
        'agents/double_dqn.py',
        'agents/dueling_double_dqn.py',
        'agents/gae_actor_critic.py',
        'agents/reinforce.py',
        'nets/2x2-grid/2x2.net.xml',
        'nets/2x2-grid/2x2-medium.rou.xml',
    )
    # Phase 2 P2-3 扩展：把当前评测场景的 net/route 文件也纳入哈希清单，保证复现性可追溯。
    for scenario_path in (training.ENV_PARAMS.get('net_file'), training.ENV_PARAMS.get('route_file')):
        if scenario_path and scenario_path not in paths and os.path.isfile(scenario_path):
            paths = paths + (scenario_path,)
    return {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description='Run one fixed-protocol training/evaluation task.')
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--ippo-init-checkpoint', default=None,
                        help='Matching training-seed IPPO checkpoint for BR-IPPO/GR-GMAPPO-L.')
    parser.add_argument('--sumo-seed', type=int)
    parser.add_argument('--train-episodes', type=int, default=50)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=list(range(1042, 1052)))
    parser.add_argument('--output-dir', required=True)
    # Phase 2 P2-3 扩展：场景参数化（默认 2x2-medium 不变）
    parser.add_argument('--net-file', default=None,
                        help='场景路网文件；不传则用默认 2x2.net.xml')
    parser.add_argument('--route-file', default=None,
                        help='场景需求文件；不传则用默认 2x2-medium.rou.xml')
    parser.add_argument('--tls-min-links', type=int, default=None,
                        help='路口筛选阈值；不传则默认 4')
    parser.add_argument('--resume', action='store_true',
                        help='断点续跑：加载 resume_state.pt 从中断 episode 继续；'
                             '需配合已存在的 status=running state.json')
    args = parser.parse_args()
    if args.train_episodes <= 0 or not args.eval_seeds:
        parser.error('train episodes and evaluation seeds must be positive')
    sumo_seed = args.train_seed if args.sumo_seed is None else args.sumo_seed
    state_path = os.path.join(args.output_dir, 'protocol_state.json')
    if os.path.exists(state_path):
        if args.resume:
            with open(state_path, encoding='utf-8') as f:
                existing = json.load(f)
            if existing.get('status') != 'running':
                raise FileExistsError(
                    f'Cannot --resume: state status={existing.get("status")!r}, expected running: {state_path}'
                )
        else:
            raise FileExistsError(f'Protocol task already has state: {state_path}')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.ippo_init_checkpoint:
        os.environ['GC_MAPPO_IPPO_INIT_CHECKPOINT'] = os.path.abspath(args.ippo_init_checkpoint)
    if args.method in ('BR-IPPO', 'GR-GMAPPO-L') and not os.environ.get('GC_MAPPO_IPPO_INIT_CHECKPOINT'):
        parser.error('--ippo-init-checkpoint (or GC_MAPPO_IPPO_INIT_CHECKPOINT) is required for this exploratory method')
    _configure(args.train_seed, sumo_seed,
               net_file=args.net_file, route_file=args.route_file,
               tls_min_links=args.tls_min_links)
    training._enforce_execution_standard(
        training.ENV_PARAMS['net_file'], training.ENV_PARAMS['route_file'],
        training.ENV_PARAMS['tls_min_links'], sumo_seed,
    )
    state = {
        'status': 'running', 'started_at': datetime.now().isoformat(timespec='seconds'),
        'method': args.method, 'train_seed': args.train_seed, 'sumo_seed': sumo_seed,
        'train_episodes': args.train_episodes, 'evaluation_seeds': args.eval_seeds,
        'environment': dict(training.ENV_PARAMS), 'source_hashes': _source_hashes(),
        'variant_config': dict(FORMAL_VARIANTS[args.method]) if args.method in FORMAL_VARIANTS else None,
        'ippo_initialization_checkpoint': (
            os.path.abspath(os.environ['GC_MAPPO_IPPO_INIT_CHECKPOINT'])
            if args.method in ('BR-IPPO', 'GR-GMAPPO-L') else None
        ),
        'evaluation_mode': (
            'fresh environment, checkpoint reload, '
            + ('deterministic actions' if os.environ.get('GC_MAPPO_EVAL_DETERMINISTIC', '1') == '1' else 'stochastic actions')
            + ', learning disabled'
        ),
        'gc_mappo_actor': (
            'independent_graph_actor'
            if args.method in ('GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P')
            else 'independent_graph_actor'
            if args.method == 'GC-MAPPO' and os.environ.get('GC_MAPPO_INDEPENDENT_ACTOR', '0') == '1'
            else 'shared_graph_actor'
            if args.method == 'GC-MAPPO'
            else None
        ),
        'metric_standard': {
            'speed': 'vehicle-weighted mean speed over unique controlled lanes',
            'queue': 'mean halting length over unique controlled lanes, halting_number * 7.5 m',
            'waiting': 'mean waiting time over unique vehicles on controlled lanes',
            'travel_time': 'mean completed-vehicle arrival time minus departure time',
            'delay': 'mean completed-vehicle travel time minus route free-flow travel time',
            'throughput': 'cumulative arrived vehicles per evaluation episode',
            'throughput_rate': 'throughput divided by simulation hours',
            'completion_rate': 'completed departed vehicles divided by departed vehicles',
            'incomplete_vehicles': 'departed vehicles not completed by episode end',
            'teleport_count': 'unique vehicles observed in SUMO teleport events',
            'sampling_interval_seconds': int(training.METRIC_SAMPLE_INTERVAL),
            'deduplicate_lanes': True,
            'deduplicate_vehicles': True,
        },
        'method_metadata': {
            'DPG-MAPPO': 'new candidate extending graph MAPPO with phase-specific directional downstream occupancy and gated bounded residual logits',
            'GC-MAPPO': 'graph-cooperative MAPPO extension with graph communication and cooperative centralized training',
            'GC-MAPPO-ind-noP': 'independent graph actors with centralized critic, pressure observation, soft-Q execution, and zero action-prior scale',
            'GC-MAPPO-ind-P': 'independent graph actors with centralized critic, pressure observation, soft-Q execution, and pressure prior scale 2.25',
            'MAPPO': 'graph-free multi-agent PPO with centralized training and decentralized execution',
            'IPPO': 'independent PPO policies without graph communication, centralized critic, or action prior',
            'IPPO-P': 'independent PPO policies trained and evaluated with the pressure prior at scale 2.25',
            'BR-IPPO': 'exploratory frozen IPPO actor with gated bounded one-hop neighbor-mean residual and local critic',
            'GR-GMAPPO-L': 'exploratory jointly trained IPPO actor with gated bounded one-hop neighbor-mean residual and local critic',
            training.COLIGHT_NAME: 'lightweight CoLight reimplementation; core adjacency multi-head attention neighbor communication reproduction, not the complete original paper',
            training.PRESSLIGHT_NAME: 'PressLight core-mechanism reproduction from agents/presslight.py; phase-pressure state, pressure reward, and Double-DQN target',
            'Max-Pressure': 'deterministic pressure controller; no training',
            'Fixed-Time': 'deterministic non-learning fixed-time baseline; no training',
            'Fixed-Time-Program': 'descriptive non-learning anchor; replays the net static program order (no adaptive control); no training',
            'Max-Pressure-Descriptive': 'descriptive non-learning Varaiya-style max-pressure (argmax per-phase net queue pressure); behaviorally identical to Max-Pressure; no training',
        }.get(args.method),
        'training_objective': {
            'DPG-MAPPO': 'local-global cooperative reward with semi-Markov decision return, dense centralized critic, and cached pre-action phase-direction features',
            'GC-MAPPO': 'local-global cooperative environment reward with semi-Markov decision return',
            'GC-MAPPO-ind-noP': 'local-global cooperative reward with semi-Markov actor return, dense centralized critic, and zero action-prior scale',
            'GC-MAPPO-ind-P': 'local-global cooperative reward with semi-Markov actor return, dense centralized critic, and pressure prior scale 2.25',
            'MAPPO': 'local-global cooperative environment reward with PPO return',
            'IPPO': 'local-global cooperative environment reward with independent PPO returns and no action prior',
            'IPPO-P': 'local-global cooperative environment reward with independent PPO returns and rollout/update-consistent pressure prior',
            'BR-IPPO': 'local-global cooperative reward; clipped PPO on gated residual and local critic with frozen IPPO actor',
            'GR-GMAPPO-L': 'local-global cooperative reward; clipped PPO jointly updates local actor, gated residual, and local critic',
            training.COLIGHT_NAME: 'CoLight core-mechanism reproduction from agents/colight.py with adjacency multi-head attention neighbor communication and Double-DQN replay updates',
            training.PRESSLIGHT_NAME: 'PressLight reproduction with phase-pressure state, pressure reward, and Double-DQN replay updates',
            training.GRAPH_ATTENTION_DDQN_NAME: 'optional Graph-Attention Double DQN mechanism baseline',
            training.PRESSURE_STATE_DDQN_NAME: 'optional Pressure-State Double DQN mechanism baseline',
            'Max-Pressure': 'deterministic pressure controller; no training return',
            'Fixed-Time': 'deterministic non-learning fixed-time controller; no training return',
            'Fixed-Time-Program': 'no training return; descriptive non-RL anchor',
            'Max-Pressure-Descriptive': 'no training return; descriptive non-RL max-pressure',
            training.PARAMETER_SHARED_DDQN_NAME: 'Double-DQN replay updates; implementation uses one parameter-shared network across intersections',
            training.PARAMETER_SHARED_DUELING_DDQN_NAME: 'Dueling Double-DQN replay updates; implementation uses one parameter-shared network across intersections',
            training.SHARED_POLICY_REINFORCE_NAME: 'local-global cooperative environment reward with vanilla on-policy categorical REINFORCE Monte-Carlo return and no learned value baseline',
            training.SHARED_POLICY_GAE_NAME: 'GAE actor-critic return; implementation uses one shared policy/value network across intersections',
        }.get(args.method, 'algorithm-specific training objective'),
        'reward_standard': {
            'evaluation_reward_field': 'evaluation_reward',
            'training_return_is_algorithm_specific': True,
            'global_reward_weight': os.environ.get('GC_MAPPO_GLOBAL_REWARD_WEIGHT', '0.30'),
            'clearance_reward_coef': os.environ.get('GC_MAPPO_CLEARANCE_REWARD_COEF', '0.0'),
            'calibrated_centers_scales': {
                'speed': [os.environ.get('GC_MAPPO_REWARD_SPEED_CENTER', '6.0'), os.environ.get('GC_MAPPO_REWARD_SPEED_SCALE', '1.0')],
                'queue': [os.environ.get('GC_MAPPO_REWARD_QUEUE_CENTER', '8.0'), os.environ.get('GC_MAPPO_REWARD_QUEUE_SCALE', '3.0')],
                'waiting': [os.environ.get('GC_MAPPO_REWARD_WAITING_CENTER', '12.0'), os.environ.get('GC_MAPPO_REWARD_WAITING_SCALE', '4.0')],
            },
        },
        'execution_options': {
            'gc_mappo_pressure_confidence_mask': (
                os.environ.get('GC_MAPPO_PRESSURE_CONFIDENCE_MASK') == '1'
            ),
            'gc_mappo_pressure_confidence_threshold': (
                training.GC_MAPPO_PRESSURE_CONFIDENCE_THRESHOLD
            ),
            'gc_mappo_action_prior_scale': os.environ.get('GC_MAPPO_ACTION_PRIOR_SCALE', '2.25'),
            'gc_mappo_hold_prior_bonus': os.environ.get('GC_MAPPO_HOLD_PRIOR_BONUS', '0.0'),
            'gc_mappo_eval_policy_temperature': os.environ.get(
                'GC_MAPPO_EVAL_POLICY_TEMPERATURE',
                os.environ.get('GC_MAPPO_TEMP_END', '0.30'),
            ),
            'gc_mappo_eval_deterministic': (
                os.environ.get('GC_MAPPO_EVAL_DETERMINISTIC', '1') == '1'
            ),
            'gc_mappo_eval_policy_seed_offset': os.environ.get(
                'GC_MAPPO_EVAL_POLICY_SEED_OFFSET', '0'
            ),
            'gc_mappo_soft_q_execution': (
                os.environ.get('GC_MAPPO_SOFT_Q_EXECUTION', '1') == '1'
            ),
            'gc_mappo_soft_q_policy_support_only': (
                os.environ.get('GC_MAPPO_SOFT_Q_POLICY_SUPPORT_ONLY', '0') == '1'
            ),
            'gc_mappo_dense_critic_exclude_decision_states': (
                os.environ.get('GC_MAPPO_DENSE_CRITIC_EXCLUDE_DECISION_STATES', '0') == '1'
            ),
            'global_reward_weight': os.environ.get('GC_MAPPO_GLOBAL_REWARD_WEIGHT', '0.30'),
            'clearance_reward_coef': os.environ.get('GC_MAPPO_CLEARANCE_REWARD_COEF', '0.0'),
            'reward_calibration': {
                'speed_center': os.environ.get('GC_MAPPO_REWARD_SPEED_CENTER', '6.0'),
                'speed_scale': os.environ.get('GC_MAPPO_REWARD_SPEED_SCALE', '1.0'),
                'queue_center': os.environ.get('GC_MAPPO_REWARD_QUEUE_CENTER', '8.0'),
                'queue_scale': os.environ.get('GC_MAPPO_REWARD_QUEUE_SCALE', '3.0'),
                'waiting_center': os.environ.get('GC_MAPPO_REWARD_WAITING_CENTER', '12.0'),
                'waiting_scale': os.environ.get('GC_MAPPO_REWARD_WAITING_SCALE', '4.0'),
            },
        },
    }
    with open(state_path, 'w', encoding='utf-8') as output:
        json.dump(state, output, indent=2, ensure_ascii=False)
    checkpoint_path = None
    if args.method in ('Max-Pressure', 'Fixed-Time') or args.method in DESCRIPTIVE_METHODS:
        train_metrics = None
    else:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')
        resume_path = os.path.join(args.output_dir, 'resume_state.pt')
        resume_state = None
        if args.resume and os.path.isfile(resume_path):
            resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
            _ep = resume_state.get('episode', 0)
            print(f'[{args.method}] --resume: loaded resume_state.pt (episode={_ep}/{args.train_episodes})')
        if resume_state and int(resume_state.get('episode', 0)) >= args.train_episodes and os.path.isfile(checkpoint_path):
            # Training already completed (interrupted during evaluation); skip retraining
            print(f'[{args.method}] --resume: episode={resume_state.get("episode")} >= {args.train_episodes}, '
                  f'training already done, skipping to evaluation')
            train_metrics = resume_state.get('metrics')
        else:
            # training_log.jsonl: append one JSON line per episode for full training curve.
            # On --resume, truncate to start_episode lines first to avoid duplicate entries.
            log_path = os.path.join(args.output_dir, 'training_log.jsonl')
            start_episode = int((resume_state or {}).get('episode', 0))
            if start_episode > 0 and os.path.isfile(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if len(lines) > start_episode:
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines[:start_episode])
            elif not os.path.isfile(log_path):
                open(log_path, 'w').close()

            # save_progress callback: save resume_state.pt after each episode (atomic write)
            def save_progress(episode, metrics, agent, replay_buffer=None):
                chk = {
                    'episode': int(episode),
                    'metrics': metrics,
                    'rng_state': training._capture_rng_state(),
                    'agent': agent.state_dict() if agent is not None else None,
                    'method': args.method,
                }
                if replay_buffer is not None:
                    chk['replay_buffer'] = {
                        'memory': replay_buffer.memory,
                        'steps_done': replay_buffer.steps_done,
                    }
                tmp_path = f'{resume_path}.tmp'
                torch.save(chk, tmp_path)
                os.replace(tmp_path, resume_path)

                # Append per-episode metrics to training_log.jsonl (one JSON per line, durable)
                try:
                    last_reward = metrics['rewards'][-1] if metrics.get('rewards') else None
                    last_speed = metrics['speeds'][-1] if metrics.get('speeds') else None
                    last_queue = metrics['queue_lengths'][-1] if metrics.get('queue_lengths') else None
                    last_waiting = metrics['waiting_times'][-1] if metrics.get('waiting_times') else None
                except (IndexError, KeyError, TypeError):
                    last_reward = last_speed = last_queue = last_waiting = None
                log_entry = {
                    'episode': int(episode),
                    'reward': last_reward,
                    'speed': last_speed,
                    'queue': last_queue,
                    'waiting': last_waiting,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'method': args.method,
                    'train_seed': args.train_seed,
                    'sumo_seed': sumo_seed,
                }
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    f.flush()

            training_output = _runner(args.method)(
                args.train_episodes,
                checkpoint_path=checkpoint_path,
                resume_state=resume_state,
                on_episode_complete=save_progress,
            )
            train_metrics = training_output.get(args.method, training_output)
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(f'Training completed without checkpoint: {checkpoint_path}')
        # Persist train_metrics early so evaluation-interruption can be recovered
        state['training_metrics'] = train_metrics
        with open(state_path, 'w', encoding='utf-8') as output:
            json.dump(state, output, indent=2, ensure_ascii=False)
    evaluation_metrics = _evaluate(args.method, checkpoint_path, args.eval_seeds)
    state.update({
        'status': 'completed', 'finished_at': datetime.now().isoformat(timespec='seconds'),
        'checkpoint_path': checkpoint_path, 'training_metrics': train_metrics,
        'evaluation_metrics': evaluation_metrics,
    })
    with open(state_path, 'w', encoding='utf-8') as output:
        json.dump(state, output, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as output:
        json.dump(evaluation_metrics, output, indent=2, ensure_ascii=False)
    print(f"Protocol task completed: {args.method}, seed={args.train_seed}")


if __name__ == '__main__':
    main()
