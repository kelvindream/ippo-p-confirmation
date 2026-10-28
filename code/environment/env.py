import math
import gymnasium as gym
import os
import sys
import random
from environment.traffic_signal import TrafficSignal
from environment.metrics import VehicleMetricTracker, collect_network_metrics

# SUMO 安装在受限目录时，禁用 .pyc 写入以避免训练被沙箱拦截。
sys.dont_write_bytecode = True

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib


def build_intersection_graph(traffic_signals, sumo, connect_distance=800.0):
    """根据路口几何位置构建邻接矩阵与距离矩阵。"""
    n = len(traffic_signals)
    adjacency = [[0] * n for _ in range(n)]
    distance = [[0.0] * n for _ in range(n)]
    positions = []
    for ts in traffic_signals:
        positions.append(sumo.junction.getPosition(ts.ts_id))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
            distance[i][j] = d
            if d <= connect_distance:
                adjacency[i][j] = 1
    return adjacency, distance


class SumoEnv(gym.Env):
    def __init__(
        self,
        net_file: str,
        route_file: str,
        skip_range: int,
        simulation_time: float,
        yellow_time: int,
        delta_rs_update_time: int,
        use_gui: bool = False,
        multi_agent: bool = False,
        tls_min_links: int = 0,
        reward_cooperation_enabled: bool = True,
        normalize_reward: bool = False,
        sumo_seed: int = 42,
    ):
        self._net = net_file
        self._route = route_file
        self.skip_range = skip_range
        self.simulation_time = simulation_time
        self.use_gui = use_gui
        self.yellow_time = yellow_time
        self.delta_rs_update_time = delta_rs_update_time
        self.multi_agent = multi_agent
        self.tls_min_links = int(tls_min_links or 0)
        self.reward_cooperation_enabled = bool(reward_cooperation_enabled)
        self.normalize_reward = bool(normalize_reward)
        self.sumo_seed = int(sumo_seed)
        self._rng = random.Random(self.sumo_seed)
        self.global_reward_weight = float(os.environ.get('GC_MAPPO_GLOBAL_REWARD_WEIGHT', '0.30'))
        if not 0.0 <= self.global_reward_weight <= 1.0:
            raise ValueError('GC_MAPPO_GLOBAL_REWARD_WEIGHT must be in [0, 1]')
        self.clearance_reward_coef = float(
            os.environ.get('GC_MAPPO_CLEARANCE_REWARD_COEF', '0.0')
        )
        if not 0.0 <= self.clearance_reward_coef <= 1.0:
            raise ValueError('GC_MAPPO_CLEARANCE_REWARD_COEF must be in [0, 1]')
        self._global_prev_metrics = None
        self.vehicle_metrics = None
        self.tripinfo_output = os.environ.get('SUMO_TRIPINFO_OUTPUT')
        self.time_to_teleport = int(os.environ.get('SUMO_TIME_TO_TELEPORT', '1000'))

        self.train_state = None
        self.train_states = None
        self.last_phase_state = None
        self.change_action_times = {}
        self.sumo = None
        self.sumoBinary = 'sumo-gui' if use_gui else 'sumo'
        self.ts_id = None
        self.traffic_signal = None
        self.traffic_signals = []
        self.ts_ids = []
        self.adjacency_matrix = None
        self.distance_matrix = None

    def _normalize_actions(self, action):
        if isinstance(action, (list, tuple)):
            return list(action)
        if hasattr(action, '__iter__') and not isinstance(action, (str, bytes)):
            return list(action)
        return [int(action)]

    def step(self, action):
        actions = self._normalize_actions(action)
        if not self.multi_agent:
            actions = [actions[0]]

        while len(actions) < len(self.traffic_signals):
            actions.append(actions[-1])

        do_actions, nominal_yellow_advance_count = self._apply_actions_and_advance(actions)

        start_flags = []
        for i, ts in enumerate(self.traffic_signals):
            start = False
            if i in self.change_action_times:
                if self.sumo.simulation.getTime() >= self.change_action_times[i]:
                    del self.change_action_times[i]
                    start = True
            start_flags.append(start)

        if self.multi_agent:
            next_states = self._compute_next_states()
            rewards = self._compute_rewards(start_flags, do_actions)
            train_states = self._compute_states()
            if any(start_flags):
                self.train_states = train_states
            done = self._compute_done()
            info = {
                'do_actions': do_actions,
                'multi_agent': True,
                'nominal_yellow_advance_count': nominal_yellow_advance_count,
                'reward_components': [dict(ts.last_reward_components) for ts in self.traffic_signals],
                'local_rewards': self._last_reward_info.get('local_rewards', []),
                'global_reward': self._last_reward_info.get('global_reward'),
                'global_metrics': self._last_reward_info.get('global_metrics', {}),
            }
            return next_states, rewards, done, info

        next_state = self._compute_next_state()
        reward = self._compute_reward(start_flags[0], do_actions[0])
        if start_flags[0]:
            self.train_state = self._compute_state()
        if reward is None:
            reward = 0.0
        done = self._compute_done()
        info = {'do_action': do_actions[0], 'multi_agent': False}
        info['reward_components'] = [dict(self.traffic_signal.last_reward_components)]
        return next_state, reward, done, info

    def step_multi_agent(self, actions):
        """执行一次多智能体环境步并返回 next_states、rewards、done、info。"""
        if not self.multi_agent:
            raise RuntimeError('step_multi_agent requires multi_agent=True')
        actions = self._normalize_actions(actions)
        while len(actions) < len(self.traffic_signals):
            actions.append(actions[-1] if actions else 0)
        actions = actions[:len(self.traffic_signals)]

        do_actions, nominal_yellow_advance_count = self._apply_actions_and_advance(actions)

        start_flags = []
        for i in range(len(self.traffic_signals)):
            start = False
            if i in self.change_action_times:
                if self.sumo.simulation.getTime() >= self.change_action_times[i]:
                    del self.change_action_times[i]
                    start = True
            start_flags.append(start)

        next_states = self._compute_next_states()
        rewards = self._compute_rewards(start_flags, do_actions)
        done = self._compute_done()
        if any(start_flags):
            self.train_states = self._compute_states()
        return next_states, rewards, done, {
            'do_actions': do_actions,
            'start_flags': start_flags,
            'multi_agent': True,
            'nominal_yellow_advance_count': nominal_yellow_advance_count,
            'reward_components': [dict(ts.last_reward_components) for ts in self.traffic_signals],
            'local_rewards': self._last_reward_info.get('local_rewards', []),
            'global_reward': self._last_reward_info.get('global_reward'),
            'global_metrics': self._last_reward_info.get('global_metrics', {}),
        }

    def _apply_actions_and_advance(self, actions):
        do_actions = []
        newly_started_yellow = []
        current_time = self.sumo.simulation.getTime()
        for i, ts in enumerate(self.traffic_signals):
            decision_available = bool(
                ts.yellow_phase is None and current_time >= ts.end_time
            )
            do_action = ts.change_phase(actions[i])
            do_actions.append(do_action)
            self.action_event_log.append({
                'time': current_time,
                'signal_index': i,
                'requested_action': int(actions[i]),
                'executed_action': do_action,
                'decision_available': decision_available,
            })
            if do_action == -1 and i not in self.change_action_times:
                self.change_action_times[i] = current_time + self.yellow_time
                newly_started_yellow.append(i)

        nominal_yellow_advance_count = 0
        yellow_completion_time = current_time + self.yellow_time
        target_time = min(
            current_time + self.delta_rs_update_time,
            float(self.simulation_time),
        )
        if newly_started_yellow and yellow_completion_time < target_time:
            self.sumo.simulationStep(yellow_completion_time)
            if self.vehicle_metrics is not None:
                self.vehicle_metrics.update()
            for signal_index in newly_started_yellow:
                self.traffic_signals[signal_index].change_phase(actions[signal_index])
            nominal_yellow_advance_count = len(newly_started_yellow)
        self.sumo.simulationStep(target_time)
        if self.vehicle_metrics is not None:
            self.vehicle_metrics.update()
        return do_actions, nominal_yellow_advance_count

    def _random_skip(self, skip_range=30):
        for ts in self.traffic_signals:
            ts.sumo = self.sumo
            rand_idx = self._rng.randint(0, len(ts.all_green_phases) - 1)
            ts.yellow_phase = None
            ts.green_phase = ts.all_green_phases[rand_idx]
            self.sumo.trafficlight.setRedYellowGreenState(ts.ts_id, ts.green_phase.state)
            ts.update_end_time()
            ts.rs_update_time = 0

        skip_seconds = self._rng.randint(0, skip_range)
        initial_states = self._compute_states()
        for _ in range(skip_seconds):
            actions = [
                self._rng.randint(0, len(ts.all_green_phases) - 1)
                for ts in self.traffic_signals
            ]
            next_states, _, _, _ = self.step(actions)
            if self.multi_agent:
                if all(s is not None for s in next_states):
                    initial_states = next_states
            elif next_states is not None:
                initial_states = [next_states]

        if self.multi_agent:
            self.train_states = initial_states
            return initial_states
        self.train_state = initial_states[0]
        return initial_states[0]

    def reset(self):
        try:
            traci.close()
        except Exception:
            pass

        sumo_cmd = [
            sumolib.checkBinary(self.sumoBinary),
            '-c', self._net.replace('net.xml', 'sumocfg'),
            '--route-files', os.path.abspath(self._route),
            '--time-to-teleport', str(self.time_to_teleport),
            '--no-step-log', 'true',
            '--duration-log.disable', 'true',
            '--no-warnings', 'true',
            '--seed', str(self.sumo_seed),
        ]
        if self.tripinfo_output:
            sumo_cmd.extend([
                '--tripinfo-output', os.path.abspath(self.tripinfo_output),
                '--tripinfo-output.write-unfinished', 'true',
            ])
        self.sumo_command = list(sumo_cmd)
        sumo_error_log = os.environ.get('SUMO_ERROR_LOG')
        if sumo_error_log:
            sumo_cmd.extend(['--error-log', sumo_error_log])
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        traci.start(sumo_cmd)
        self.sumo = traci
        self.change_action_times = {}
        self.action_event_log = []
        self._rng = random.Random(self.sumo_seed)
        self._global_prev_metrics = None
        self._last_reward_info = {}
        self.vehicle_metrics = VehicleMetricTracker(
            self.sumo,
            route_file=self._route,
            demand_end=self.simulation_time,
        )

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            self.skip_range = 0

        self.sumo.simulationStep()
        self.vehicle_metrics.update()

        self.ts_ids = list(traci.trafficlight.getIDList())
        if self.tls_min_links > 0:
            filtered_ts_ids = []
            for ts_id in self.ts_ids:
                controlled_links = [
                    links for links in traci.trafficlight.getControlledLinks(ts_id) if links
                ]
                if len(controlled_links) >= self.tls_min_links:
                    filtered_ts_ids.append(ts_id)
            if filtered_ts_ids:
                self.ts_ids = filtered_ts_ids
        self.traffic_signals = []
        for ts_id in self.ts_ids:
            ts = TrafficSignal(
                ts_id=ts_id,
                yellow_time=self.yellow_time,
                simulation_time=self.simulation_time,
                delta_rs_update_time=self.delta_rs_update_time,
                sumo=self.sumo,
            )
            self.traffic_signals.append(ts)

        self.traffic_signal = self.traffic_signals[0]
        self.ts_id = self.ts_ids[0]
        self.adjacency_matrix, self.distance_matrix = build_intersection_graph(
            self.traffic_signals, self.sumo
        )

        self._observation_space = self.traffic_signal.observation_space
        self._action_space = self.traffic_signal.action_space
        
        # 添加多智能体支持的属性
        self.num_agents = len(self.traffic_signals)

        return self._random_skip(self.skip_range)
    
    def get_agent_spaces(self):
        """返回每个智能体的观测空间和动作空间"""
        spaces = []
        for ts in self.traffic_signals:
            spaces.append((ts.observation_space, ts.action_space))
        return spaces

    def close(self):
        if self.sumo is None:
            return
        try:
            self.sumo.close()
        except Exception:
            raise

    def render(self):
        pass

    def _compute_states(self):
        return [ts.compute_state() for ts in self.traffic_signals]

    def _compute_state(self):
        return self.traffic_signal.compute_state()

    def _compute_next_states(self):
        return [ts.compute_next_state() for ts in self.traffic_signals]

    def _compute_next_state(self):
        return self.traffic_signal.compute_next_state()

    def _collect_global_metrics(self):
        """Collect network metrics with the shared metric definitions."""
        metrics = collect_network_metrics(self.sumo, self.traffic_signals)
        if self.vehicle_metrics is not None:
            metrics.update(self.vehicle_metrics.snapshot())
        return metrics

    def _compute_global_reward(self, metrics):
        """Compute a bounded shared reward from the same global metrics used in evaluation."""
        ts = self.traffic_signals[0]
        speed_score = math.tanh(
            (metrics['speed'] - ts.reward_speed_center) / ts.reward_speed_scale
        )
        queue_score = math.tanh(
            (ts.reward_queue_center - metrics['queue']) / ts.reward_queue_scale
        )
        waiting_score = math.tanh(
            (ts.reward_waiting_center - metrics['waiting']) / ts.reward_waiting_scale
        )
        current_score = (
            0.42 * speed_score + 0.22 * queue_score + 0.36 * waiting_score
        )
        delta_score = 0.0
        if self._global_prev_metrics is not None:
            speed_delta = math.tanh(
                (metrics['speed'] - self._global_prev_metrics['speed']) / 0.42
            )
            queue_delta = math.tanh(
                (self._global_prev_metrics['queue'] - metrics['queue']) / 8.0
            )
            waiting_delta = math.tanh(
                (self._global_prev_metrics['waiting'] - metrics['waiting']) / 7.0
            )
            delta_score = 0.52 * speed_delta + 0.18 * queue_delta + 0.30 * waiting_delta
        base_reward = 0.78 * current_score + 0.22 * delta_score
        arrival_fraction = metrics['arrived_number'] / max(
            1.0,
            metrics['arrived_number'] + metrics['vehicle_count'],
        )
        clearance_score = math.tanh(arrival_fraction / 0.10)
        metrics['clearance_score'] = clearance_score
        self._global_prev_metrics = dict(metrics)
        return float(
            (1.0 - self.clearance_reward_coef) * base_reward
            + self.clearance_reward_coef * clearance_score
        )

    def _mix_rewards(self, local_rewards, global_reward):
        weight = self.global_reward_weight if self.reward_cooperation_enabled else 0.0
        return [
            (1.0 - weight) * float(local_reward) + weight * global_reward
            for local_reward in local_rewards
        ]

    def _compute_rewards(self, start_flags, do_actions):
        local_rewards = [
            ts.compute_reward(start_flags[i], do_actions[i])
            for i, ts in enumerate(self.traffic_signals)
        ]
        metrics = self._collect_global_metrics()
        global_reward = self._compute_global_reward(metrics)
        self._last_reward_info = {
            'local_rewards': [float(value) for value in local_rewards],
            'global_reward': global_reward,
            'global_metrics': metrics,
        }
        return self._mix_rewards(local_rewards, global_reward)

    def _compute_reward(self, start_flag, do_action):
        local_reward = self.traffic_signal.compute_reward(start_flag, do_action)
        metrics = self._collect_global_metrics()
        global_reward = self._compute_global_reward(metrics)
        self._last_reward_info = {
            'local_rewards': [float(local_reward)],
            'global_reward': global_reward,
            'global_metrics': metrics,
        }
        return self._mix_rewards([local_reward], global_reward)[0]

    def _compute_done(self):
        return self.sumo.simulation.getTime() >= self.simulation_time

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
