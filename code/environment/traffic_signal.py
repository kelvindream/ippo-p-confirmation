import os

import numpy as np
from gymnasium import spaces
from environment.metrics import collect_lane_metrics


class TrafficSignal:
    def __init__(
        self,
        ts_id: str,
        yellow_time: int,
        simulation_time: float,
        delta_rs_update_time: int,
        sumo
    ):
        self.ts_id = ts_id
        self.yellow_time = yellow_time
        self.simulation_time = simulation_time
        self.delta_rs_update_time = delta_rs_update_time
        self.rs_update_time = 0
        self.sumo = sumo
        self.green_phase = None
        self.yellow_phase = None
        self.end_time = 0
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id) + len(self.all_green_phases), dtype=np.float32),
            high=np.ones(len(self.lanes_id) + len(self.all_green_phases), dtype=np.float32))
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        self.last_measure = 0.0
        self.reward_speed_center = float(os.environ.get('GC_MAPPO_REWARD_SPEED_CENTER', '6.0'))
        self.reward_speed_scale = float(os.environ.get('GC_MAPPO_REWARD_SPEED_SCALE', '1.0'))
        self.reward_queue_center = float(os.environ.get('GC_MAPPO_REWARD_QUEUE_CENTER', '8.0'))
        self.reward_queue_scale = float(os.environ.get('GC_MAPPO_REWARD_QUEUE_SCALE', '3.0'))
        self.reward_waiting_center = float(os.environ.get('GC_MAPPO_REWARD_WAITING_CENTER', '12.0'))
        self.reward_waiting_scale = float(os.environ.get('GC_MAPPO_REWARD_WAITING_SCALE', '4.0'))
        self.prev_avg_speed = None
        self.prev_avg_queue_length = None
        self.prev_avg_waiting_time = None
        self.continue_reward = False
        self.dict_lane_veh = None
        self.last_reward_components = {}
        self.phase_lane_map = self._build_phase_lane_map()
        self.phase_link_map = self._build_phase_link_map()
        self._init_current_phase()

    def _build_phase_lane_map(self):
        """按相位动态映射各绿灯相位所服务的进口车道（不依赖硬编码车道名）。"""
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        incoming_lanes = []
        for links in controlled_links:
            if links:
                incoming_lanes.append(links[0][0])

        phase_lane_map = []
        for green_phase in self.all_green_phases:
            lanes = []
            state = green_phase.state
            for link_idx, lane_id in enumerate(incoming_lanes):
                if link_idx < len(state) and state[link_idx] in ('G', 'g'):
                    if lane_id not in lanes:
                        lanes.append(lane_id)
            if not lanes:
                lanes = list(self.lanes_id)
            phase_lane_map.append(lanes)
        return phase_lane_map

    def _build_phase_link_map(self):
        """Map every green phase to its active incoming/outgoing lane pairs."""
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        phase_links = []
        for green_phase in self.all_green_phases:
            links = []
            for link_idx, alternatives in enumerate(controlled_links):
                if link_idx >= len(green_phase.state) or green_phase.state[link_idx] not in ('G', 'g'):
                    continue
                for incoming_lane, outgoing_lane, _ in alternatives:
                    pair = (incoming_lane, outgoing_lane)
                    if pair not in links:
                        links.append(pair)
            phase_links.append(links)
        return phase_links

    def phase_pressure(self, phase_index):
        """Return net queue pressure for a candidate phase: incoming minus outgoing queues."""
        links = self.phase_link_map[int(phase_index)]
        if not links:
            return float(sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.phase_lane_map[int(phase_index)]))
        return float(sum(
            self.sumo.lane.getLastStepHaltingNumber(incoming_lane)
            - self.sumo.lane.getLastStepHaltingNumber(outgoing_lane)
            for incoming_lane, outgoing_lane in links
        ))

    def pressure_state(self):
        """Per-phase net pressures used by pressure-state control baselines."""
        return np.asarray(
            [self.phase_pressure(index) for index in range(len(self.all_green_phases))],
            dtype=np.float32,
        )

    def max_pressure_action(self):
        pressures = self.pressure_state()
        return int(np.argmax(pressures)) if pressures.size else 0

    def _init_current_phase(self):
        current_state = self.sumo.trafficlight.getRedYellowGreenState(self.ts_id)
        for phase in self.all_green_phases:
            if phase.state == current_state:
                self.green_phase = phase
                self.yellow_phase = None
                self.update_end_time()
                return
        self.green_phase = self.all_green_phases[0]
        self.yellow_phase = None
        self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
        self.update_end_time()

    def change_phase(self, new_green_phase):
        """
        :param new_green_phase: 绿灯相位索引
        :return: do_action -> 实际执行的动作索引；-1 表示黄灯过渡中；None 表示无效动作
        """
        new_green_phase = self.all_green_phases[new_green_phase]
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is not None:
            if current_time >= self.end_time:
                self.yellow_phase = None
                self.update_end_time()
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                do_action = self.green_phase
            else:
                do_action = self.yellow_phase
        else:
            if current_time >= self.end_time:
                if new_green_phase.state == self.green_phase.state:
                    do_action = None
                else:
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    self.green_phase = new_green_phase
                    self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time
                    self.update_end_time()
                    do_action = self.yellow_phase
            else:
                do_action = self.green_phase

        if do_action is None:
            return None

        if 'y' in do_action.state:
            do_action = -1
        else:
            for i, green_phase in enumerate(self.all_green_phases):
                if do_action.state == green_phase.state:
                    do_action = i
                    break

        return do_action

    def update_end_time(self):
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is None:
            self.end_time = current_time + self.green_phase.duration
        else:
            self.end_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        # rs_update_time 只由相位切换设置，不能在到期后推到仿真结束，
        # 否则 compute_next_state 会长期返回 None，丢失绝大多数 transition。
        return self._reward_with_speed(start, do_action)

    def _action_wait_counts(self):
        """各相位对应车道的等待车辆数（动态映射）。"""
        self.dict_lane_veh = {
            lane_id: self.sumo.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in self.lanes_id
        }
        counts = []
        for lanes in self.phase_lane_map:
            counts.append(sum(self.dict_lane_veh.get(lane_id, 0) for lane_id in lanes))
        return counts

    def _directional_delta(self, current_value, previous_value, scale, higher_is_better=True):
        """
        将业务指标变化映射到 [-1, 1]，并对“变差”给予更强惩罚，
        使 reward 更稳定地反映真实交通效果。
        """
        if previous_value is None:
            return 0.0

        raw_delta = (current_value - previous_value) if higher_is_better else (previous_value - current_value)
        score = float(np.tanh(raw_delta / scale))
        if score < 0.0:
            score *= 1.25
        return score

    def _reward_with_speed(self, start, do_action):
        dict_action_wait_num = self._action_wait_counts()
        best_action = int(np.argmax(dict_action_wait_num)) if dict_action_wait_num else 0

        # 启发式动作只保留很弱的引导作用，避免掩盖真实交通指标。
        if do_action is not None and do_action >= 0:
            action_reward = 0.02 if best_action == do_action else -0.02
        else:
            action_reward = -0.03

        avg_speed = self.get_average_speed()
        avg_queue_length = self.get_average_queue_length()
        avg_waiting_time = self.get_average_waiting_time()

        # 指标口径调整后，speed 变成了“按车辆加权、包含近停滞状态”的真实均速，
        # 需要把 reward 的目标中心同步回当前 2x2-core 场景的实际数值范围。
        speed_score = float(np.tanh(
            (avg_speed - self.reward_speed_center) / self.reward_speed_scale
        ))
        queue_score = float(np.tanh(
            (self.reward_queue_center - avg_queue_length) / self.reward_queue_scale
        ))
        waiting_score = float(np.tanh(
            (self.reward_waiting_center - avg_waiting_time) / self.reward_waiting_scale
        ))
        linear_score = 0.42 * speed_score + 0.22 * queue_score + 0.36 * waiting_score

        if self.prev_avg_speed is None:
            speed_delta = 0.0
            queue_delta = 0.0
            waiting_delta = 0.0
            delta_score = 0.0
        else:
            speed_delta = self._directional_delta(avg_speed, self.prev_avg_speed, scale=0.42, higher_is_better=True)
            queue_delta = self._directional_delta(avg_queue_length, self.prev_avg_queue_length, scale=8.0, higher_is_better=False)
            waiting_delta = self._directional_delta(avg_waiting_time, self.prev_avg_waiting_time, scale=7.0, higher_is_better=False)
            delta_score = 0.52 * speed_delta + 0.18 * queue_delta + 0.30 * waiting_delta

        speed_gate = 0.5 * (1.0 + speed_score)
        queue_gate = 0.5 * (1.0 + queue_score)
        waiting_gate = 0.5 * (1.0 + waiting_score)
        congestion_gate = 0.44 * queue_gate + 0.56 * waiting_gate

        # 联合门控仍是主体，但额外增加 speed_alignment_gate：
        # 高 speed 只有在拥堵指标也达标时，才会被放大为真正的高 reward。
        joint_gate = speed_gate * queue_gate * waiting_gate
        weakest_gate = min(speed_gate, queue_gate, waiting_gate)
        speed_alignment_gate = speed_gate * (0.25 + 0.75 * congestion_gate)
        balanced_state_score = 2.0 * (
            0.44 * joint_gate
            + 0.22 * weakest_gate
            + 0.34 * speed_alignment_gate
        ) - 1.0

        positive_speed_delta = max(0.0, speed_delta) if self.prev_avg_speed is not None else 0.0
        positive_queue_delta = max(0.0, queue_delta) if self.prev_avg_speed is not None else 0.0
        positive_waiting_delta = max(0.0, waiting_delta) if self.prev_avg_speed is not None else 0.0
        aligned_progress_bonus = positive_speed_delta * (
            0.55 * positive_queue_delta + 0.45 * positive_waiting_delta
        )
        aligned_speed_bonus = max(0.0, speed_gate - 0.50) * (0.25 + 0.75 * congestion_gate)

        negative_speed_delta = max(0.0, -speed_delta) if self.prev_avg_speed is not None else 0.0
        negative_queue_delta = max(0.0, -queue_delta) if self.prev_avg_speed is not None else 0.0
        negative_waiting_delta = max(0.0, -waiting_delta) if self.prev_avg_speed is not None else 0.0
        regression_penalty = (
            0.48 * negative_speed_delta
            + 0.18 * negative_queue_delta
            + 0.34 * negative_waiting_delta
        )

        # 分成两层惩罚：
        # 1) overload 处理一般性拥堵偏高；
        # 2) barrier 专门压制 eval 中出现过的高等待极端反例。
        queue_overload = max(0.0, avg_queue_length - 66.0)
        waiting_overload = max(0.0, avg_waiting_time - 28.0)
        overload_penalty = (
            -0.26 * float(np.tanh(queue_overload / 10.0))
            -0.48 * float(np.tanh(waiting_overload / 14.0))
        )

        queue_barrier = max(0.0, avg_queue_length - 85.0)
        waiting_barrier = max(0.0, avg_waiting_time - 80.0)
        barrier_penalty = (
            -0.30 * float(np.tanh(queue_barrier / 10.0))
            -0.60 * float(np.tanh(waiting_barrier / 28.0))
        )

        speed_misalignment = max(0.0, speed_gate - congestion_gate)
        slow_flow_penalty = max(0.0, 0.48 - speed_gate)
        anti_counterexample_penalty = (
            -0.36 * float(np.tanh(speed_misalignment / 0.14))
            -0.22 * float(np.tanh(slow_flow_penalty / 0.16))
        )

        self.prev_avg_speed = avg_speed
        self.prev_avg_queue_length = avg_queue_length
        self.prev_avg_waiting_time = avg_waiting_time

        # 最终 reward 的优先级：
        # 1. 高 reward 必须同时具备较好的速度与较低拥堵；
        # 2. 速度提升若伴随 queue/waiting 改善，会得到额外奖励；
        # 3. 高等待极端样本直接拉低 reward，避免 eval 再次出现“reward 好但业务差”。
        base_reward = (
            0.02 * action_reward
            + 0.22 * linear_score
            + 0.14 * delta_score
            + 0.30 * balanced_state_score
            + 0.16 * aligned_progress_bonus
            + 0.12 * aligned_speed_bonus
            + 0.10 * overload_penalty
            + 0.10 * barrier_penalty
            + 0.08 * anti_counterexample_penalty
            - 0.12 * regression_penalty
        )
        
        # 关键修复：在极度拥堵情况下，强制 reward 为负，避免奖励函数被利用
        # 只要满足以下任一条件，reward 最多只能是 -0.1
        extreme_congestion = (
            avg_speed < 1.0 
            or avg_queue_length > 100.0 
            or avg_waiting_time > 150.0
        )
        if extreme_congestion:
            self.last_measure = min(base_reward, -0.1)
        else:
            self.last_measure = base_reward
        # 调试期记录 reward 分解项，供上层在 train/eval 现场抓取证据。
        self.last_reward_components = {
            'avg_speed': float(avg_speed),
            'avg_queue_length': float(avg_queue_length),
            'avg_waiting_time': float(avg_waiting_time),
            'action_reward': float(action_reward),
            'linear_score': float(linear_score),
            'delta_score': float(delta_score),
            'balanced_state_score': float(balanced_state_score),
            'aligned_progress_bonus': float(aligned_progress_bonus),
            'aligned_speed_bonus': float(aligned_speed_bonus),
            'overload_penalty': float(overload_penalty),
            'barrier_penalty': float(barrier_penalty),
            'anti_counterexample_penalty': float(anti_counterexample_penalty),
            'regression_penalty': float(regression_penalty),
            'speed_gate': float(speed_gate),
            'queue_gate': float(queue_gate),
            'waiting_gate': float(waiting_gate),
            'congestion_gate': float(congestion_gate),
            'joint_gate': float(joint_gate),
            'weakest_gate': float(weakest_gate),
            'speed_alignment_gate': float(speed_alignment_gate),
            'best_action': int(best_action),
            'do_action': (None if do_action is None else int(do_action)),
            'reward': float(self.last_measure),
        }
        return self.last_measure

    def compute_next_state(self):
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            density = self.get_lanes_density()
            phase = self.get_current_phase_one_hot()
            return np.array(density + phase, dtype=np.float32)
        return None

    def compute_state(self):
        density = self.get_lanes_density()
        phase = self.get_current_phase_one_hot()
        return np.array(density + phase, dtype=np.float32)

    def get_current_phase_one_hot(self):
        phase_one_hot = [0.0] * len(self.all_green_phases)
        current_state = self.sumo.trafficlight.getRedYellowGreenState(self.ts_id)
        for i, phase in enumerate(self.all_green_phases):
            if phase.state == current_state:
                phase_one_hot[i] = 1.0
                break
        return phase_one_hot

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5
        return [
            min(1, self.sumo.lane.getLastStepVehicleNumber(lane_id) /
                (self.lanes_length[lane_id] / vehicle_size_min_gap))
            for lane_id in self.lanes_id
        ]

    def _local_metrics(self):
        return collect_lane_metrics(self.sumo, self.lanes_id, deduplicate=True)

    def get_average_speed(self):
        return self._local_metrics()['speed']

    def get_average_queue_length(self):
        return self._local_metrics()['queue']

    def get_average_waiting_time(self):
        return self._local_metrics()['waiting']
