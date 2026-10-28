import numpy as np
import torch
import torch.nn as nn

from agents.gc_mappo import GCMAPPO, device


VEHICLE_SPACING_METERS = 7.5


def directional_phase_features(traffic_signals):
    """Build [agents, phases, 3] incoming queue, outgoing occupancy, storage pressure."""
    features = []
    for signal in traffic_signals:
        phase_features = []
        for links in signal.phase_link_map:
            incoming = [pair[0] for pair in links]
            outgoing = [pair[1] for pair in links]
            in_queue = [signal.sumo.lane.getLastStepHaltingNumber(lane) for lane in incoming]
            out_occupancy, storage_pressure = [], []
            for lane in outgoing:
                capacity = max(1.0, signal.lanes_length.get(
                    lane, signal.sumo.lane.getLength(lane)
                ) / VEHICLE_SPACING_METERS)
                vehicles = signal.sumo.lane.getLastStepVehicleNumber(lane)
                out_occupancy.append(min(1.0, vehicles / capacity))
                out_queues = signal.sumo.lane.getLastStepHaltingNumber(lane)
                storage_pressure.append(min(1.0, out_queues / capacity))
            phase_features.append([
                min(1.0, float(np.mean(in_queue)) / 20.0) if in_queue else 0.0,
                float(np.mean(out_occupancy)) if out_occupancy else 0.0,
                float(np.mean(storage_pressure)) if storage_pressure else 0.0,
            ])
        features.append(phase_features)
    return np.asarray(features, dtype=np.float32)


class DirectionalActor(nn.Module):
    def __init__(self, actor, num_agents, action_dim, hidden_dim, feature_dim=3, residual_limit=0.5):
        super().__init__()
        self.actor = actor
        self.residual_limit = float(residual_limit)
        self.residual = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.gate = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, observations, adjacency, distances, directional_features=None):
        logits = self.actor(observations, adjacency, distances)
        if directional_features is None:
            return logits
        features = torch.as_tensor(directional_features, dtype=logits.dtype, device=logits.device)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        if features.shape[:3] != logits.shape:
            raise ValueError("directional_features must have shape [batch, agents, actions, features]")
        residual = self.residual_limit * torch.tanh(self.residual(features).squeeze(-1))
        gate = torch.sigmoid(self.gate(features).squeeze(-1))
        return logits + gate * residual


class DPGMAPPO(GCMAPPO):
    def __init__(self, *args, directional_feature_dim=3, residual_limit=0.5, actor_lr=3e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_actor = self.actor
        self.actor = DirectionalActor(
            self.base_actor, self.num_agents, self.action_dim, self.base_actor.encoder.hidden_dim,
            feature_dim=directional_feature_dim, residual_limit=residual_limit,
        ).to(device)
        for group in self.actor_optimizer.param_groups:
            group['params'] = list(self.actor.parameters())
        self.actor_optimizer.state.clear()

    @torch.no_grad()
    def select_actions(self, observations, action_masks=None, action_priors=None,
                       policy_temperature=None, deterministic=False, pressure_context=None,
                       directional_features=None):
        obs = self._tensor_observations(observations)
        masks = self._tensor_masks(action_masks)
        logits = self._apply_policy_temperature(
            self._logits(obs, masks, action_priors, directional_features=directional_features),
            policy_temperature,
        )
        from torch.distributions import Categorical
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        context = None if pressure_context is None else torch.as_tensor(pressure_context, dtype=torch.float32, device=device).unsqueeze(0)
        value = self._critic_values(obs, context)
        self.steps_done += 1
        return actions.squeeze(0).cpu().numpy().astype(np.int64).tolist(), log_probs.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()

    def _logits(self, observations, action_masks=None, action_priors=None,
                soft_q_bias_override=None, directional_features=None):
        if directional_features is None:
            directional_features = getattr(self, "_update_directional_features", None)
        base = self.base_actor(observations, self.adjacency_matrix, self.distance_matrix)
        if action_priors is not None:
            priors = torch.as_tensor(action_priors, dtype=base.dtype, device=device)
            if priors.ndim == 2:
                priors = priors.unsqueeze(0)
        if directional_features is None:
            logits = base
        else:
            logits = self.actor(
                observations, self.adjacency_matrix, self.distance_matrix, directional_features
            )
        if action_priors is not None:
            logits = logits + self.action_prior_scale * priors
        if soft_q_bias_override is None:
            logits, soft_execution_rate = self._soft_q_logits(logits, observations)
        else:
            logits = logits + soft_q_bias_override.detach()
            soft_execution_rate = self.last_soft_q_execution_rate
        self.last_soft_q_execution_rate = soft_execution_rate
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
        return logits

    def evaluate_actions(self, observations, actions, action_masks=None, action_priors=None,
                         policy_temperatures=None, soft_q_bias_override=None,
                         pressure_context=None, directional_features=None):
        masks = None if action_masks is None else torch.as_tensor(action_masks, dtype=torch.bool, device=device)
        logits = self._apply_policy_temperature(
            self._logits(observations, masks, action_priors,
                         soft_q_bias_override=soft_q_bias_override,
                         directional_features=directional_features), policy_temperatures
        )
        from torch.distributions import Categorical
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions.long()), distribution.entropy(), self._critic_values(observations, pressure_context)

    def update(self, *args, directional_features=None, **kwargs):
        self._update_directional_features = directional_features
        try:
            return super().update(*args, **kwargs)
        finally:
            self._update_directional_features = None

    def save(self, path):
        super().save(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        checkpoint["actor"] = self.actor.state_dict()
        checkpoint["dpg_actor"] = self.actor.state_dict()
        torch.save(checkpoint, path)

    def load(self, path, map_location=device):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        actor_state = checkpoint["dpg_actor"] if "dpg_actor" in checkpoint else checkpoint["actor"]
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(checkpoint["critic"])
        if self.residual_critic is not None and "residual_critic" in checkpoint:
            self.residual_critic.load_state_dict(checkpoint["residual_critic"])
        if self.action_value_head is not None and "action_value_head" in checkpoint:
            self.action_value_head.load_state_dict(checkpoint["action_value_head"])
        if self.action_value_reference_head is not None and "action_value_reference_head" in checkpoint:
            self.action_value_reference_head.load_state_dict(checkpoint["action_value_reference_head"])

    def state_dict(self):
        state = super().state_dict()
        state["actor"] = self.actor.state_dict()
        state["actor_type"] = "dpg"
        return state

    def load_state_dict(self, state):
        saved = dict(state)
        saved["actor_type"] = "independent" if self.independent_actor else "shared"
        actor_state = saved.pop("actor")
        super().load_state_dict(saved)
        self.actor.load_state_dict(actor_state)
