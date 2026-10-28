"""Graph-Cooperative Multi-Agent PPO for multi-intersection traffic signal control.

The module provides centralized training and decentralized execution (CTDE):
- actors use local observations plus distance-aware neighbor observations;
- the critic uses the concatenated global observation;
- discrete actions support per-agent legal-action masks;
- PPO clipping, GAE, entropy regularization, and gradient clipping are used.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class MAPPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    action_masks: Optional[torch.Tensor]


class GraphAttentionEncoder(nn.Module):
    """Encode local observations with distance-aware graph attention."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.local = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.distance_decay = nn.Parameter(torch.zeros(heads))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, observations, adjacency, distances):
        # observations: [B, N, obs_dim], adjacency/distances: [N, N]
        local = self.local(observations)
        batch_size, num_agents, _ = local.shape
        q = self.query(local).view(batch_size, num_agents, self.heads, self.head_dim)
        k = self.key(local).view(batch_size, num_agents, self.heads, self.head_dim)
        v = self.value(local).view(batch_size, num_agents, self.heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bijh", q, k)
        scores = scores / np.sqrt(self.head_dim)

        valid_neighbors = adjacency.bool().clone()
        valid_neighbors.fill_diagonal_(True)
        distance_scale = distances[valid_neighbors].mean().clamp_min(1.0)
        normalized_distances = distances / distance_scale
        distance_penalty = torch.nn.functional.softplus(self.distance_decay).view(1, 1, -1)
        scores = scores - normalized_distances.unsqueeze(-1) * distance_penalty
        scores = scores.masked_fill(~valid_neighbors.unsqueeze(0).unsqueeze(-1), -1e9)
        attention = torch.softmax(scores, dim=2)
        messages = torch.einsum("bijh,bjhd->bihd", attention, v)
        messages = messages.reshape(batch_size, num_agents, self.hidden_dim)
        return self.output(torch.cat([local, messages], dim=-1))


class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, heads=4, agent_embedding_dim=0):
        super().__init__()
        self.agent_embedding_dim = agent_embedding_dim
        self.encoder = GraphAttentionEncoder(obs_dim, hidden_dim, heads)
        if agent_embedding_dim > 0:
            self.agent_embedding = nn.Embedding(256, agent_embedding_dim)
        actor_input_dim = hidden_dim + agent_embedding_dim
        self.policy = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations, adjacency, distances, agent_ids=None):
        features = self.encoder(observations, adjacency, distances)
        if self.agent_embedding_dim > 0:
            if agent_ids is None:
                agent_ids = torch.arange(
                    observations.shape[1], device=observations.device
                )
            embedding = self.agent_embedding(agent_ids.long())
            features = torch.cat(
                [features, embedding.view(1, -1, self.agent_embedding_dim).expand(features.shape[0], -1, -1)],
                dim=-1,
            )
        return self.policy(features)


class IndependentGraphActor(nn.Module):
    """Keep graph encoding shared while giving each intersection its own policy head."""

    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=128, heads=4,
                 agent_embedding_dim=0):
        super().__init__()
        self.encoder = GraphAttentionEncoder(obs_dim, hidden_dim, heads)
        self.agent_embedding_dim = agent_embedding_dim
        if agent_embedding_dim > 0:
            self.agent_embedding = nn.Embedding(256, agent_embedding_dim)
        actor_input_dim = hidden_dim + agent_embedding_dim
        self.policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(actor_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            for _ in range(num_agents)
        ])

    def forward(self, observations, adjacency, distances, agent_ids=None):
        features = self.encoder(observations, adjacency, distances)
        if self.agent_embedding_dim > 0:
            if agent_ids is None:
                agent_ids = torch.arange(observations.shape[1], device=observations.device)
            embedding = self.agent_embedding(agent_ids.long())
            features = torch.cat([
                features,
                embedding.view(1, -1, self.agent_embedding_dim).expand(features.shape[0], -1, -1),
            ], dim=-1)
        logits = [policy(features[:, agent_idx]) for agent_idx, policy in enumerate(self.policies)]
        return torch.stack(logits, dim=1)


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_agents * obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agents),
        )

    def forward(self, observations):
        return self.network(observations.reshape(observations.shape[0], -1))


class ContextualResidualCritic(nn.Module):
    """Add an optional zero-initialized traffic-intensity value correction."""

    def __init__(self, critic, num_agents, pressure_context=False):
        super().__init__()
        self.critic = critic
        self.pressure_context = bool(pressure_context)
        self.context_head = None
        if self.pressure_context:
            self.context_head = nn.Linear(num_agents, num_agents)
            nn.init.zeros_(self.context_head.weight)
            nn.init.zeros_(self.context_head.bias)

    def forward(self, observations, pressure_context=None):
        values = self.critic(observations)
        if self.context_head is None:
            return values
        if pressure_context is None:
            raise ValueError("pressure_context is required for contextual residual values")
        if pressure_context.shape != values.shape:
            raise ValueError("pressure_context must have shape [batch, num_agents]")
        return values + self.context_head(pressure_context)


class ActionConditionedValueHead(nn.Module):
    """Centralized dueling Q residual for real phase-decision transitions."""

    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(num_agents * obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.action_advantages = nn.Linear(hidden_dim, num_agents * action_dim)
        # Preserve C46's initial rollout distribution before this residual learns.
        nn.init.zeros_(self.action_advantages.weight)
        nn.init.zeros_(self.action_advantages.bias)

    def forward(self, observations, state_values):
        batch_size = observations.shape[0]
        features = self.trunk(observations.reshape(batch_size, -1))
        raw_advantages = self.action_advantages(features).view(
            batch_size, self.num_agents, self.action_dim
        )
        centered_advantages = raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)
        return state_values.unsqueeze(-1) + centered_advantages, centered_advantages


class GCMAPPO:
    """Graph-Cooperative Multi-Agent PPO agent.

    Inputs are represented as [num_agents, obs_dim]. The shared actor is used
    independently for every intersection, while the critic receives all agents'
    observations during training.
    """

    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        adjacency_matrix=None,
        distance_matrix=None,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        actor_lr=3e-4,
        critic_lr=1e-3,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=5,
        max_grad_norm=0.5,
        hidden_dim=128,
        agent_embedding_dim=0,
        action_prior_scale=0.0,
        relative_value_coef=0.0,
        action_value_head=False,
        action_value_correction_scale=0.10,
        action_value_lr=None,
        action_value_consensus=False,
        soft_q_execution=False,
        soft_q_policy_agreement=False,
        soft_q_policy_agreement_scale=0.25,
        soft_q_policy_support_only=False,
        soft_q_value_disagreement_gate=False,
        soft_q_margin_calibration=False,
        action_value_margin_calibration=False,
        action_value_per_agent_normalization=False,
        action_value_unselected_anchor=False,
        action_value_unselected_conservative_anchor=False,
        action_value_unselected_behavior_anchor=False,
        dense_critic_exclude_decision_states=False,
        residual_global_dominant_sharpening=False,
        residual_critic=False,
        adaptive_residual_consensus=False,
        residual_pressure_context=False,
        independent_actor=False,
        adaptive_residual_energy_mix=False,
        residual_energy_power=2.0,
        residual_consensus_detach=False,
        family='GC-MAPPO',
        prior_mode='pressure_zscore_hold',
        pressure_observation=True,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.action_prior_scale = float(action_prior_scale)
        self.relative_value_coef = float(relative_value_coef)
        self.action_value_head = None
        self.action_value_optimizer = None
        self.action_value_reference_head = None
        self.action_value_reference_optimizer = None
        self.action_value_consensus = bool(action_value_consensus)
        self.soft_q_execution = bool(soft_q_execution)
        self.soft_q_policy_agreement = bool(soft_q_policy_agreement)
        self.soft_q_policy_agreement_scale = float(soft_q_policy_agreement_scale)
        self.soft_q_policy_support_only = bool(soft_q_policy_support_only)
        self.soft_q_value_disagreement_gate = bool(soft_q_value_disagreement_gate)
        self.soft_q_margin_calibration = bool(soft_q_margin_calibration)
        self.action_value_margin_calibration = bool(action_value_margin_calibration)
        self.action_value_per_agent_normalization = bool(action_value_per_agent_normalization)
        self.action_value_unselected_anchor = bool(action_value_unselected_anchor)
        self.action_value_unselected_conservative_anchor = bool(
            action_value_unselected_conservative_anchor
        )
        self.action_value_unselected_behavior_anchor = bool(
            action_value_unselected_behavior_anchor
        )
        self.dense_critic_exclude_decision_states = bool(
            dense_critic_exclude_decision_states
        )
        self.residual_global_dominant_sharpening = bool(residual_global_dominant_sharpening)
        self.residual_critic_enabled = bool(residual_critic)
        self.adaptive_residual_consensus = bool(adaptive_residual_consensus)
        self.residual_pressure_context = bool(residual_pressure_context)
        self.independent_actor = bool(independent_actor)
        self.adaptive_residual_energy_mix = bool(adaptive_residual_energy_mix)
        self.residual_energy_power = float(residual_energy_power)
        self.residual_consensus_detach = bool(residual_consensus_detach)
        self.family = str(family)
        self.prior_mode = str(prior_mode)
        self.pressure_observation = bool(pressure_observation)
        self.hidden_dim = int(hidden_dim)
        self.last_soft_q_execution_rate = 0.0
        self.last_soft_q_policy_agreement = 0.0
        self.last_soft_q_policy_support_rate = 0.0
        self.last_soft_q_value_agreement = 0.0
        self.action_value_margin_scale = 0.0
        self.last_soft_q_margin_scale = 0.0
        self.action_value_update_scale = 0.0
        self.last_action_value_update_scale = 0.0
        self.last_action_value_agent_scale_spread = 0.0
        self.last_residual_global_sharpen_rate = 0.0
        self.last_residual_global_mix = 0.0
        self.action_value_correction_scale = float(action_value_correction_scale)

        actor_class = IndependentGraphActor if self.independent_actor else SharedActor
        if self.independent_actor:
            self.actor = actor_class(
                obs_dim, action_dim, num_agents, hidden_dim, agent_embedding_dim=agent_embedding_dim
            ).to(device)
        else:
            self.actor = actor_class(
                obs_dim, action_dim, hidden_dim, agent_embedding_dim=agent_embedding_dim
            ).to(device)
        self.critic = CentralizedCritic(num_agents, obs_dim, hidden_dim * 2).to(device)
        self.residual_critic = None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.residual_critic_optimizer = None
        if self.residual_critic_enabled:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            residual_critic = CentralizedCritic(num_agents, obs_dim, hidden_dim * 2).to(device)
            nn.init.zeros_(residual_critic.network[-1].weight)
            nn.init.zeros_(residual_critic.network[-1].bias)
            self.residual_critic = ContextualResidualCritic(
                residual_critic, num_agents, self.residual_pressure_context
            ).to(device)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self.residual_critic_optimizer = torch.optim.Adam(
                self.residual_critic.parameters(), lr=critic_lr
            )
        if action_value_head:
            # Head construction must not advance the RNG stream used to sample
            # the first rollout relative to the C54 baseline.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.action_value_head = ActionConditionedValueHead(
                num_agents, obs_dim, action_dim, hidden_dim * 2
            ).to(device)
            if self.action_value_consensus:
                self.action_value_reference_head = ActionConditionedValueHead(
                    num_agents, obs_dim, action_dim, hidden_dim * 2
                ).to(device)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self.action_value_optimizer = torch.optim.Adam(
                self.action_value_head.parameters(),
                lr=critic_lr if action_value_lr is None else action_value_lr,
            )
            if self.action_value_reference_head is not None:
                self.action_value_reference_optimizer = torch.optim.Adam(
                    self.action_value_reference_head.parameters(),
                    lr=critic_lr if action_value_lr is None else action_value_lr,
                )

        self.adjacency_matrix = self._matrix_or_default(adjacency_matrix, 1.0)
        self.distance_matrix = self._matrix_or_default(distance_matrix, 0.0)
        self.steps_done = 0

    def _matrix_or_default(self, matrix, default):
        if matrix is None:
            matrix = np.full((self.num_agents, self.num_agents), default, dtype=np.float32)
            if default == 0.0:
                np.fill_diagonal(matrix, 0.0)
            return torch.as_tensor(matrix, dtype=torch.float32, device=device)
        array = np.asarray(matrix, dtype=np.float32)
        if array.shape != (self.num_agents, self.num_agents):
            raise ValueError("graph matrix shape must be [num_agents, num_agents]")
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def set_graph(self, adjacency_matrix, distance_matrix):
        self.adjacency_matrix = self._matrix_or_default(adjacency_matrix, 1.0)
        self.distance_matrix = self._matrix_or_default(distance_matrix, 0.0)

    def _tensor_observations(self, observations):
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.num_agents, self.obs_dim):
            raise ValueError("observations must have shape [num_agents, obs_dim]")
        return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)

    def _tensor_masks(self, action_masks):
        if action_masks is None:
            return None
        array = np.asarray(action_masks, dtype=bool)
        if array.shape != (self.num_agents, self.action_dim):
            raise ValueError("action_masks must have shape [num_agents, action_dim]")
        if np.any(array.sum(axis=1) == 0):
            raise ValueError("each agent must have at least one valid action")
        return torch.as_tensor(array, dtype=torch.bool, device=device).unsqueeze(0)

    def _critic_values(self, observations, pressure_context=None):
        primary_values = self.critic(observations)
        if self.residual_critic is None:
            self.last_residual_global_mix = 0.0
            return primary_values
        residual_values = self.residual_critic(observations, pressure_context)
        if self.adaptive_residual_consensus:
            # A common residual is useful when the four junction values agree,
            # but organized traffic needs the local residual differences.  The
            # mix is inferred from the current residual decomposition instead
            # of being scheduled by episode number.
            global_residual = residual_values.mean(dim=1, keepdim=True)
            local_offsets = residual_values - global_residual
            global_magnitude = global_residual.abs()
            local_disagreement = local_offsets.pow(2).mean(dim=1, keepdim=True).sqrt()
            if self.adaptive_residual_energy_mix:
                global_energy = global_magnitude.pow(self.residual_energy_power)
                local_energy = local_disagreement.pow(self.residual_energy_power)
                global_mix = global_energy / (global_energy + local_energy + 1e-8)
                if self.residual_global_dominant_sharpening:
                    global_dominant = global_magnitude > local_disagreement
                    sharpened_global_energy = global_magnitude.pow(
                        self.residual_energy_power + 1.0
                    )
                    sharpened_local_energy = local_disagreement.pow(
                        self.residual_energy_power + 1.0
                    )
                    sharpened_mix = sharpened_global_energy / (
                        sharpened_global_energy + sharpened_local_energy + 1e-8
                    )
                    global_mix = torch.where(global_dominant, sharpened_mix, global_mix)
                    self.last_residual_global_sharpen_rate = float(
                        global_dominant.float().mean().item()
                    )
                else:
                    self.last_residual_global_sharpen_rate = 0.0
            else:
                global_mix = global_magnitude / (
                    global_magnitude + local_disagreement + 1e-8
                )
                self.last_residual_global_sharpen_rate = 0.0
            mix_for_residual = global_mix.detach() if self.residual_consensus_detach else global_mix
            residual_values = global_residual + (1.0 - mix_for_residual) * local_offsets
            self.last_residual_global_mix = float(global_mix.detach().mean().item())
        else:
            self.last_residual_global_mix = 0.0
            self.last_residual_global_sharpen_rate = 0.0
        return primary_values + residual_values

    def _soft_q_logits(self, logits, observations):
        if (
            not self.soft_q_execution
            or self.action_value_head is None
            or self.action_value_reference_head is None
        ):
            self.last_soft_q_policy_agreement = 0.0
            self.last_soft_q_policy_support_rate = 0.0
            self.last_soft_q_value_agreement = 0.0
            return logits, 0.0
        with torch.no_grad():
            baseline_values = self.critic(observations)
            q_values, _ = self.action_value_head(observations, baseline_values)
            reference_q_values, _ = self.action_value_reference_head(
                observations, baseline_values
            )
            policy_best = logits.argmax(dim=-1)
            if self.soft_q_policy_support_only:
                # The Q heads receive return targets only for executed actions.
                # Restrict their execution role to supporting the actor's own
                # preferred action, rather than ranking unexecuted alternatives.
                primary_policy_advantage = (
                    q_values.gather(-1, policy_best.unsqueeze(-1)).squeeze(-1)
                    - baseline_values
                )
                reference_policy_advantage = (
                    reference_q_values.gather(-1, policy_best.unsqueeze(-1)).squeeze(-1)
                    - baseline_values
                )
                mean_policy_advantage = 0.5 * (
                    primary_policy_advantage + reference_policy_advantage
                )
                disagreement = (
                    primary_policy_advantage - reference_policy_advantage
                ).abs()
                support = (
                    (primary_policy_advantage > 0.0)
                    & (reference_policy_advantage > 0.0)
                )
                support_confidence = mean_policy_advantage.abs() / (
                    mean_policy_advantage.abs() + disagreement + 1e-8
                )
                support_bias = (
                    self.action_value_correction_scale
                    * torch.tanh(
                        mean_policy_advantage / (
                            mean_policy_advantage.abs() + disagreement + 1e-8
                        )
                    )
                    * support_confidence
                    * support.to(logits.dtype)
                )
                bias = torch.zeros_like(logits).scatter_(
                    -1, policy_best.unsqueeze(-1), support_bias.unsqueeze(-1)
                )
                support_rate = float(support.float().mean().item())
                self.last_soft_q_policy_agreement = support_rate
                self.last_soft_q_policy_support_rate = support_rate
                self.last_soft_q_value_agreement = 0.0
                return logits + bias, support_rate
            primary_best = q_values.argmax(dim=-1)
            reference_best = reference_q_values.argmax(dim=-1)
            primary_spread = q_values.std(dim=-1, unbiased=False)
            reference_spread = reference_q_values.std(dim=-1, unbiased=False)
            consensus = (
                primary_best.eq(reference_best)
                & (primary_spread > 1e-6)
                & (reference_spread > 1e-6)
            )
            ensemble_q_values = 0.5 * (q_values + reference_q_values)
            ensemble_advantages = ensemble_q_values - baseline_values.unsqueeze(-1)
            ensemble_spread = ensemble_advantages.std(dim=-1, unbiased=False)
            head_disagreement = (q_values - reference_q_values).abs().mean(dim=-1)
            value_agreement = ensemble_spread / (
                ensemble_spread + head_disagreement + 1e-8
            )
            top_values = ensemble_q_values.topk(k=2, dim=-1).values
            margin = top_values[..., 0] - top_values[..., 1]
            if self.soft_q_margin_calibration and self.action_value_margin_scale > 0.0:
                margin_scale = ensemble_advantages.new_tensor(self.action_value_margin_scale)
                confidence = margin / (margin + margin_scale + 1e-8)
                normalized_advantages = ensemble_advantages / (margin_scale + 1e-8)
                self.last_soft_q_margin_scale = self.action_value_margin_scale
            else:
                confidence = margin / (margin + ensemble_spread + 1e-8)
                normalized_advantages = ensemble_advantages / (ensemble_spread.unsqueeze(-1) + 1e-8)
                self.last_soft_q_margin_scale = 0.0
            policy_q_agreement = policy_best.eq(primary_best) & consensus
            policy_confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
            bias = (
                self.action_value_correction_scale
                * torch.tanh(normalized_advantages)
                * confidence.unsqueeze(-1)
                * consensus.unsqueeze(-1).to(logits.dtype)
            )
            if self.soft_q_value_disagreement_gate:
                bias = bias * value_agreement.unsqueeze(-1)
            if self.soft_q_policy_agreement:
                # Reinforce a phase only when the graph actor and independent
                # Q heads agree. Disagreement retains the C58 correction,
                # avoiding the hard Q override that failed in C57.
                reinforcement = 1.0 + self.soft_q_policy_agreement_scale * (
                    policy_q_agreement.to(logits.dtype) * policy_confidence
                )
                bias = bias * reinforcement.unsqueeze(-1)
            consensus_rate = float(consensus.float().mean().item())
            self.last_soft_q_policy_agreement = float(
                policy_q_agreement.float().mean().item()
            )
            self.last_soft_q_policy_support_rate = 0.0
            self.last_soft_q_value_agreement = float(value_agreement.mean().item())
        # Keep the Q-derived bias detached while preserving the actor graph.
        return logits + bias, consensus_rate

    def _base_logits(self, observations, action_priors=None):
        logits = self.actor(
            observations,
            self.adjacency_matrix,
            self.distance_matrix,
        )
        if action_priors is not None:
            priors = torch.as_tensor(action_priors, dtype=logits.dtype, device=device)
            if priors.ndim == 2:
                priors = priors.unsqueeze(0)
            if priors.shape != logits.shape:
                raise ValueError("action_priors must have shape [batch, num_agents, action_dim]")
            logits = logits + self.action_prior_scale * priors
        return logits

    def _logits(self, observations, action_masks=None, action_priors=None,
                soft_q_bias_override=None):
        logits = self._base_logits(observations, action_priors)
        if soft_q_bias_override is None:
            logits, soft_execution_rate = self._soft_q_logits(logits, observations)
        else:
            if soft_q_bias_override.shape != logits.shape:
                raise ValueError("soft_q_bias_override must match policy logits")
            logits = logits + soft_q_bias_override.detach()
            soft_execution_rate = self.last_soft_q_execution_rate
        self.last_soft_q_execution_rate = soft_execution_rate
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
        return logits

    @staticmethod
    def _apply_policy_temperature(logits, policy_temperature=None):
        if policy_temperature is None:
            return logits
        temperature = torch.as_tensor(
            policy_temperature, dtype=logits.dtype, device=logits.device
        ).clamp_min(0.05)
        if temperature.ndim == 0:
            return logits / temperature
        if temperature.ndim == 1:
            if temperature.shape[0] != logits.shape[0]:
                raise ValueError("policy_temperatures must have one value per rollout step")
            return logits / temperature.view(-1, 1, 1)
        raise ValueError("policy_temperature must be scalar or shape [steps]")

    @torch.no_grad()
    def select_actions(self, observations, action_masks=None, action_priors=None,
                       policy_temperature=None, deterministic=False, pressure_context=None):
        obs = self._tensor_observations(observations)
        masks = self._tensor_masks(action_masks)
        logits = self._apply_policy_temperature(
            self._logits(obs, masks, action_priors), policy_temperature
        )
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        context = None if pressure_context is None else torch.as_tensor(
            pressure_context, dtype=torch.float32, device=device
        ).unsqueeze(0)
        value = self._critic_values(obs, context)
        self.steps_done += 1
        return (
            actions.squeeze(0).cpu().numpy().astype(np.int64).tolist(),
            log_probs.squeeze(0).cpu().numpy(),
            value.squeeze(0).cpu().numpy(),
        )

    def evaluate_actions(self, observations, actions, action_masks=None, action_priors=None,
                         policy_temperatures=None, soft_q_bias_override=None,
                         pressure_context=None):
        masks = None
        if action_masks is not None:
            masks = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
        logits = self._apply_policy_temperature(
            self._logits(
                observations, masks, action_priors,
                soft_q_bias_override=soft_q_bias_override,
            ),
            policy_temperatures,
        )
        distribution = Categorical(logits=logits)
        actions = actions.long()
        return (
            distribution.log_prob(actions), distribution.entropy(),
            self._critic_values(observations, pressure_context),
        )

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        if rewards.ndim != 2 or rewards.shape[1] != self.num_agents:
            raise ValueError("rewards must have shape [steps, num_agents]")
        if values.shape != rewards.shape:
            raise ValueError("values must have shape [steps, num_agents]")
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.num_agents, dtype=np.float32)
        next_value = np.broadcast_to(last_value, (self.num_agents,)).astype(np.float32).copy()
        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def compute_semi_markov_gae(
        self,
        rewards,
        values,
        dones,
        transition_mask,
        transition_discounts,
        gae_discounts,
        last_value=0.0,
    ):
        """Compute vector GAE over each agent's executable phase decisions."""
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        mask = np.asarray(transition_mask, dtype=bool)
        bootstrap_discounts = np.asarray(transition_discounts, dtype=np.float32)
        trace_discounts = np.asarray(gae_discounts, dtype=np.float32)
        expected_shape = rewards.shape
        if (
            rewards.ndim != 2
            or rewards.shape[1] != self.num_agents
            or values.shape != expected_shape
            or dones.shape != expected_shape
            or mask.shape != expected_shape
            or bootstrap_discounts.shape != expected_shape
            or trace_discounts.shape != expected_shape
        ):
            raise ValueError("semi-Markov rollout tensors must have shape [steps, num_agents]")

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        fallback_last_value = np.broadcast_to(last_value, (self.num_agents,)).astype(np.float32)
        for agent_index in range(self.num_agents):
            decision_indices = np.flatnonzero(mask[:, agent_index])
            next_advantage = 0.0
            for position in range(len(decision_indices) - 1, -1, -1):
                step_index = decision_indices[position]
                terminal = bool(dones[step_index, agent_index])
                if terminal:
                    next_value = 0.0
                elif position + 1 < len(decision_indices):
                    next_value = values[decision_indices[position + 1], agent_index]
                else:
                    next_value = fallback_last_value[agent_index]
                delta = (
                    rewards[step_index, agent_index]
                    + bootstrap_discounts[step_index, agent_index] * next_value * (1.0 - float(terminal))
                    - values[step_index, agent_index]
                )
                next_advantage = (
                    delta
                    + trace_discounts[step_index, agent_index]
                    * (1.0 - float(terminal))
                    * next_advantage
                )
                advantages[step_index, agent_index] = next_advantage
                returns[step_index, agent_index] = next_advantage + values[step_index, agent_index]

        valid_advantages = advantages[mask]
        if valid_advantages.size:
            advantages[mask] = (
                (valid_advantages - valid_advantages.mean())
                / (valid_advantages.std() + 1e-8)
            )
        return returns, advantages

    def update(self, observations, actions, old_log_probs, rewards, values, dones, action_masks=None,
               action_priors=None, policy_temperatures=None, last_value=0.0, transition_mask=None,
               transition_discounts=None, gae_discounts=None, dense_critic_observations=None,
               dense_critic_rewards=None, dense_critic_values=None, dense_critic_dones=None,
               dense_critic_decision_rows=None,
               pressure_contexts=None, dense_pressure_contexts=None):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        old_log_probs = np.asarray(old_log_probs, dtype=np.float32)
        if transition_mask is None:
            returns, advantages = self.compute_gae(rewards, values, dones, last_value)
        else:
            if transition_discounts is None or gae_discounts is None:
                raise ValueError("semi-Markov updates require bootstrap and GAE discounts")
            returns, advantages = self.compute_semi_markov_gae(
                rewards,
                values,
                dones,
                transition_mask,
                transition_discounts,
                gae_discounts,
                last_value,
            )

        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        act = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        masks = None if action_masks is None else torch.as_tensor(action_masks, dtype=torch.bool, device=device)
        priors = None if action_priors is None else torch.as_tensor(action_priors, dtype=torch.float32, device=device)
        temperatures = None if policy_temperatures is None else torch.as_tensor(
            policy_temperatures, dtype=torch.float32, device=device
        )
        contexts = None if pressure_contexts is None else torch.as_tensor(
            pressure_contexts, dtype=torch.float32, device=device
        )
        if self.residual_pressure_context and contexts is None:
            raise ValueError("pressure_contexts are required for contextual residual updates")
        if transition_mask is None:
            valid_entries = torch.ones_like(ret)
        else:
            valid_entries = torch.as_tensor(transition_mask, dtype=torch.float32, device=device)
            if valid_entries.shape != ret.shape:
                raise ValueError("transition_mask must have shape [steps, num_agents]")

        action_value_correction = None
        action_value_confidence = 0.0
        action_value_correction_std = 0.0
        action_value_consensus_rate = 0.0
        if self.action_value_head is not None:
            with torch.no_grad():
                baseline_values = self._critic_values(obs, contexts)
                q_values, _ = self.action_value_head(obs, baseline_values)
                selected_q_advantages = (
                    q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1) - baseline_values
                )
                reference_q_values = None
                if self.action_value_reference_head is not None:
                    reference_q_values, _ = self.action_value_reference_head(
                        obs, baseline_values
                    )
                    reference_selected_advantages = (
                        reference_q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                        - baseline_values
                    )
                    matching_best_action = q_values.argmax(dim=-1).eq(
                        reference_q_values.argmax(dim=-1)
                    )
                    matching_selected_direction = (
                        selected_q_advantages * reference_selected_advantages > 0.0
                    )
                    consensus_gate = (
                        matching_best_action & matching_selected_direction
                    ).to(q_values.dtype)
                if (
                    self.action_value_margin_calibration
                    and reference_q_values is not None
                    and self.action_value_update_scale > 0.0
                ):
                    ensemble_q_values = 0.5 * (q_values + reference_q_values)
                    ensemble_selected_advantages = (
                        ensemble_q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                        - baseline_values
                    )
                    top_q_values = ensemble_q_values.topk(k=2, dim=-1).values
                    action_margin = top_q_values[..., 0] - top_q_values[..., 1]
                    update_scale = ensemble_q_values.new_tensor(self.action_value_update_scale)
                    confidence = action_margin / (action_margin + update_scale + 1e-8)
                    action_value_correction = (
                        self.action_value_correction_scale
                        * torch.tanh(ensemble_selected_advantages / (update_scale + 1e-8))
                        * confidence
                    )
                    self.last_action_value_update_scale = self.action_value_update_scale
                else:
                    normalized_q_advantages = torch.zeros_like(selected_q_advantages)
                    if self.action_value_per_agent_normalization:
                        agent_scales = []
                        for agent_index in range(self.num_agents):
                            agent_entries = valid_entries[:, agent_index].bool()
                            agent_advantages = selected_q_advantages[:, agent_index]
                            active_agent_advantages = agent_advantages[agent_entries]
                            if active_agent_advantages.numel() > 1:
                                agent_scale = active_agent_advantages.std(unbiased=False)
                                normalized_q_advantages[agent_entries, agent_index] = (
                                    (active_agent_advantages - active_agent_advantages.mean())
                                    / (agent_scale + 1e-8)
                                )
                                agent_scales.append(agent_scale)
                        if agent_scales:
                            self.last_action_value_agent_scale_spread = float(
                                torch.stack(agent_scales).std(unbiased=False).item()
                            )
                    else:
                        active_q_advantages = selected_q_advantages[valid_entries.bool()]
                        if active_q_advantages.numel() > 1:
                            normalized_q_advantages[valid_entries.bool()] = (
                                (active_q_advantages - active_q_advantages.mean())
                                / (active_q_advantages.std(unbiased=False) + 1e-8)
                            )
                        self.last_action_value_agent_scale_spread = 0.0
                    top_q_values = q_values.topk(k=2, dim=-1).values
                    action_margin = top_q_values[..., 0] - top_q_values[..., 1]
                    action_spread = q_values.std(dim=-1, unbiased=False)
                    confidence = action_margin / (action_margin + action_spread + 1e-8)
                    action_value_correction = (
                        self.action_value_correction_scale
                        * torch.tanh(normalized_q_advantages)
                        * confidence
                    )
                    self.last_action_value_update_scale = 0.0
                if reference_q_values is not None:
                    action_value_correction = action_value_correction * consensus_gate
                active_correction = action_value_correction[valid_entries.bool()]
                active_confidence = confidence[valid_entries.bool()]
                if active_correction.numel():
                    action_value_correction_std = float(
                        active_correction.std(unbiased=False).item()
                    )
                    action_value_confidence = float(active_confidence.mean().item())
                    if self.action_value_reference_head is not None:
                        action_value_consensus_rate = float(
                            consensus_gate[valid_entries.bool()].mean().item()
                        )
            # C46 SMDP-GAE remains primary; C55 additionally requires two
            # independent Q estimates to agree before a residual is applied.
            adv = adv + action_value_correction

        dense_obs = None
        dense_ret = None
        dense_entries = None
        dense_critic_decision_exclusion_rate = 0.0
        dense_critic_decision_target_gap = 0.0
        dense_args = (
            dense_critic_observations,
            dense_critic_rewards,
            dense_critic_values,
            dense_critic_dones,
        )
        if any(value is not None for value in dense_args):
            if not all(value is not None for value in dense_args):
                raise ValueError("dense critic updates require observations, rewards, values, and dones")
            dense_returns, _ = self.compute_gae(
                dense_critic_rewards, dense_critic_values, dense_critic_dones
            )
            dense_obs = torch.as_tensor(dense_critic_observations, dtype=torch.float32, device=device)
            dense_ret = torch.as_tensor(dense_returns, dtype=torch.float32, device=device)
            dense_entries = torch.ones_like(dense_ret)
            if dense_critic_decision_rows is not None:
                decision_rows = np.asarray(dense_critic_decision_rows, dtype=np.int64)
                if decision_rows.shape != dense_returns.shape:
                    raise ValueError(
                        "dense_critic_decision_rows must have shape [steps, num_agents]"
                    )
                decision_entries = decision_rows >= 0
                if np.any(decision_entries):
                    dense_step_indices, agent_indices = np.nonzero(decision_entries)
                    decision_targets = returns[
                        decision_rows[dense_step_indices, agent_indices], agent_indices
                    ]
                    dense_critic_decision_target_gap = float(np.mean(np.abs(
                        dense_returns[dense_step_indices, agent_indices] - decision_targets
                    )))
                    if self.dense_critic_exclude_decision_states:
                        dense_entries = torch.as_tensor(
                            (~decision_entries).astype(np.float32),
                            dtype=torch.float32,
                            device=device,
                        )
                        dense_critic_decision_exclusion_rate = float(
                            decision_entries.mean()
                        )
            dense_contexts = None if dense_pressure_contexts is None else torch.as_tensor(
                dense_pressure_contexts, dtype=torch.float32, device=device
            )
            if self.residual_pressure_context and dense_contexts is None:
                raise ValueError("dense_pressure_contexts are required for contextual residual updates")
        else:
            dense_contexts = None

        def masked_mean(values_to_reduce, weights):
            return (values_to_reduce * weights).sum() / weights.sum().clamp_min(1.0)

        def relative_value_loss(predicted_values, target_values, weights):
            active_counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            relative_values = predicted_values - (
                (predicted_values * weights).sum(dim=-1, keepdim=True) / active_counts
            )
            relative_returns = target_values - (
                (target_values * weights).sum(dim=-1, keepdim=True) / active_counts
            )
            relational_entries = weights * (active_counts > 1.0).float()
            return masked_mean(
                torch.nn.functional.smooth_l1_loss(
                    relative_values, relative_returns, reduction="none"
                ),
                relational_entries,
            )

        def apply_critic_update(critic, optimizer, decision_loss, dense_loss):
            optimizer.zero_grad(set_to_none=True)
            gradient_cosine = 0.0
            gradient_projected = 0.0
            if dense_loss is None:
                (self.value_coef * decision_loss).backward()
            else:
                critic_parameters = tuple(critic.parameters())
                decision_grads = torch.autograd.grad(
                    self.value_coef * decision_loss,
                    critic_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                dense_grads = torch.autograd.grad(
                    self.value_coef * dense_loss,
                    critic_parameters,
                    allow_unused=True,
                )
                decision_norm_sq = sum(
                    gradient.pow(2).sum() for gradient in decision_grads if gradient is not None
                )
                dense_norm_sq = sum(
                    gradient.pow(2).sum() for gradient in dense_grads if gradient is not None
                )
                gradient_dot = sum(
                    decision_gradient.mul(dense_gradient).sum()
                    for decision_gradient, dense_gradient in zip(decision_grads, dense_grads)
                    if decision_gradient is not None and dense_gradient is not None
                )
                if decision_norm_sq.item() > 0.0 and dense_norm_sq.item() > 0.0:
                    gradient_cosine = float((
                        gradient_dot / (decision_norm_sq.sqrt() * dense_norm_sq.sqrt() + 1e-8)
                    ).item())
                project_dense = gradient_dot.item() < 0.0 and decision_norm_sq.item() > 0.0
                if project_dense:
                    gradient_projected = 1.0
                projection_scale = gradient_dot / decision_norm_sq.clamp_min(1e-12)
                for parameter, decision_gradient, dense_gradient in zip(
                    critic_parameters, decision_grads, dense_grads
                ):
                    decision_component = (
                        torch.zeros_like(parameter)
                        if decision_gradient is None else decision_gradient
                    )
                    dense_component = (
                        torch.zeros_like(parameter)
                        if dense_gradient is None else dense_gradient
                    )
                    if project_dense:
                        dense_component = dense_component - projection_scale * decision_component
                    parameter.grad = decision_component + dense_component
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            optimizer.step()
            return gradient_cosine, gradient_projected

        frozen_q_bias = None
        if self.action_value_head is not None and self.action_value_reference_head is not None:
            rollout_base_logits = self._base_logits(obs, priors)
            rollout_q_logits, _ = self._soft_q_logits(rollout_base_logits, obs)
            frozen_q_bias = (rollout_q_logits - rollout_base_logits).detach()

        metrics = {}
        for _ in range(self.update_epochs):
            new_log, entropy, new_values = self.evaluate_actions(
                obs, act, masks, priors, temperatures,
                soft_q_bias_override=frozen_q_bias,
                pressure_context=contexts,
                **({"directional_features": self._update_directional_features} if hasattr(self, "_update_directional_features") and self._update_directional_features is not None else {}),
            )
            primary_values = self.critic(obs)
            residual_values = (
                None if self.residual_critic is None else self.residual_critic(obs, contexts)
            )
            ratio = torch.exp(new_log - old_log)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            actor_objective = masked_mean(
                torch.minimum(ratio * adv, clipped_ratio * adv), valid_entries
            )
            policy_actor_loss = -actor_objective
            entropy_actor_loss = -self.entropy_coef * masked_mean(entropy, valid_entries)
            actor_loss = policy_actor_loss + entropy_actor_loss
            critic_loss = masked_mean(
                torch.nn.functional.smooth_l1_loss(primary_values, ret, reduction="none"), valid_entries
            )
            if self.relative_value_coef > 0.0:
                # Preserve the team-return level while making vector-head value
                # differences reflect each junction's relative return.
                critic_loss = critic_loss + self.relative_value_coef * relative_value_loss(
                    primary_values, ret, valid_entries
                )
            decision_critic_loss = critic_loss
            dense_critic_loss = None
            if dense_obs is not None:
                dense_values = self.critic(dense_obs)
                dense_critic_loss = masked_mean(
                    torch.nn.functional.smooth_l1_loss(
                        dense_values, dense_ret, reduction="none"
                    ),
                    dense_entries,
                )
                if self.relative_value_coef > 0.0:
                    dense_critic_loss = dense_critic_loss + self.relative_value_coef * relative_value_loss(
                        dense_values, dense_ret, dense_entries
                    )
                critic_loss = critic_loss + dense_critic_loss

            residual_critic_loss = None
            residual_dense_critic_loss = None
            if residual_values is not None:
                residual_target = (ret - primary_values.detach()).detach()
                residual_critic_loss = masked_mean(
                    torch.nn.functional.smooth_l1_loss(
                        residual_values, residual_target, reduction="none"
                    ),
                    valid_entries,
                )
                if self.relative_value_coef > 0.0:
                    residual_critic_loss = residual_critic_loss + self.relative_value_coef * relative_value_loss(
                        residual_values, residual_target, valid_entries
                    )
                if dense_obs is not None:
                    residual_dense_values = self.residual_critic(dense_obs, dense_contexts)
                    primary_dense_values = self.critic(dense_obs).detach()
                    residual_dense_target = (dense_ret - primary_dense_values).detach()
                    residual_dense_critic_loss = masked_mean(
                        torch.nn.functional.smooth_l1_loss(
                            residual_dense_values,
                            residual_dense_target,
                            reduction="none",
                        ),
                        dense_entries,
                    )
                    if self.relative_value_coef > 0.0:
                        residual_dense_critic_loss = residual_dense_critic_loss + self.relative_value_coef * relative_value_loss(
                            residual_dense_values,
                            residual_dense_target,
                            dense_entries,
                        )

            action_value_loss = None
            action_value_reference_loss = None
            action_value_unselected_anchor_loss = None
            if self.action_value_head is not None:
                q_values, _ = self.action_value_head(obs, new_values.detach())
                selected_q_values = q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1)
                action_value_loss = masked_mean(
                    torch.nn.functional.smooth_l1_loss(
                        selected_q_values, ret, reduction="none"
                    ),
                    valid_entries,
                )
                if self.action_value_unselected_anchor:
                    unselected_actions = torch.ones_like(q_values, dtype=torch.bool)
                    unselected_actions.scatter_(-1, act.unsqueeze(-1), False)
                    unselected_entries = unselected_actions & valid_entries.bool().unsqueeze(-1)
                    q_anchor = self.critic(obs).detach().unsqueeze(-1).expand_as(q_values)
                    action_value_unselected_anchor_loss = masked_mean(
                        torch.nn.functional.smooth_l1_loss(
                            q_values, q_anchor, reduction="none"
                        ),
                        unselected_entries,
                    )
                    action_value_loss = action_value_loss + action_value_unselected_anchor_loss
                elif (
                    self.action_value_unselected_conservative_anchor
                    or self.action_value_unselected_behavior_anchor
                ):
                    unselected_actions = torch.ones_like(q_values, dtype=torch.bool)
                    unselected_actions.scatter_(-1, act.unsqueeze(-1), False)
                    unselected_entries = unselected_actions & valid_entries.bool().unsqueeze(-1)
                    q_anchor = self.critic(obs).detach().unsqueeze(-1).expand_as(q_values)
                    if self.action_value_unselected_behavior_anchor:
                        selected_q_anchor = selected_q_values.detach().unsqueeze(-1)
                    else:
                        selected_q_anchor = q_anchor
                    positive_unselected_advantages = (
                        q_values - selected_q_anchor
                    ).clamp_min(0.0)
                    action_value_unselected_anchor_loss = masked_mean(
                        positive_unselected_advantages.pow(2),
                        unselected_entries,
                    )
                    action_value_loss = action_value_loss + action_value_unselected_anchor_loss
                if self.action_value_reference_head is not None:
                    reference_q_values, _ = self.action_value_reference_head(
                        obs, new_values.detach()
                    )
                    reference_selected_q_values = reference_q_values.gather(
                        -1, act.unsqueeze(-1)
                    ).squeeze(-1)
                    action_value_reference_loss = masked_mean(
                        torch.nn.functional.smooth_l1_loss(
                            reference_selected_q_values, ret, reduction="none"
                        ),
                        valid_entries,
                    )
                    if action_value_unselected_anchor_loss is not None:
                        reference_unselected_anchor_loss = masked_mean(
                            torch.nn.functional.smooth_l1_loss(
                                reference_q_values, q_anchor, reduction="none"
                            ),
                            unselected_entries,
                        )
                        action_value_reference_loss = (
                            action_value_reference_loss + reference_unselected_anchor_loss
                        )
                    if self.soft_q_margin_calibration or self.action_value_margin_calibration:
                        ensemble_advantages = 0.5 * (
                            q_values.detach() + reference_q_values.detach()
                        ) - new_values.detach().unsqueeze(-1)
                        active_magnitudes = ensemble_advantages.abs()[valid_entries.bool()]
                        if active_magnitudes.numel():
                            batch_scale = float(active_magnitudes.mean().item())
                            if self.soft_q_margin_calibration:
                                if self.action_value_margin_scale <= 0.0:
                                    self.action_value_margin_scale = batch_scale
                                else:
                                    self.action_value_margin_scale = (
                                        0.95 * self.action_value_margin_scale + 0.05 * batch_scale
                                    )
                            if self.action_value_margin_calibration:
                                if self.action_value_update_scale <= 0.0:
                                    self.action_value_update_scale = batch_scale
                                else:
                                    self.action_value_update_scale = (
                                        0.95 * self.action_value_update_scale + 0.05 * batch_scale
                                    )

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_gradient_cosine = 0.0
            actor_gradient_projected = 0.0
            actor_parameters = tuple(self.actor.parameters())
            policy_grads = torch.autograd.grad(
                policy_actor_loss,
                actor_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            entropy_grads = torch.autograd.grad(
                entropy_actor_loss,
                actor_parameters,
                allow_unused=True,
            )
            policy_norm_sq = sum(
                gradient.pow(2).sum() for gradient in policy_grads if gradient is not None
            )
            entropy_norm_sq = sum(
                gradient.pow(2).sum() for gradient in entropy_grads if gradient is not None
            )
            actor_gradient_dot = sum(
                policy_gradient.mul(entropy_gradient).sum()
                for policy_gradient, entropy_gradient in zip(policy_grads, entropy_grads)
                if policy_gradient is not None and entropy_gradient is not None
            )
            if policy_norm_sq.item() > 0.0 and entropy_norm_sq.item() > 0.0:
                actor_gradient_cosine = float((
                    actor_gradient_dot / (policy_norm_sq.sqrt() * entropy_norm_sq.sqrt() + 1e-8)
                ).item())
            project_entropy = actor_gradient_dot.item() < 0.0 and policy_norm_sq.item() > 0.0
            if project_entropy:
                actor_gradient_projected = 1.0
            entropy_projection_scale = actor_gradient_dot / policy_norm_sq.clamp_min(1e-12)
            for parameter, policy_gradient, entropy_gradient in zip(
                actor_parameters, policy_grads, entropy_grads
            ):
                policy_component = (
                    torch.zeros_like(parameter)
                    if policy_gradient is None else policy_gradient
                )
                entropy_component = (
                    torch.zeros_like(parameter)
                    if entropy_gradient is None else entropy_gradient
                )
                if project_entropy:
                    entropy_component = entropy_component - entropy_projection_scale * policy_component
                parameter.grad = policy_component + entropy_component
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            critic_gradient_cosine, critic_gradient_projected = apply_critic_update(
                self.critic,
                self.critic_optimizer,
                decision_critic_loss,
                dense_critic_loss,
            )
            residual_critic_gradient_cosine = 0.0
            residual_critic_gradient_projected = 0.0
            if residual_critic_loss is not None:
                residual_critic_gradient_cosine, residual_critic_gradient_projected = apply_critic_update(
                    self.residual_critic,
                    self.residual_critic_optimizer,
                    residual_critic_loss,
                    residual_dense_critic_loss,
                )

            if action_value_loss is not None:
                self.action_value_optimizer.zero_grad(set_to_none=True)
                action_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.action_value_head.parameters(), self.max_grad_norm
                )
                self.action_value_optimizer.step()
            if action_value_reference_loss is not None:
                self.action_value_reference_optimizer.zero_grad(set_to_none=True)
                action_value_reference_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.action_value_reference_head.parameters(), self.max_grad_norm
                )
                self.action_value_reference_optimizer.step()

            metrics = {
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "entropy": float(entropy.mean().item()),
                "approx_kl": float((old_log - new_log).mean().item()),
                "actor_gradient_cosine": actor_gradient_cosine,
                "actor_gradient_projected": actor_gradient_projected,
            "critic_gradient_cosine": critic_gradient_cosine,
            "critic_gradient_projected": critic_gradient_projected,
            "q_bias_frozen": float(frozen_q_bias is not None),
            "residual_critic": float(self.residual_critic is not None),
            "adaptive_residual_consensus": float(self.adaptive_residual_consensus),
            "adaptive_residual_energy_mix": float(self.adaptive_residual_energy_mix),
            "dense_critic_decision_exclusion_rate": dense_critic_decision_exclusion_rate,
            "dense_critic_decision_target_gap": dense_critic_decision_target_gap,
            "residual_consensus_detach": float(self.residual_consensus_detach),
            "residual_global_mix": self.last_residual_global_mix,
            "residual_critic_gradient_cosine": residual_critic_gradient_cosine,
            "residual_critic_gradient_projected": residual_critic_gradient_projected,
        }
            if action_value_loss is not None:
                metrics["action_value_loss"] = float(action_value_loss.item())
                metrics["action_value_confidence"] = action_value_confidence
                metrics["action_value_correction_std"] = action_value_correction_std
                metrics["action_value_consensus_rate"] = action_value_consensus_rate
                metrics["action_value_unselected_anchor_loss"] = float(
                    0.0 if action_value_unselected_anchor_loss is None
                    else action_value_unselected_anchor_loss.item()
                )
                if action_value_reference_loss is not None:
                    metrics["action_value_reference_loss"] = float(
                        action_value_reference_loss.item()
                    )
        return metrics

    def _config(self):
        return {
            'family': self.family, 'num_agents': self.num_agents,
            'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
            'actor_type': 'independent' if self.independent_actor else 'shared',
            'prior_mode': self.prior_mode, 'action_prior_scale': self.action_prior_scale,
            'pressure_observation': self.pressure_observation,
            'soft_q': self.soft_q_execution,
        }

    def _validate_config(self, state):
        config = state.get('config')
        if config is None:
            return
        expected = self._config()
        for key in (
            'family', 'num_agents', 'obs_dim', 'action_dim', 'actor_type',
            'prior_mode', 'action_prior_scale', 'pressure_observation', 'soft_q',
        ):
            if config.get(key) != expected[key]:
                raise RuntimeError(
                    f"GC-MAPPO checkpoint config mismatch for {key}: "
                    f"saved={config.get(key)!r}, current={expected[key]!r}."
                )

    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_type": "independent" if self.independent_actor else "shared",
            "config": self._config(),
        }
        if self.residual_critic is not None:
            checkpoint["residual_critic"] = self.residual_critic.state_dict()
        if self.action_value_head is not None:
            checkpoint["action_value_head"] = self.action_value_head.state_dict()
        if self.action_value_reference_head is not None:
            checkpoint["action_value_reference_head"] = (
                self.action_value_reference_head.state_dict()
            )
        torch.save(checkpoint, path)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'actor_type': 'independent' if self.independent_actor else 'shared',
            'config': self._config(),
            'critic': self.critic.state_dict(),
            'residual_critic': self.residual_critic.state_dict() if self.residual_critic is not None else None,
            'action_value_head': self.action_value_head.state_dict() if self.action_value_head is not None else None,
            'action_value_reference_head': (
                self.action_value_reference_head.state_dict()
                if self.action_value_reference_head is not None else None
            ),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'residual_critic_optimizer': (
                self.residual_critic_optimizer.state_dict()
                if self.residual_critic_optimizer is not None else None
            ),
            'action_value_optimizer': (
                self.action_value_optimizer.state_dict()
                if self.action_value_optimizer is not None else None
            ),
            'action_value_reference_optimizer': (
                self.action_value_reference_optimizer.state_dict()
                if self.action_value_reference_optimizer is not None else None
            ),
            'entropy_coef': self.entropy_coef,
            'clip_ratio': self.clip_ratio,
        }

    def load_state_dict(self, state):
        self._validate_config(state)
        expected_actor_type = 'independent' if self.independent_actor else 'shared'
        saved_actor_type = state.get('actor_type', 'shared')
        if saved_actor_type != expected_actor_type:
            raise RuntimeError(
                f"Resume actor_type mismatch: saved as '{saved_actor_type}', "
                f"but current agent is '{expected_actor_type}'."
            )
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        for name in ('residual_critic', 'action_value_head', 'action_value_reference_head'):
            module = getattr(self, name)
            if module is not None and state.get(name) is not None:
                module.load_state_dict(state[name])
        optimizer_names = (
            'actor_optimizer', 'critic_optimizer', 'residual_critic_optimizer',
            'action_value_optimizer', 'action_value_reference_optimizer',
        )
        for name in optimizer_names:
            optimizer = getattr(self, name)
            if optimizer is not None and state.get(name) is not None:
                optimizer.load_state_dict(state[name])
        self.entropy_coef = float(state.get('entropy_coef', self.entropy_coef))
        self.clip_ratio = float(state.get('clip_ratio', self.clip_ratio))

    def load(self, path, map_location=device):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self._validate_config(checkpoint)
        expected_actor_type = "independent" if self.independent_actor else "shared"
        saved_actor_type = checkpoint.get("actor_type", "shared")
        if saved_actor_type != expected_actor_type:
            raise RuntimeError(
                f"Checkpoint actor_type mismatch: saved as '{saved_actor_type}', "
                f"but current agent is '{expected_actor_type}'. "
                f"Cannot load checkpoint across different actor architectures."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.residual_critic is not None and "residual_critic" in checkpoint:
            self.residual_critic.load_state_dict(checkpoint["residual_critic"])
        if self.action_value_head is not None and "action_value_head" in checkpoint:
            self.action_value_head.load_state_dict(checkpoint["action_value_head"])
        if (
            self.action_value_reference_head is not None
            and "action_value_reference_head" in checkpoint
        ):
            self.action_value_reference_head.load_state_dict(
                checkpoint["action_value_reference_head"]
            )
