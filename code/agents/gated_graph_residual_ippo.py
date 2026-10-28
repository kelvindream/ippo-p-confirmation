"""Exploratory bottleneck-aware gated graph-residual IPPO candidates."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.ippo import IPPO, device


class _MessageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

    def forward(self, messages):
        return self.net(messages)


class GatedGraphResidualIPPO:
    """Exploratory local IPPO logits plus bounded gated one-hop residual."""

    def __init__(self, num_agents, local_obs_dim, action_dim, adjacency, hidden_dim=128,
                 residual_limit=0.5, freeze_actor=False, learning_rate=3e-4,
                 update_epochs=5, clip_ratio=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5):
        if residual_limit <= 0:
            raise ValueError('residual_limit must be positive')
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.shape != (num_agents, num_agents):
            raise ValueError('adjacency must have shape (num_agents, num_agents)')
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.residual_limit = float(residual_limit)
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
        self.base = IPPO(num_agents, local_obs_dim, action_dim, hidden_dim=hidden_dim,
                         actor_trainable=not freeze_actor, learning_rate=learning_rate,
                         update_epochs=update_epochs, clip_ratio=clip_ratio,
                         entropy_coef=entropy_coef, value_coef=value_coef,
                         max_grad_norm=max_grad_norm)
        self.message_encoder = _MessageEncoder(local_obs_dim, hidden_dim).to(device)
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + local_obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            for _ in range(num_agents)
        ]).to(device)
        self.residual_heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_agents)]).to(device)
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.gate_bias = nn.Parameter(torch.full((num_agents, 1), 1.0, device=device))
        params = list(self.message_encoder.parameters()) + list(self.gates.parameters()) + \
                 list(self.residual_heads.parameters()) + [self.gate_bias]
        if not freeze_actor:
            params += list(self.base.actors.parameters())
        self.residual_optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.freeze_actor = bool(freeze_actor)

    def initialize_from_ippo(self, path):
        self.base.initialize_actors(path)

    def _logits(self, observations):
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=device)
        adjacency = self.adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        neighbor_count = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        neighbor_mean = adjacency @ obs / neighbor_count
        messages = self.message_encoder(neighbor_mean)
        logits = []
        for i in range(self.num_agents):
            base_logits = self.base.actors[i](obs[i:i + 1]).squeeze(0)
            gate_input = torch.cat([obs[i], messages[i]], dim=-1)
            gate = torch.sigmoid(self.gates[i](gate_input).squeeze(-1) + self.gate_bias[i])
            residual = self.residual_limit * torch.tanh(self.residual_heads[i](messages[i]))
            logits.append(base_logits + gate * residual)
        return torch.stack(logits)

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False):
        logits = self._logits(observations)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        values = torch.stack([
            self.base.critics[i](torch.as_tensor(observations[i], dtype=torch.float32, device=device)).squeeze()
            for i in range(self.num_agents)
        ])
        return (actions.cpu().numpy().astype(int).tolist(),
                distribution.log_prob(actions).cpu().numpy(), values.cpu().numpy())

    def update(self, observations, actions, old_log_probs, rewards, values, dones):
        obs_np = np.asarray(observations, dtype=np.float32)
        returns, advantages = self.base._gae(np.asarray(rewards, dtype=np.float32),
                                               np.asarray(values, dtype=np.float32),
                                               np.asarray(dones, dtype=np.float32)[:, None])
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        act = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        for _ in range(self.base.update_epochs):
            log_probs, entropies = [], []
            for step in range(len(obs)):
                dist = Categorical(logits=self._logits(obs[step].detach().cpu().numpy()))
                log_probs.append(dist.log_prob(act[step]))
                entropies.append(dist.entropy())
            new_log = torch.stack(log_probs)
            ratio = torch.exp(new_log - old_log)
            clipped = ratio.clamp(1 - self.base.clip_ratio, 1 + self.base.clip_ratio)
            actor_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
            actor_loss -= self.base.entropy_coef * torch.stack(entropies).mean()
            critic_losses = [torch.nn.functional.smooth_l1_loss(
                self.base.critics[i](obs[:, i]).squeeze(-1), ret[:, i]) for i in range(self.num_agents)]
            loss = actor_loss + self.base.value_coef * torch.stack(critic_losses).mean()
            self.residual_optimizer.zero_grad(set_to_none=True)
            self.base.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.residual_optimizer.param_groups[0]['params'], self.base.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.base.critics.parameters(), self.base.max_grad_norm)
            self.residual_optimizer.step()
            self.base.critic_optimizer.step()

    def state_dict(self):
        return {'base': self.base.state_dict(), 'message_encoder': self.message_encoder.state_dict(),
                'gates': self.gates.state_dict(), 'residual_heads': self.residual_heads.state_dict(),
                'gate_bias': self.gate_bias.detach().cpu(), 'freeze_actor': self.freeze_actor,
                'residual_limit': self.residual_limit,
                'residual_optimizer': self.residual_optimizer.state_dict()}

    def load_state_dict(self, state):
        self.base.load_state_dict(state['base'])
        self.message_encoder.load_state_dict(state['message_encoder'])
        self.gates.load_state_dict(state['gates'])
        self.residual_heads.load_state_dict(state['residual_heads'])
        with torch.no_grad():
            self.gate_bias.copy_(state['gate_bias'].to(device))
        self.residual_optimizer.load_state_dict(state['residual_optimizer'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=False))
