"""Independent PPO baseline for decentralized traffic-signal control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ippo_parameter_count(obs_dim, action_dim, hidden_dim, num_agents=1):
    actor = obs_dim * hidden_dim + hidden_dim
    actor += hidden_dim * hidden_dim + hidden_dim
    actor += hidden_dim * action_dim + action_dim
    critic = obs_dim * hidden_dim + hidden_dim
    critic += hidden_dim * hidden_dim + hidden_dim
    critic += hidden_dim + 1
    return int(num_agents * (actor + critic))


def matched_hidden_dim(obs_dim, action_dim, target_parameter_count, num_agents=1):
    candidates = range(16, 513)
    return min(
        candidates,
        key=lambda hidden_dim: abs(
            ippo_parameter_count(obs_dim, action_dim, hidden_dim, num_agents)
            - target_parameter_count
        ),
    )


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim),
    )


class IPPO:
    """Independent PPO actor-critic for each intersection without shared communication."""

    def __init__(self, num_agents, obs_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, learning_rate=3e-4, entropy_coef=0.01,
                 value_coef=0.5, update_epochs=5, max_grad_norm=0.5,
                 hidden_dim=128, actor_trainable=True, action_prior_scale=0.0,
                 prior_mode='none', policy_temperature=1.0):
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
        self.hidden_dim = int(hidden_dim)
        self.action_prior_scale = float(action_prior_scale)
        self.prior_mode = str(prior_mode)
        self.policy_temperature = float(policy_temperature)
        self.actors = nn.ModuleList([_mlp(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)]).to(device)
        self.critics = nn.ModuleList([_mlp(obs_dim, 1, hidden_dim) for _ in range(num_agents)]).to(device)
        self.actor_trainable = bool(actor_trainable)
        for parameter in self.actors.parameters():
            parameter.requires_grad_(self.actor_trainable)
        self.actor_optimizer = torch.optim.Adam(
            [parameter for parameter in self.actors.parameters() if parameter.requires_grad], lr=learning_rate
        ) if self.actor_trainable else None
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False, action_priors=None, policy_temperature=1.0):
        obs = np.asarray(observations, dtype=np.float32)
        actions, log_probs, values = [], [], []
        for agent_idx in range(self.num_agents):
            state = torch.as_tensor(obs[agent_idx], dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.actors[agent_idx](state)
            if action_priors is not None:
                prior = torch.as_tensor(
                    np.asarray(action_priors, dtype=np.float32)[agent_idx],
                    dtype=torch.float32, device=device,
                )
                logits = logits + self.action_prior_scale * prior
            policy_logits = logits / max(float(policy_temperature), 1e-6)
            distribution = Categorical(logits=policy_logits)
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                action = distribution.sample()
            actions.append(int(action.item()))
            log_probs.append(float(distribution.log_prob(action).item()))
            values.append(float(self.critics[agent_idx](state).item()))
        return actions, np.asarray(log_probs, dtype=np.float32), np.asarray(values, dtype=np.float32)

    def _gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in range(len(rewards) - 1, -1, -1):
            next_value = np.zeros(self.num_agents, dtype=np.float32) if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.gamma * next_value * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean(axis=0, keepdims=True)) / (advantages.std(axis=0, keepdims=True) + 1e-8)
        return returns, advantages

    def update(self, observations, actions, old_log_probs, rewards, values, dones,
               action_priors=None, policy_temperatures=None):
        returns, advantages = self._gae(np.asarray(rewards, dtype=np.float32),
                                        np.asarray(values, dtype=np.float32),
                                        np.asarray(dones, dtype=np.float32)[:, None])
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        act = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        priors = None if action_priors is None else torch.as_tensor(
            action_priors, dtype=torch.float32, device=device
        )
        temperatures = None if policy_temperatures is None else torch.as_tensor(
            policy_temperatures, dtype=torch.float32, device=device
        )
        for _ in range(self.update_epochs):
            actor_losses, critic_losses = [], []
            for agent_idx in range(self.num_agents):
                logits = self.actors[agent_idx](obs[:, agent_idx])
                if priors is not None:
                    logits = logits + self.action_prior_scale * priors[:, agent_idx]
                if temperatures is not None:
                    logits = logits / temperatures.clamp_min(1e-6).view(-1, 1)
                distribution = Categorical(logits=logits)
                new_log = distribution.log_prob(act[:, agent_idx])
                ratio = torch.exp(new_log - old_log[:, agent_idx])
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                actor_losses.append(-torch.minimum(ratio * adv[:, agent_idx], clipped * adv[:, agent_idx]).mean()
                                    - self.entropy_coef * distribution.entropy().mean())
                critic_losses.append(torch.nn.functional.smooth_l1_loss(
                    self.critics[agent_idx](obs[:, agent_idx]).squeeze(-1), ret[:, agent_idx]))
            if self.actor_trainable:
                self.actor_optimizer.zero_grad(set_to_none=True)
                torch.stack(actor_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            (self.value_coef * torch.stack(critic_losses).mean()).backward()
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def initialize_actors(self, path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state = checkpoint.get('agent', checkpoint)
        if isinstance(state, dict) and 'base' in state:
            state = state['base']
        if 'actors' not in state:
            raise ValueError(f'IPPO actor checkpoint has no actors: {path}')
        self.actors.load_state_dict(state['actors'])

    def _config(self):
        return {
            'family': 'IPPO', 'num_agents': self.num_agents, 'obs_dim': self.obs_dim,
            'action_dim': self.action_dim, 'actor_type': 'independent_local',
            'prior_mode': self.prior_mode, 'action_prior_scale': self.action_prior_scale,
            'pressure_observation': False, 'soft_q': False,
            'policy_temperature': self.policy_temperature,
        }

    def _validate_config(self, state):
        config = state.get('config')
        if config is None:
            return
        expected_config = self._config()
        for key, expected in (
            ('family', expected_config['family']),
            ('num_agents', self.num_agents), ('obs_dim', self.obs_dim),
            ('action_dim', self.action_dim),
            ('actor_type', expected_config['actor_type']),
            ('prior_mode', self.prior_mode),
            ('action_prior_scale', self.action_prior_scale),
            ('pressure_observation', expected_config['pressure_observation']),
            ('soft_q', expected_config['soft_q']),
            ('policy_temperature', self.policy_temperature),
        ):
            if config.get(key) != expected:
                raise RuntimeError(
                    f"IPPO checkpoint config mismatch for {key}: "
                    f"saved={config.get(key)!r}, current={expected!r}."
                )

    def save(self, path):
        torch.save({
            'actors': self.actors.state_dict(), 'critics': self.critics.state_dict(),
            'config': self._config(),
        }, path)

    def state_dict(self):
        return {
            'actors': self.actors.state_dict(), 'critics': self.critics.state_dict(),
            'config': self._config(),
            'actor_optimizer': self.actor_optimizer.state_dict() if self.actor_optimizer is not None else None,
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'entropy_coef': self.entropy_coef, 'clip_ratio': self.clip_ratio,
        }

    def load_state_dict(self, state):
        self._validate_config(state)
        self.actors.load_state_dict(state['actors'])
        self.critics.load_state_dict(state['critics'])
        if self.actor_optimizer is not None and state.get('actor_optimizer') is not None:
            self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.entropy_coef = float(state.get('entropy_coef', self.entropy_coef))
        self.clip_ratio = float(state.get('clip_ratio', self.clip_ratio))

    def load(self, path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self._validate_config(checkpoint)
        self.actors.load_state_dict(checkpoint['actors'])
        self.critics.load_state_dict(checkpoint['critics'])
