"""Standard MAPPO baseline without graph message passing."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations):
        return self.network(observations)


class _CentralCritic(nn.Module):
    def __init__(self, num_agents, obs_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_agents * obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations):
        return self.network(observations.reshape(observations.shape[0], -1)).squeeze(-1)


class MAPPO:
    """CTDE PPO baseline with a shared MLP actor and centralized critic."""

    def __init__(self, num_agents, obs_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01,
                 value_coef=0.5, update_epochs=5, max_grad_norm=0.5, hidden_dim=128):
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
        self.actor = _Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = _CentralCritic(num_agents, obs_dim, hidden_dim * 2).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _observations(self, observations):
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.num_agents, self.obs_dim):
            raise ValueError("observations must have shape [num_agents, obs_dim]")
        return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)

    @torch.no_grad()
    def select_actions(self, observations, deterministic=False):
        obs = self._observations(observations)
        distribution = Categorical(logits=self.actor(obs))
        actions = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return (actions.squeeze(0).cpu().numpy().tolist(),
                distribution.log_prob(actions).squeeze(0).cpu().numpy(),
                float(self.critic(obs).item()))

    def _gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for step in range(len(rewards) - 1, -1, -1):
            next_value = 0.0 if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.gamma * next_value * (1.0 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, observations, actions, old_log_probs, rewards, values, dones):
        returns, advantages = self._gae(np.asarray(rewards, dtype=np.float32),
                                        np.asarray(values, dtype=np.float32),
                                        np.asarray(dones, dtype=np.float32))
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        act = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_log = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        for _ in range(self.update_epochs):
            distribution = Categorical(logits=self.actor(obs))
            new_log = distribution.log_prob(act)
            ratio = torch.exp(new_log - old_log)
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            actor_loss = -torch.minimum(ratio * adv.unsqueeze(-1), clipped * adv.unsqueeze(-1)).mean()
            actor_loss -= self.entropy_coef * distribution.entropy().mean()
            critic_loss = torch.nn.functional.smooth_l1_loss(self.critic(obs), ret)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            (self.value_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'entropy_coef': self.entropy_coef, 'clip_ratio': self.clip_ratio,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.entropy_coef = float(state.get('entropy_coef', self.entropy_coef))
        self.clip_ratio = float(state.get('clip_ratio', self.clip_ratio))

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
