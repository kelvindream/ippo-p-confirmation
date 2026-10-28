"""Shared-policy actor-critic trained with generalized advantage estimation."""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class GAEActorCriticNetwork(nn.Module):
    """Shared feature encoder with policy and value heads."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(256, output_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        features = self.feature_layer(x)
        return self.actor(features), self.critic(features)


class GAEActorCritic:
    """Synchronous shared-policy actor-critic with GAE and one joint optimizer step."""

    def __init__(self, input_dim, output_dim, mode='train', gamma=0.99,
                 learning_rate=0.0003, gae_lambda=0.95):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.mode = mode
        self.steps_done = 0
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = gae_lambda
        self.policy_net = GAEActorCriticNetwork(input_dim, output_dim).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.replay_buffer = []

    def select_action(self, state, steps_done=None):
        if steps_done is not None:
            self.steps_done = steps_done
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor)
            if self.mode == 'train':
                return torch.distributions.Categorical(probs=action_probs).sample().item()
            return torch.argmax(action_probs, dim=-1).item()

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append(Transition(
            torch.from_numpy(state).float().to(device), torch.tensor([action]).to(device),
            torch.from_numpy(next_state).float().to(device),
            torch.tensor([reward], dtype=torch.float32).to(device),
            torch.tensor([done], dtype=torch.bool).to(device),
        ))

    def optimize_model(self):
        if not self.replay_buffer:
            return 0
        batch = Transition(*zip(*self.replay_buffer))
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).view(-1).long()
        next_state_batch = torch.stack(batch.next_state)
        rewards = torch.stack(batch.reward).view(-1)
        dones = torch.stack(batch.done).view(-1)
        action_probs, values = self.policy_net(state_batch)
        values = values.squeeze(-1)
        with torch.no_grad():
            _, next_values = self.policy_net(next_state_batch)
            next_values = next_values.squeeze(-1)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for t in reversed(range(len(rewards))):
            not_done = (~dones[t]).float()
            bootstrap = next_values[t] if t == len(rewards) - 1 else values[t + 1].detach()
            delta = rewards[t] + self.gamma * bootstrap * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages.detach() + values.detach()
        normalized_advantages = (
            (advantages - advantages.mean())
            / (advantages.std(unbiased=False) + 1e-8)
        )
        distribution = torch.distributions.Categorical(probs=action_probs)
        actor_loss = -(distribution.log_prob(action_batch) * normalized_advantages.detach()).mean()
        actor_loss -= self.entropy_coef * distribution.entropy().mean()
        critic_loss = self.value_coef * F.smooth_l1_loss(values, returns)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.replay_buffer.clear()
        return total_loss.item()

    def update_target_network(self, steps_done=None):
        return None

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def state_dict(self):
        return {
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'replay_buffer': self.replay_buffer,
        }

    def load_state_dict(self, state):
        self.policy_net.load_state_dict(state['policy_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps_done = int(state.get('steps_done', 0))
        self.replay_buffer = state.get('replay_buffer', [])

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def get_actor_state_dict(self):
        return self.policy_net.state_dict()

    def get_critic_state_dict(self):
        return {name: value for name, value in self.policy_net.state_dict().items() if 'critic' in name}
