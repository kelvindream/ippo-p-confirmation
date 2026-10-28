"""Graph-attention Double DQN with one action-value network shared across intersections."""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GraphQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, heads=4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.local = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, observations, adjacency):
        features = self.local(observations)
        batch, agents, hidden = features.shape
        q = self.query(features).view(batch, agents, self.heads, self.head_dim)
        k = self.key(features).view(batch, agents, self.heads, self.head_dim)
        v = self.value(features).view(batch, agents, self.heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bijh", q, k) / np.sqrt(self.head_dim)
        mask = adjacency.bool().clone()
        mask.fill_diagonal_(True)
        scores = scores.masked_fill(~mask.view(1, agents, agents, 1), -1e9)
        attention = torch.softmax(scores, dim=2)
        messages = torch.einsum("bijh,bjhd->bihd", attention, v).reshape(batch, agents, hidden)
        return self.q_head(torch.cat([features, messages], dim=-1))


class GraphAttentionDoubleDQN:
    """Parameter-shared Double DQN using graph attention for neighbor messages."""

    mechanism_name = "Graph-Attention Double DQN"
    formal_name = "CoLight"
    reproduction_scope = "adjacency multi-head attention neighbor communication and Double-DQN"


    def __init__(self, num_agents, obs_dim, action_dim, adjacency_matrix, gamma=0.99,
                 learning_rate=1e-4, batch_size=64, capacity=20000, target_interval=200,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_interval = target_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.learn_steps = 0
        self.adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.bool, device=device)
        self.online = _GraphQNetwork(obs_dim, action_dim).to(device)
        self.target = _GraphQNetwork(obs_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=capacity)

    def _epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)

    @torch.no_grad()
    def select_actions(self, states, deterministic=False):
        state = torch.as_tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = self.online(state, self.adjacency).squeeze(0)
        self.steps += 1
        if not deterministic and random.random() < self._epsilon():
            return [random.randrange(self.action_dim) for _ in range(self.num_agents)]
        return torch.argmax(q_values, dim=-1).cpu().numpy().tolist()

    def store(self, states, actions, rewards, next_states, done):
        self.replay.append((np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64),
                            np.asarray(rewards, dtype=np.float32), np.asarray(next_states, dtype=np.float32), float(done)))

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        state = torch.as_tensor(states, dtype=torch.float32, device=device)
        action = torch.as_tensor(actions, dtype=torch.long, device=device)
        reward = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        done = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1, 1)
        q = self.online(state, self.adjacency).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_actions = self.online(next_state, self.adjacency).argmax(dim=-1, keepdim=True)
            next_q = self.target(next_state, self.adjacency).gather(-1, next_actions).squeeze(-1)
            target = reward + self.gamma * (1.0 - done) * next_q
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 0.5)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({'online': self.online.state_dict(), 'target': self.target.state_dict()}, path)

    def state_dict(self):
        return {
            'online': self.online.state_dict(), 'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'steps': self.steps,
            'learn_steps': self.learn_steps, 'replay': list(self.replay),
        }

    def load_state_dict(self, state):
        self.online.load_state_dict(state['online'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps = int(state.get('steps', 0))
        self.learn_steps = int(state.get('learn_steps', 0))
        self.replay.clear()
        self.replay.extend(state.get('replay', []))

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])
