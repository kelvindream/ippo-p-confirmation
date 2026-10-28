"""Double DQN using per-phase pressure as the shared intersection state."""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _PressureQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states):
        return self.network(states)


class PressureStateDoubleDQN:
    """Parameter-shared Double DQN trained on pressure-state observations."""

    mechanism_name = "Pressure-State Double DQN"
    formal_name = "PressLight"
    reproduction_scope = "pressure-state, pressure reward, and Double-DQN target"


    def __init__(self, input_dim, action_dim, gamma=0.99, learning_rate=1e-4,
                 batch_size=64, capacity=20000, target_interval=200,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=20000):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_interval = target_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.learn_steps = 0
        self.online = _PressureQNetwork(input_dim, action_dim).to(device)
        self.target = _PressureQNetwork(input_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=capacity)

    def _epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)

    @torch.no_grad()
    def select_actions(self, pressure_states, deterministic=False):
        states = torch.as_tensor(pressure_states, dtype=torch.float32, device=device)
        values = self.online(states)
        self.steps += 1
        actions = torch.argmax(values, dim=-1).cpu().numpy().tolist()
        if not deterministic:
            epsilon = self._epsilon()
            actions = [random.randrange(self.action_dim) if random.random() < epsilon else action for action in actions]
        return actions

    def store(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay.append((np.asarray(state, dtype=np.float32), int(action), float(reward),
                                np.asarray(next_state, dtype=np.float32), float(done)))

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        state = torch.as_tensor(states, dtype=torch.float32, device=device)
        action = torch.as_tensor(actions, dtype=torch.long, device=device)
        reward = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        done = torch.as_tensor(dones, dtype=torch.float32, device=device)
        q = self.online(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = self.online(next_state).argmax(dim=1, keepdim=True)
            next_q = self.target(next_state).gather(1, next_action).squeeze(1)
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
