import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DuelingDoubleDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU())
        self.value_layer = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 1))
        self.advantage_layer = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, output_dim))

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DuelingDoubleDQNAgent:
    """Dueling Double DQN with parameters and replay updates shared across intersections."""

    def __init__(self, mode, replay_buffer, target_update, gamma, eps_start, eps_end,
                 eps_decay, input_dim, output_dim, batch_size=32, network_file='',
                 learning_rate=0.0001, warmup_transitions=None, train_freq=1):
        self.mode = mode
        self.replay_buffer = replay_buffer
        self.target_update = target_update
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.min_replay_size = max(batch_size, warmup_transitions or batch_size)
        self.train_freq = max(1, train_freq)
        self.output_dim = output_dim
        self.policy_net = DuelingDoubleDQNNetwork(input_dim, output_dim).to(device)
        self.target_net = DuelingDoubleDQNNetwork(input_dim, output_dim).to(device)
        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, weight_decay=0.00001)
        self.learn_steps = 0

    def select_action(self, state, steps_done, invalid_action=None):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-steps_done / self.eps_decay)
        if self.mode == 'train' and random.random() < eps_threshold:
            return random.randint(0, self.output_dim - 1)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            q_values = self.policy_net(state_tensor)
            action = q_values.max(0)[1].item()
        if invalid_action:
            for idx in torch.argsort(q_values, descending=True):
                if idx.item() != action:
                    return idx.item()
        return action

    def optimize_model(self):
        self.learn_steps += 1
        if self.learn_steps % self.train_freq != 0 or len(self.replay_buffer) < self.min_replay_size:
            return 0
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).view(-1, 1).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        reward_batch = torch.cat(batch.reward).view(-1, 1).to(device)
        done_batch = torch.cat(batch.done).view(-1, 1).to(device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].view(-1, 1)
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions)
            expected = reward_batch + self.gamma * next_state_values * (~done_batch).float()
        loss = F.smooth_l1_loss(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self, steps_done):
        if steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def state_dict(self):
        return {'policy_net': self.policy_net.state_dict(), 'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'learn_steps': self.learn_steps}

    def load_state_dict(self, state):
        self.policy_net.load_state_dict(state['policy_net'])
        self.target_net.load_state_dict(state['target_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.learn_steps = int(state.get('learn_steps', 0))
