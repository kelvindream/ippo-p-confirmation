import random
import torch
from collections import namedtuple, deque
import numpy as np

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 经验元组定义
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """普通经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.steps_done = 0

    def add(self, state, next_state, reward, action, done=False) -> None:
        """添加经验到缓冲区"""
        if reward is not None and action is not None:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
            done = torch.tensor(done, dtype=torch.bool).unsqueeze(0).to(device)
            self.memory.append(Transition(state, action, next_state, reward, done))
            self.steps_done += 1

    def sample(self, batch_size):
        """从缓冲区中采样经验"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回缓冲区大小"""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha
        self.steps_done = 0

    def add(self, state, next_state, reward, action, done=False) -> None:
        """添加经验到缓冲区"""
        if reward is not None and action is not None:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
            done = torch.tensor(done, dtype=torch.bool).unsqueeze(0).to(device)
            
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.memory.append(Transition(state, action, next_state, reward, done))
            self.priorities.append(max_priority)
            
            if len(self.memory) > self.capacity:
                del self.memory[0]
                del self.priorities[0]
            
            self.steps_done += 1
    
    def push(self, state, action, next_state, reward, done):
        """push方法别名，兼容enhanced_dueling_dqn的接口"""
        self.add(state, next_state, reward, action, done)

    def sample(self, batch_size, beta=0.4):
        """从缓冲区中采样经验"""
        if len(self.memory) == 0:
            return [], [], []
        
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[i] for i in indices]
        
        # 计算重要性采样权重
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device).unsqueeze(1)
        
        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        """更新经验的优先级"""
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error.item()) + 1e-6

    def __len__(self):
        """返回缓冲区大小"""
        return len(self.memory)
