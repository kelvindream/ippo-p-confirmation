"""Classic Monte-Carlo REINFORCE with a categorical policy."""

from collections import namedtuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ("state", "action", "reward", "done"))


class ReinforcePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, states):
        return self.network(states)


class ReinforceAgent:
    """Vanilla REINFORCE: no value baseline, replay, entropy bonus, or off-policy update."""

    formal_name = "REINFORCE"
    reproduction_scope = "on-policy categorical Monte-Carlo return"

    def __init__(self, input_dim, output_dim, mode="train", gamma=0.99, lr=1e-3):
        self.mode = mode
        self.gamma = gamma
        self.policy = ReinforcePolicy(input_dim, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.trajectory = []

    def select_action(self, state, steps_done=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.policy(state)
        if self.mode == "train":
            return Categorical(logits=logits).sample().item()
        return logits.argmax(dim=-1).item()

    def store_transition(self, state, action, reward, done):
        self.trajectory.append(Transition(
            torch.as_tensor(state, dtype=torch.float32, device=device),
            int(action), float(reward), bool(done),
        ))

    def optimize_model(self):
        if not self.trajectory:
            return 0.0
        returns = []
        running = 0.0
        for transition in reversed(self.trajectory):
            running = transition.reward + self.gamma * running * (not transition.done)
            returns.append(running)
        returns.reverse()
        states = torch.stack([item.state for item in self.trajectory])
        actions = torch.as_tensor([item.action for item in self.trajectory], device=device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        logits = self.policy(states)
        log_probs = Categorical(logits=logits).log_prob(actions)
        loss = -(log_probs * returns).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.trajectory.clear()
        return float(loss.item())

    def update_target_network(self, steps_done=None):
        return None

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trajectory": self.trajectory,
        }

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.trajectory = state.get("trajectory", [])

    def load(self, path):
        state = torch.load(path, map_location=device)
        self.policy.load_state_dict(state.get("policy", state))
