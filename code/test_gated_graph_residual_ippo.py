import os
import tempfile
import unittest

import numpy as np
import torch

from agents.gated_graph_residual_ippo import GatedGraphResidualIPPO
from agents.ippo import IPPO


class GatedGraphResidualIPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.num_agents, self.obs_dim, self.action_dim = 3, 4, 2
        self.adjacency = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        self.observations = np.random.randn(4, self.num_agents, self.obs_dim).astype(np.float32)
        self.checkpoint = tempfile.NamedTemporaryFile(delete=False).name
        IPPO(self.num_agents, self.obs_dim, self.action_dim, hidden_dim=16).save(self.checkpoint)

    def tearDown(self):
        if os.path.exists(self.checkpoint):
            os.unlink(self.checkpoint)

    def _agent(self, freeze_actor):
        agent = GatedGraphResidualIPPO(
            self.num_agents, self.obs_dim, self.action_dim, self.adjacency,
            hidden_dim=16, freeze_actor=freeze_actor, update_epochs=1,
        )
        agent.initialize_from_ippo(self.checkpoint)
        return agent

    def test_frozen_actor_and_checkpoint_roundtrip(self):
        agent = self._agent(True)
        original = [parameter.detach().clone() for parameter in agent.base.actors.parameters()]
        actions, log_probs, values = agent.select_actions(self.observations[0])
        self.assertEqual(len(actions), self.num_agents)
        self.assertEqual(log_probs.shape, (self.num_agents,))
        self.assertEqual(values.shape, (self.num_agents,))
        agent.update(self.observations, np.zeros((4, self.num_agents), dtype=np.int64),
                     np.tile(log_probs, (4, 1)), np.ones((4, self.num_agents), dtype=np.float32),
                     np.zeros((4, self.num_agents), dtype=np.float32), np.zeros(4, dtype=np.float32))
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(original, agent.base.actors.parameters())))
        agent.save(self.checkpoint)
        agent.load(self.checkpoint)

    def test_joint_actor_updates(self):
        agent = self._agent(False)
        original = [parameter.detach().clone() for parameter in agent.base.actors.parameters()]
        _, log_probs, _ = agent.select_actions(self.observations[0])
        agent.update(self.observations, np.zeros((4, self.num_agents), dtype=np.int64),
                     np.tile(log_probs, (4, 1)), np.ones((4, self.num_agents), dtype=np.float32),
                     np.zeros((4, self.num_agents), dtype=np.float32), np.zeros(4, dtype=np.float32))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(original, agent.base.actors.parameters())))


if __name__ == '__main__':
    unittest.main()
