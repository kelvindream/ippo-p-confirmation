import tempfile
import unittest

import numpy as np
import torch

from agents.dpg_mappo import DPGMAPPO, directional_phase_features


class DirectionalSignal:
    phase_link_map = [[('in_a', 'out_a'), ('in_b', 'out_a')]]
    lanes_length = {'out_a': 75}

    class Sumol:
        class lane:
            @staticmethod
            def getLastStepHaltingNumber(lane):
                return {'in_a': 10, 'in_b': 30, 'out_a': 5}[lane]

            @staticmethod
            def getLastStepVehicleNumber(lane):
                return 5

            @staticmethod
            def getLength(lane):
                return 75

    sumo = Sumol()


class DPGMAPPOTests(unittest.TestCase):
    def test_directional_mapping_and_zero_initialized_bounded_residual(self):
        features = directional_phase_features([DirectionalSignal()])
        np.testing.assert_allclose(features, [[[1.0, 0.5, 0.5]]])
        agent = DPGMAPPO(1, 4, 1, hidden_dim=16, update_epochs=1)
        observations = agent._tensor_observations([[0, 0, 0, 1]])
        directions = torch.tensor(features)
        base = agent.actor.actor(observations, agent.adjacency_matrix, agent.distance_matrix)
        gated = agent.actor(observations, agent.adjacency_matrix, agent.distance_matrix, directions)
        torch.testing.assert_close(base, gated)
        with torch.no_grad():
            agent.actor.residual[-1].bias.fill_(100.0)
        changed = agent.actor(observations, agent.adjacency_matrix, agent.distance_matrix, directions)
        self.assertLessEqual(float((changed - base).abs().max()), 0.5)

    def test_cached_features_update_and_checkpoint_roundtrip(self):
        agent = DPGMAPPO(1, 4, 2, hidden_dim=16, update_epochs=1)
        observations = np.zeros((2, 1, 4), dtype=np.float32)
        directions = np.ones((2, 1, 2, 3), dtype=np.float32)
        actions, logs, _ = agent.select_actions(observations[0], directional_features=directions[0])
        metrics = agent.update(
            observations=observations,
            actions=np.asarray([actions, actions]),
            old_log_probs=np.asarray([logs, logs]),
            rewards=np.ones((2, 1), dtype=np.float32),
            values=np.zeros((2, 1), dtype=np.float32),
            dones=np.zeros((2, 1), dtype=np.float32),
            directional_features=directions,
        )
        self.assertIn('actor_loss', metrics)
        with tempfile.NamedTemporaryFile(suffix='.pt') as file:
            agent.save(file.name)
            restored = DPGMAPPO(1, 4, 2, hidden_dim=16, update_epochs=1)
            restored.load(file.name)
            for left, right in zip(agent.actor.parameters(), restored.actor.parameters()):
                torch.testing.assert_close(left, right)


if __name__ == '__main__':
    unittest.main()
