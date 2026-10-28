import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import paper_protocol
import test_algorithms as training
from agents.gc_mappo import GCMAPPO
from agents.ippo import IPPO


class FormalFourGridTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ippo_prior_select_update_uses_same_prior_and_temperature(self):
        agent = IPPO(
            2, 3, 2, hidden_dim=8, update_epochs=1, learning_rate=0.0,
            action_prior_scale=2.25, prior_mode='pressure_zscore_hold',
            policy_temperature=1.0,
        )
        observations = np.asarray([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.2, 0.1, 0.0], [0.6, 0.4, 0.2]],
        ], dtype=np.float32)
        priors = np.asarray([
            [[1.0, -1.0], [-0.5, 0.5]],
            [[0.25, -0.25], [0.75, -0.75]],
        ], dtype=np.float32)
        temperatures = np.ones(2, dtype=np.float32)
        actions, old_logs, values = [], [], []
        for step in range(2):
            step_actions, step_logs, step_values = agent.select_actions(
                observations[step], deterministic=True,
                action_priors=priors[step], policy_temperature=temperatures[step],
            )
            actions.append(step_actions)
            old_logs.append(step_logs)
            values.append(step_values)
        captured = []
        original_exp = torch.exp

        def capture_exp(value):
            captured.append(value.detach().cpu())
            return original_exp(value)

        with mock.patch('agents.ippo.torch.exp', side_effect=capture_exp):
            agent.update(
                observations, np.asarray(actions), np.asarray(old_logs),
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.asarray(values), np.zeros(2, dtype=np.float32),
                action_priors=priors, policy_temperatures=temperatures,
            )
        self.assertEqual(len(captured), 2)
        for log_delta in captured:
            torch.testing.assert_close(log_delta, torch.zeros_like(log_delta), atol=1e-6, rtol=0)

    def test_gc_formal_wrappers_only_change_prior_scale(self):
        with mock.patch.object(training, 'run_gc_mappo', return_value={}) as runner:
            training.run_gc_mappo_ind_no_prior(num_episodes=1)
            no_prior = runner.call_args.kwargs
            training.run_gc_mappo_ind_prior(num_episodes=1)
            with_prior = runner.call_args.kwargs
        self.assertTrue(no_prior['independent_actor'])
        self.assertTrue(with_prior['independent_actor'])
        self.assertEqual(no_prior['action_prior_scale'], 0.0)
        self.assertEqual(with_prior['action_prior_scale'], 2.25)
        shared_keys = set(no_prior) | set(with_prior)
        for key in shared_keys - {'action_prior_scale', 'method_name'}:
            self.assertEqual(no_prior.get(key), with_prior.get(key))

    def test_checkpoint_config_roundtrip_and_variant_validation(self):
        ippo = IPPO(2, 3, 2, hidden_dim=8, action_prior_scale=2.25,
                    prior_mode='pressure_zscore_hold', policy_temperature=1.0)
        state = ippo.state_dict()
        self.assertEqual(state['config']['action_prior_scale'], 2.25)
        restored = IPPO(2, 3, 2, hidden_dim=8, action_prior_scale=2.25,
                        prior_mode='pressure_zscore_hold', policy_temperature=1.0)
        restored.load_state_dict(state)
        incompatible = IPPO(2, 3, 2, hidden_dim=8, action_prior_scale=0.0,
                            prior_mode='none', policy_temperature=1.0)
        with self.assertRaises(RuntimeError):
            incompatible.load_state_dict(state)

        gc = GCMAPPO(
            2, 5, 2, adjacency_matrix=np.eye(2), distance_matrix=np.zeros((2, 2)),
            hidden_dim=8, independent_actor=True, action_prior_scale=0.0,
            action_value_head=False, soft_q_execution=True,
        )
        gc_state = gc.state_dict()
        self.assertEqual(gc_state['config']['actor_type'], 'independent')
        self.assertTrue(gc_state['config']['pressure_observation'])
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint:
            gc.save(checkpoint.name)
            loaded = torch.load(checkpoint.name, map_location='cpu', weights_only=False)
            self.assertEqual(loaded['config'], gc_state['config'])

    def test_protocol_registers_formal_four_grid(self):
        expected = {'IPPO', 'IPPO-P', 'GC-MAPPO-ind-noP', 'GC-MAPPO-ind-P'}
        self.assertTrue(expected.issubset(set(paper_protocol.METHODS)))
        self.assertEqual(set(paper_protocol.FORMAL_VARIANTS), expected)
        for method in expected:
            self.assertIsNotNone(paper_protocol._runner(method))
        self.assertEqual(paper_protocol.FORMAL_VARIANTS['IPPO']['policy_temperature'], 1.0)
        self.assertEqual(paper_protocol.FORMAL_VARIANTS['IPPO-P']['action_prior_scale'], 2.25)
        self.assertEqual(paper_protocol.FORMAL_VARIANTS['GC-MAPPO-ind-noP']['action_prior_scale'], 0.0)
        self.assertEqual(paper_protocol.FORMAL_VARIANTS['GC-MAPPO-ind-P']['action_prior_scale'], 2.25)


if __name__ == '__main__':
    unittest.main()
