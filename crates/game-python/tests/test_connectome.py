import unittest

import numpy as np
import torch

from flybrain_training.connectome import SparseConnectome, _EdgeSparseMM
from flybrain_training.connectome_policy import MaleCnsPolicy


def tiny_connectome() -> SparseConnectome:
    # Measured orientation is row=post, column=pre: 0 -> 1 -> 2.
    return SparseConnectome(
        np.array([0, 0, 1, 2]),
        np.array([0, 1]),
        np.array([2, 3]),
    )


class SparseConnectomeTest(unittest.TestCase):
    def test_edge_orientation_requires_measured_path(self) -> None:
        network = tiny_connectome()
        state = torch.tensor([[1.0], [0.0], [0.0]])

        one_step = network(state)
        two_steps = network(state, steps=2)

        self.assertGreater(one_step[1, 0], 0)
        self.assertEqual(one_step[2, 0], 0)
        self.assertGreater(two_steps[2, 0], 0)

    def test_recurrent_sparse_forward_and_gradients_match_dense(self) -> None:
        network = tiny_connectome()
        state = torch.tensor(
            [[0.7], [-0.2], [0.3]], dtype=torch.float32, requires_grad=True
        )
        sparse = network(state, steps=4)
        sparse.square().sum().backward()
        sparse_gradients = (
            network.edge_gain.grad.clone(),
            network.leak.grad.clone(),
            state.grad.clone(),
        )

        network.zero_grad(set_to_none=True)
        state.grad = None
        dense = state
        matrix = network.matrix().to_dense()
        leak = (0.05 + 0.90 * torch.sigmoid(network.leak))[:, None]
        for _ in range(4):
            dense = (1 - leak) * dense + leak * torch.tanh(matrix @ dense)
        dense.square().sum().backward()

        torch.testing.assert_close(sparse, dense)
        for sparse_gradient, dense_gradient in zip(
            sparse_gradients,
            (network.edge_gain.grad, network.leak.grad, state.grad),
            strict=True,
        ):
            torch.testing.assert_close(sparse_gradient, dense_gradient)

    def test_training_cannot_change_topology(self) -> None:
        network = tiny_connectome()
        before = (network.crow.clone(), network.column.clone())
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        loss = network(torch.ones(3, 1), steps=3).square().mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(network.edge_gain.grad).all())
        self.assertTrue(torch.isfinite(network.leak.grad).all())
        self.assertTrue(torch.equal(network.crow, before[0]))
        self.assertTrue(torch.equal(network.column, before[1]))
        self.assertEqual(network.matrix()._nnz(), 2)

    def test_sparse_operator_passes_double_precision_gradcheck(self) -> None:
        network = tiny_connectome().double()
        values = network.base.clone().requires_grad_(True)
        state = torch.tensor(
            [[0.3, -0.2], [-0.4, 0.6], [0.9, 0.1]],
            dtype=torch.double,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda edge_values, recurrent_state: _EdgeSparseMM.apply(
                    edge_values,
                    network.crow,
                    network.column,
                    network.rows,
                    recurrent_state,
                ),
                (values, state),
            )
        )

    def test_policy_carries_state_and_all_trainable_layers_receive_gradients(self) -> None:
        graph = {
            "crow": np.array([0, 0, 1, 2]),
            "col": np.array([0, 1]),
            "counts": np.array([2, 3]),
        }
        policy = MaleCnsPolicy(
            graph,
            sensory_ids=np.array([0]),
            motor_ids=np.array([1, 2]),
            input_size=33,
        )
        features = torch.linspace(-1, 1, 3 * 4 * 33).reshape(3, 4, 33)

        targets, strike_logits, final_state = policy(features)
        loss = targets.square().mean() + strike_logits.square().mean()
        loss.backward()

        self.assertEqual(targets.shape, (3, 4, 2))
        self.assertEqual(strike_logits.shape, (3, 4))
        self.assertEqual(final_state.shape, (3, 3))
        self.assertGreater(torch.count_nonzero(final_state).item(), 0)
        for parameter in policy.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        reset_targets, _, _ = policy(features[:, -1:])
        carried_targets, _, _ = policy(features[:, -1:], final_state.detach())
        self.assertFalse(torch.allclose(reset_targets, carried_targets))


if __name__ == "__main__":
    unittest.main()
