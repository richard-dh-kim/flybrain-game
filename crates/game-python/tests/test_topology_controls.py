import unittest

import numpy as np

from flybrain_training.topology_controls import (
    build_topology_control,
    topology_digest,
)


def toy_graph() -> dict[str, np.ndarray]:
    return {
        "crow": np.array([0, 3, 5, 7, 10, 12, 15], dtype=np.int64),
        "col": np.array(
            [0, 2, 5, 1, 4, 0, 3, 1, 2, 5, 0, 4, 1, 3, 5], dtype=np.int64
        ),
        "counts": np.arange(1, 16, dtype=np.float32),
        "body_ids": np.arange(100, 106, dtype=np.int64),
    }


class TopologyControlsTest(unittest.TestCase):
    def assert_common_invariants(
        self, original: dict[str, np.ndarray], controlled: dict[str, np.ndarray]
    ) -> None:
        np.testing.assert_array_equal(controlled["crow"], original["crow"])
        np.testing.assert_array_equal(controlled["counts"], original["counts"])
        np.testing.assert_array_equal(controlled["body_ids"], original["body_ids"])
        self.assertEqual(len(controlled["col"]), len(original["col"]))
        self.assertTrue(np.all(controlled["col"] >= 0))
        self.assertTrue(np.all(controlled["col"] < len(original["crow"]) - 1))
        for start, end in zip(
            controlled["crow"][:-1], controlled["crow"][1:], strict=True
        ):
            row = controlled["col"][start:end]
            self.assertEqual(len(row), len(np.unique(row)))

    def test_measured_topology_is_unchanged(self) -> None:
        graph = toy_graph()
        controlled, metadata = build_topology_control(graph, "measured", 1701)

        self.assert_common_invariants(graph, controlled)
        np.testing.assert_array_equal(controlled["col"], graph["col"])
        self.assertEqual(
            metadata["digest"], topology_digest(graph["crow"], graph["col"])
        )

    def test_presynaptic_shuffle_is_deterministic_and_degree_matched(self) -> None:
        graph = toy_graph()
        first, metadata = build_topology_control(
            graph, "shuffled-presynaptic", 1701
        )
        second, _ = build_topology_control(graph, "shuffled-presynaptic", 1701)

        self.assert_common_invariants(graph, first)
        np.testing.assert_array_equal(first["col"], second["col"])
        self.assertFalse(np.array_equal(first["col"], graph["col"]))
        np.testing.assert_array_equal(
            np.sort(np.bincount(first["col"], minlength=6)),
            np.sort(np.bincount(graph["col"], minlength=6)),
        )
        self.assertTrue(metadata["preserves_presynaptic_out_degree_multiset"])

    def test_random_sparse_is_deterministic_and_has_unique_rows(self) -> None:
        graph = toy_graph()
        first, metadata = build_topology_control(graph, "random-sparse", 1701)
        second, _ = build_topology_control(graph, "random-sparse", 1701)
        other_seed, _ = build_topology_control(graph, "random-sparse", 1702)

        self.assert_common_invariants(graph, first)
        np.testing.assert_array_equal(first["col"], second["col"])
        self.assertFalse(np.array_equal(first["col"], other_seed["col"]))
        self.assertFalse(metadata["preserves_presynaptic_out_degree_multiset"])

    def test_rejects_invalid_variant_and_seed(self) -> None:
        with self.assertRaises(ValueError):
            build_topology_control(toy_graph(), "unknown", 1701)
        with self.assertRaises(ValueError):
            build_topology_control(toy_graph(), "measured", -1)


if __name__ == "__main__":
    unittest.main()
