import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flybrain_training.packed_connectome import (
    FORMAT_NAME,
    FORMAT_VERSION,
    PackedConnectome,
    seal_manifest,
    write_array,
)


class PackedConnectomeTest(unittest.TestCase):
    def create_package(self, directory: Path) -> None:
        arrays = {
            "row_offsets": write_array(
                directory,
                "row_offsets.u32.bin",
                np.array([0, 0, 1, 2, 2]),
                "u32",
            ),
            "column_indices": write_array(
                directory, "column_indices.u32.bin", np.array([0, 1]), "u32"
            ),
            "edge_weights": write_array(
                directory,
                "edge_weights.f32.bin",
                np.array([0.4, 0.7]),
                "f32",
            ),
            "leak": write_array(
                directory,
                "leak.f32.bin",
                np.array([0.5, 0.4, 0.3, 0.2]),
                "f32",
            ),
            "sensory_indices": write_array(
                directory, "sensory_indices.u32.bin", np.array([0]), "u32"
            ),
            "sensory_feature_ids": write_array(
                directory, "sensory_feature_ids.u8.bin", np.array([1]), "u8"
            ),
            "sensory_signs": write_array(
                directory, "sensory_signs.i8.bin", np.array([-1]), "i8"
            ),
            "motor_indices": write_array(
                directory, "motor_indices.u32.bin", np.array([2, 3]), "u32"
            ),
            "readout_weight": write_array(
                directory,
                "readout_weight.f32.bin",
                np.array([[1, 0], [0, 1], [0.5, -0.25]]),
                "f32",
            ),
            "readout_bias": write_array(
                directory,
                "readout_bias.f32.bin",
                np.array([0.1, -0.2, 0.3]),
                "f32",
            ),
        }
        manifest = seal_manifest(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "dimensions": {
                    "neurons": 4,
                    "edges": 2,
                    "features": 3,
                    "sensory_neurons": 1,
                    "motor_neurons": 2,
                    "outputs": 3,
                },
                "outputs": {"strike_threshold": 0.5},
                "arrays": arrays,
            }
        )
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def test_step_matches_direct_recurrence_and_handles_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.create_package(directory)
            policy = PackedConnectome(directory)
            features = np.array([0.2, 0.6, -0.4], dtype=np.float32)
            state = np.array([0.8, -0.3, 0.2, -0.1], dtype=np.float32)

            target, strike, actual_state = policy.step(features, state)

            signal = np.array([-0.6, 0.4 * 0.8, 0.7 * -0.3, 0], dtype=np.float32)
            leak = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
            expected_state = (1 - leak) * state + leak * np.tanh(signal)
            raw = (
                np.array([[1, 0], [0, 1], [0.5, -0.25]], dtype=np.float32)
                @ expected_state[[2, 3]]
                + np.array([0.1, -0.2, 0.3], dtype=np.float32)
            )
            expected_target = 1 / (1 + np.exp(-raw[:2]))

            np.testing.assert_allclose(actual_state, expected_state, rtol=0, atol=1e-7)
            np.testing.assert_allclose(target, expected_target, rtol=0, atol=1e-7)
            self.assertAlmostEqual(float(strike), float(raw[2]), places=7)

    def test_corrupt_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.create_package(directory)
            path = directory / "leak.f32.bin"
            contents = bytearray(path.read_bytes())
            contents[0] ^= 1
            path.write_bytes(contents)

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                PackedConnectome(directory)

    def test_corrupt_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.create_package(directory)
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["dimensions"]["neurons"] = 5
            path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                PackedConnectome(directory)


if __name__ == "__main__":
    unittest.main()
