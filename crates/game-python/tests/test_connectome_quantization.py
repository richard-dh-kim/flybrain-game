import unittest

import numpy as np

from flybrain_training.quantized_connectome import quantize_rowwise


class ConnectomeQuantizationTest(unittest.TestCase):
    def test_rowwise_quantization_uses_independent_scales(self) -> None:
        offsets = np.array([0, 0, 3, 5, 5])
        weights = np.array([0.1, 0.5, 1.0, 10.0, 20.0], dtype=np.float32)

        result = quantize_rowwise(weights, offsets, 8)

        self.assertEqual(result.values.dtype, np.uint8)
        self.assertEqual(result.scales[0], 0)
        self.assertEqual(result.scales[3], 0)
        self.assertEqual(result.values[2], 255)
        self.assertEqual(result.values[4], 255)
        np.testing.assert_allclose(
            result.dequantized,
            weights,
            rtol=0,
            atol=float(result.scales.max()) / 2 + 1e-7,
        )

    def test_invalid_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not cover"):
            quantize_rowwise(np.ones(2), np.array([0, 1]), 8)
        with self.assertRaisesRegex(ValueError, "8 or 16"):
            quantize_rowwise(np.ones(2), np.array([0, 2]), 4)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            quantize_rowwise(np.array([1, -1]), np.array([0, 2]), 8)


if __name__ == "__main__":
    unittest.main()
