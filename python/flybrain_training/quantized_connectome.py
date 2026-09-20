"""Experimental row-wise quantization for packed MaleCNS inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RowwiseQuantization:
    bits: int
    values: np.ndarray
    scales: np.ndarray
    dequantized: np.ndarray


def quantize_rowwise(
    edge_weights: object,
    row_offsets: object,
    bits: int,
    *,
    row_chunk: int = 4096,
) -> RowwiseQuantization:
    """Quantize positive CSR weights using one maximum-derived scale per row."""

    if bits not in (8, 16):
        raise ValueError("row-wise quantization supports 8 or 16 bits")
    weights = np.asarray(edge_weights, dtype=np.float32)
    offsets = np.asarray(row_offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("row offsets must be a nonempty vector")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(weights):
        raise ValueError("row offsets do not cover the edge weights")
    lengths = np.diff(offsets)
    if np.any(lengths < 0):
        raise ValueError("row offsets must be monotonic")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("edge weights must be finite and nonnegative")

    levels = (1 << bits) - 1
    storage_dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    maxima = np.zeros(len(lengths), dtype=np.float32)
    nonempty = lengths > 0
    maxima[nonempty] = np.maximum.reduceat(
        weights, np.asarray(offsets[:-1][nonempty], dtype=np.intp)
    )
    scales = maxima / np.float32(levels)
    quantized = np.empty(len(weights), dtype=storage_dtype)
    dequantized = np.empty(len(weights), dtype=np.float32)

    for row_start in range(0, len(lengths), row_chunk):
        row_end = min(row_start + row_chunk, len(lengths))
        edge_start = int(offsets[row_start])
        edge_end = int(offsets[row_end])
        if edge_start == edge_end:
            continue
        edge_scales = np.repeat(scales[row_start:row_end], lengths[row_start:row_end])
        ratios = np.divide(
            weights[edge_start:edge_end],
            edge_scales,
            out=np.zeros(edge_end - edge_start, dtype=np.float32),
            where=edge_scales != 0,
        )
        chunk = np.rint(ratios).clip(0, levels).astype(storage_dtype)
        quantized[edge_start:edge_end] = chunk
        dequantized[edge_start:edge_end] = chunk.astype(np.float32) * edge_scales

    return RowwiseQuantization(bits, quantized, scales, dequantized)


class FrozenConnectomePolicy:
    """PyTorch evaluator for already-fused packed inference arrays."""

    def __init__(
        self,
        *,
        row_offsets: object,
        column_indices: object,
        edge_weights: object,
        leak: object,
        sensory_indices: object,
        sensory_feature_ids: object,
        sensory_signs: object,
        motor_indices: object,
        readout_weight: object,
        readout_bias: object,
        feature_count: int,
        device: torch.device,
    ) -> None:
        row_tensor = torch.as_tensor(
            np.asarray(row_offsets).copy(), dtype=torch.int64, device=device
        )
        column_tensor = torch.as_tensor(
            np.asarray(column_indices).copy(), dtype=torch.int64, device=device
        )
        value_tensor = torch.as_tensor(
            np.asarray(edge_weights).copy(), dtype=torch.float32, device=device
        )
        neuron_count = len(row_tensor) - 1
        self.matrix = torch.sparse_csr_tensor(
            row_tensor,
            column_tensor,
            value_tensor,
            size=(neuron_count, neuron_count),
            check_invariants=False,
        )
        self.leak = torch.as_tensor(
            np.asarray(leak).copy(), dtype=torch.float32, device=device
        )[:, None]
        self.sensory_indices = torch.as_tensor(
            np.asarray(sensory_indices).copy(), dtype=torch.int64, device=device
        )
        self.sensory_feature_ids = torch.as_tensor(
            np.asarray(sensory_feature_ids).copy(), dtype=torch.int64, device=device
        )
        self.sensory_signs = torch.as_tensor(
            np.asarray(sensory_signs).copy(), dtype=torch.float32, device=device
        )
        self.motor_indices = torch.as_tensor(
            np.asarray(motor_indices).copy(), dtype=torch.int64, device=device
        )
        self.readout_weight = torch.as_tensor(
            np.asarray(readout_weight).copy(), dtype=torch.float32, device=device
        )
        self.readout_bias = torch.as_tensor(
            np.asarray(readout_bias).copy(), dtype=torch.float32, device=device
        )
        self.feature_count = feature_count
        self.neuron_count = neuron_count
        self.device = device

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.neuron_count,
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

    def step(
        self, features: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("feature shape differs from the packed interface")
        drive = torch.zeros_like(state)
        drive[self.sensory_indices] = (
            features[:, self.sensory_feature_ids].T * self.sensory_signs[:, None]
        )
        signal = torch.sparse.mm(self.matrix, state) + drive
        state = (1 - self.leak) * state + self.leak * torch.tanh(signal)
        motor_state = state[self.motor_indices].T
        raw = motor_state @ self.readout_weight.T + self.readout_bias
        return torch.sigmoid(raw[:, :2]), raw[:, 2], state

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[2] != self.feature_count:
            raise ValueError("feature sequence shape differs from the packed interface")
        state = self.initial_state(features.shape[0])
        targets = []
        strikes = []
        for tick in range(features.shape[1]):
            target, strike, state = self.step(features[:, tick], state)
            targets.append(target)
            strikes.append(strike)
        return torch.stack(targets, dim=1), torch.stack(strikes, dim=1), state
