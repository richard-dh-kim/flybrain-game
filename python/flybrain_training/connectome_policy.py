"""Engineered game interface around the measured MaleCNS sparse topology."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .connectome import SparseConnectome


class MaleCnsPolicy(nn.Module):
    """Map game features through MaleCNS state to target and strike outputs.

    The feature-to-sensory assignment is frozen and balanced. Recurrent edge
    gains, neuron leaks, and the motor-neuron readout are trainable. Every
    output therefore depends on state propagated through declared measured
    edges; there is no observation-to-action bypass.
    """

    def __init__(
        self,
        graph: object,
        sensory_ids: object,
        motor_ids: object,
        input_size: int,
        seed: int = 1701,
    ) -> None:
        super().__init__()
        self.core = SparseConnectome(graph["crow"], graph["col"], graph["counts"])
        sensory = np.asarray(sensory_ids, dtype=np.int64)
        motor = np.asarray(motor_ids, dtype=np.int64)
        if len(sensory) == 0 or len(motor) == 0:
            raise ValueError("MaleCNS policy needs nonempty sensory and motor sets")
        generator = np.random.default_rng(seed)
        feature_ids = np.arange(len(sensory), dtype=np.int64) % input_size
        generator.shuffle(feature_ids)
        input_signs = generator.choice((-1.0, 1.0), size=len(sensory)).astype(
            np.float32
        )
        self.register_buffer("sensory_ids", torch.as_tensor(sensory))
        self.register_buffer("motor_ids", torch.as_tensor(motor))
        self.register_buffer("feature_ids", torch.as_tensor(feature_ids))
        self.register_buffer("input_signs", torch.as_tensor(input_signs))
        self.readout = nn.Linear(len(motor), 3)
        nn.init.normal_(self.readout.weight, std=1.0 / np.sqrt(len(motor)))
        nn.init.zeros_(self.readout.bias)
        with torch.no_grad():
            self.readout.bias[2] = -3.0

    def initial_state(
        self, batch_size: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        parameter = self.core.edge_gain
        return torch.zeros(
            self.core.n,
            batch_size,
            dtype=parameter.dtype,
            device=device if device is not None else parameter.device,
        )

    def step(
        self, features: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, features]")
        drive = torch.zeros_like(state)
        drive[self.sensory_ids] = (
            features[:, self.feature_ids].T * self.input_signs[:, None]
        )
        state = self.core(state, steps=1, drive=drive)
        raw = self.readout(state[self.motor_ids].T)
        return torch.sigmoid(raw[:, :2]), raw[:, 2], state

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, features]")
        if state is None:
            state = self.initial_state(features.shape[0], device=features.device)
        targets = []
        strikes = []
        for tick in range(features.shape[1]):
            target, strike, state = self.step(features[:, tick], state)
            targets.append(target)
            strikes.append(strike)
        return torch.stack(targets, dim=1), torch.stack(strikes, dim=1), state
