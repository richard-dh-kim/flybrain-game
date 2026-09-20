"""Small conventional policies used before the connectome controller."""

from __future__ import annotations

import torch
from torch import nn


class MlpPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(features)
        return torch.sigmoid(output[..., :2]), output[..., 2]


class GruPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 3)

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recurrent, hidden = self.recurrent(features, hidden)
        output = self.output(recurrent)
        return torch.sigmoid(output[..., :2]), output[..., 2], hidden
