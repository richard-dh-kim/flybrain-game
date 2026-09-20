"""Sparse recurrent rate model with immutable measured adjacency.

Adapted from flyhard commit 328906f4a0e62c8f9fc18805cf6edae6989b82a5.
Copyright (c) 2026 Mark Unthank. Distributed under the MIT License; see
third_party/flyhard/LICENSE and third_party/flyhard/NOTICE.md.

Each measured edge has a trainable bounded gain, and every neuron has a
trainable leak rate. Transmission is currently an unsigned numerical proxy;
transmitter and receptor biology are not asserted by this model.
"""

from __future__ import annotations

import torch
from torch import nn


class _EdgeSparseMM(torch.autograd.Function):
    """First-order SpMM derivative evaluated only at measured edges.

    PyTorch 2.8's CSR-value backward allocates a dense N-by-N intermediate on
    the full graph. The exact derivative at an edge i<-j is the dot product of
    dL/dY[i] and X[j]. Chunked indexing bounds temporary memory.
    """

    @staticmethod
    def forward(
        context: object,
        values: torch.Tensor,
        crow: torch.Tensor,
        column: torch.Tensor,
        rows: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        neuron_count = len(crow) - 1
        matrix = torch.sparse_csr_tensor(
            crow,
            column,
            values,
            size=(neuron_count, neuron_count),
            check_invariants=False,
        )
        context.save_for_backward(values, crow, column, rows, state)
        return torch.sparse.mm(matrix, state)

    @staticmethod
    def backward(
        context: object, output_gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None, torch.Tensor | None]:
        values, crow, column, rows, state = context.saved_tensors
        value_gradient = (
            torch.empty_like(values) if context.needs_input_grad[0] else None
        )
        if value_gradient is not None:
            for start in range(0, len(values), 262_144):
                end = min(start + 262_144, len(values))
                value_gradient[start:end] = (
                    output_gradient[rows[start:end]] * state[column[start:end]]
                ).sum(dim=1)
        state_gradient = None
        if context.needs_input_grad[4]:
            neuron_count = len(crow) - 1
            matrix = torch.sparse_csr_tensor(
                crow,
                column,
                values,
                size=(neuron_count, neuron_count),
                check_invariants=False,
            )
            state_gradient = torch.sparse.mm(matrix.transpose(0, 1), output_gradient)
        return value_gradient, None, None, None, state_gradient


class SparseConnectome(nn.Module):
    def __init__(
        self,
        crow: object,
        column: object,
        counts: object,
        *,
        edge_init: float = 0.0,
        leak_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.n = len(crow) - 1
        self.register_buffer("crow", torch.as_tensor(crow, dtype=torch.int64))
        self.register_buffer("column", torch.as_tensor(column, dtype=torch.int64))
        counts_tensor = torch.as_tensor(counts, dtype=torch.float32)
        rows = torch.repeat_interleave(torch.arange(self.n), torch.diff(self.crow))
        self.register_buffer("rows", rows)
        totals = torch.zeros(self.n).index_add_(0, rows, counts_tensor)
        self.register_buffer("base", counts_tensor / totals[rows].clamp_min(1))
        self.edge_gain = nn.Parameter(
            torch.full_like(counts_tensor, float(edge_init))
        )
        self.leak = nn.Parameter(torch.full((self.n,), float(leak_init)))

    def edge_values(self) -> torch.Tensor:
        return self.base * (0.05 + 0.90 * torch.sigmoid(self.edge_gain))

    def matrix(self, values: torch.Tensor | None = None) -> torch.Tensor:
        if values is None:
            values = self.edge_values()
        return torch.sparse_csr_tensor(
            self.crow,
            self.column,
            values,
            size=(self.n, self.n),
            check_invariants=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        steps: int = 1,
        drive: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance `[neurons, batch]` state through measured connections."""

        values = self.edge_values()
        leak = (0.05 + 0.90 * torch.sigmoid(self.leak))[:, None]
        for _ in range(steps):
            signal = _EdgeSparseMM.apply(
                values, self.crow, self.column, self.rows, state
            )
            if drive is not None:
                signal = signal + drive
            state = (1 - leak) * state + leak * torch.tanh(signal)
        return state
