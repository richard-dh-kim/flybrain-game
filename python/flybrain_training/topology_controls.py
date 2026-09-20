"""Deterministic topology controls for MaleCNS comparison experiments."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


TOPOLOGY_VARIANTS = ("measured", "shuffled-presynaptic", "random-sparse")


def topology_digest(crow: np.ndarray, column: np.ndarray) -> str:
    """Hash the CSR structure without including trainable edge values."""

    digest = hashlib.sha256()
    for array in (crow, column):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_topology_control(
    graph: Mapping[str, object], variant: str, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return a graph with a measured or deterministic control topology.

    All variants preserve the number of neurons, the number of edges, every
    postsynaptic row degree, and the original synapse-count array. The shuffled
    control applies one bijection to presynaptic neuron identities, which also
    preserves the multiset of presynaptic out-degrees. The random-sparse control
    samples unique presynaptic partners independently for each row.
    """

    if variant not in TOPOLOGY_VARIANTS:
        raise ValueError(f"unknown topology variant: {variant}")
    if seed < 0:
        raise ValueError("topology seed must be nonnegative")

    crow = np.asarray(graph["crow"], dtype=np.int64)
    original_column = np.asarray(graph["col"], dtype=np.int64)
    counts = np.asarray(graph["counts"], dtype=np.float32)
    if crow.ndim != 1 or original_column.ndim != 1 or counts.ndim != 1:
        raise ValueError("connectome CSR arrays must be one-dimensional")
    if len(original_column) != len(counts) or int(crow[-1]) != len(original_column):
        raise ValueError("connectome CSR arrays have inconsistent edge counts")

    neuron_count = len(crow) - 1
    degrees = np.diff(crow)
    if np.any(degrees < 0) or np.any(degrees > neuron_count):
        raise ValueError("connectome rows must contain valid unique edge counts")
    if np.any(original_column < 0) or np.any(original_column >= neuron_count):
        raise ValueError("connectome column index is out of range")

    generator = np.random.default_rng(seed)
    if variant == "measured":
        column = original_column
        preserves_out_degree_multiset = True
        construction = "unaltered measured MaleCNS CSR topology"
    elif variant == "shuffled-presynaptic":
        presynaptic_permutation = generator.permutation(neuron_count)
        column = presynaptic_permutation[original_column]
        preserves_out_degree_multiset = True
        construction = (
            "one seeded bijection of all presynaptic neuron identities; "
            "postsynaptic rows and edge weights remain fixed"
        )
    else:
        column = np.empty_like(original_column)
        for start, end in zip(crow[:-1], crow[1:], strict=True):
            degree = int(end - start)
            if degree:
                column[start:end] = generator.choice(
                    neuron_count, size=degree, replace=False, shuffle=False
                )
        preserves_out_degree_multiset = False
        construction = (
            "seeded uniform presynaptic samples without replacement inside "
            "each fixed postsynaptic row"
        )

    controlled = {
        "crow": crow,
        "col": column,
        "counts": counts,
    }
    if "body_ids" in graph:
        controlled["body_ids"] = np.asarray(graph["body_ids"])
    metadata: dict[str, object] = {
        "variant": variant,
        "seed": seed,
        "digest": topology_digest(crow, column),
        "construction": construction,
        "preserves_neuron_count": True,
        "preserves_edge_count": True,
        "preserves_postsynaptic_row_degrees": True,
        "preserves_synapse_count_array": True,
        "preserves_presynaptic_out_degree_multiset": preserves_out_degree_multiset,
        "allows_self_edges": True,
        "duplicate_presynaptic_partners_within_row": False,
    }
    return controlled, metadata
