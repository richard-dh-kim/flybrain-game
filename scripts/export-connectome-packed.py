#!/usr/bin/env python3
"""Export a selected MaleCNS checkpoint as packed inference-only arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow.feather as feather
import torch

from flybrain_training.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from flybrain_training.packed_connectome import (
    FORMAT_NAME,
    FORMAT_VERSION,
    seal_manifest,
    sha256_file,
    write_array,
)


EXPECTED_GRAPH_SHA256 = "eff4093bf53c4dd17d7ee4f2f838f6ae5ede70570f9a91317dd8824cca5c771d"


def resolve_graph_path(requested: Path | None) -> Path:
    candidates = [
        requested,
        Path(os.environ["FLYBRAIN_GRAPH"]) if "FLYBRAIN_GRAPH" in os.environ else None,
        Path("data/graph-traced-v1"),
        Path("/tmp/flybrain-upstream-review/flyhard/data/graph-traced-v1"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "graph.npz").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MaleCNS graph not found; pass --graph or set FLYBRAIN_GRAPH"
    )


def fused_edges(
    crow: np.ndarray,
    counts: np.ndarray,
    gains: torch.Tensor,
    output_path: Path,
    *,
    row_chunk: int = 4096,
) -> None:
    """Fuse normalized synapse counts and bounded gains without a rows array."""

    row_lengths = np.diff(crow)
    nonempty = row_lengths > 0
    row_totals = np.zeros(len(row_lengths), dtype=np.float32)
    row_totals[nonempty] = np.add.reduceat(
        counts, np.asarray(crow[:-1][nonempty], dtype=np.intp)
    )
    output = np.memmap(output_path, dtype="<f4", mode="w+", shape=(len(counts),))
    for row_start in range(0, len(row_lengths), row_chunk):
        row_end = min(row_start + row_chunk, len(row_lengths))
        edge_start = int(crow[row_start])
        edge_end = int(crow[row_end])
        if edge_start == edge_end:
            continue
        denominators = np.repeat(
            row_totals[row_start:row_end].clip(min=1),
            row_lengths[row_start:row_end],
        )
        count_tensor = torch.from_numpy(np.asarray(counts[edge_start:edge_end]))
        denominator_tensor = torch.from_numpy(denominators)
        gain_tensor = gains[edge_start:edge_end]
        values = (count_tensor / denominator_tensor) * (
            0.05 + 0.90 * torch.sigmoid(gain_tensor)
        )
        output[edge_start:edge_end] = values.numpy()
    output.flush()
    del output


def array_entry(path: Path, dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "file": path.name,
        "dtype": dtype,
        "shape": list(shape),
        "byte_length": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/connectome-v1-selected.pt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/connectome-packed-v1")
    )
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to replace existing package: {args.out}")
    graph_directory = resolve_graph_path(args.graph)
    graph_path = graph_directory / "graph.npz"
    graph_sha256 = sha256_file(graph_path)
    if graph_sha256 != EXPECTED_GRAPH_SHA256:
        raise ValueError(f"unexpected MaleCNS graph: {graph_sha256}")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["graph_sha256"] != graph_sha256:
        raise ValueError("checkpoint and graph hashes differ")
    if int(checkpoint["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint uses another feature schema")
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("checkpoint feature names differ from the active schema")
    trainable = checkpoint["trainable_state_dict"]
    expected_parameters = {
        "core.edge_gain",
        "core.leak",
        "readout.weight",
        "readout.bias",
    }
    if set(trainable) != expected_parameters:
        raise ValueError("unexpected trainable checkpoint layout")

    nodes = feather.read_table(graph_directory / "nodes.feather")
    superclasses = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_indices = np.flatnonzero(superclasses == "vnc_sensory")
    motor_indices = np.flatnonzero(superclasses == "vnc_motor")
    seed = int(checkpoint["seed"])
    generator = np.random.default_rng(seed)
    sensory_feature_ids = np.arange(len(sensory_indices), dtype=np.int64) % len(
        FEATURE_NAMES
    )
    generator.shuffle(sensory_feature_ids)
    sensory_feature_ids = sensory_feature_ids.astype(np.uint8)
    sensory_signs = generator.choice((-1, 1), size=len(sensory_indices)).astype(
        np.int8
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.out.name}-", dir=args.out.parent)
    )
    try:
        with np.load(graph_path) as graph:
            crow = np.asarray(graph["crow"])
            column = np.asarray(graph["col"])
            counts = np.asarray(graph["counts"])
            neuron_count = len(crow) - 1
            edge_count = len(column)
            if len(superclasses) != neuron_count:
                raise ValueError("node annotations and graph have different lengths")
            if trainable["core.edge_gain"].shape != (edge_count,):
                raise ValueError("edge gain count differs from graph")
            if trainable["core.leak"].shape != (neuron_count,):
                raise ValueError("leak count differs from graph")
            if trainable["readout.weight"].shape != (3, len(motor_indices)):
                raise ValueError("readout shape differs from motor population")

            arrays = {
                "row_offsets": write_array(
                    temporary, "row_offsets.u32.bin", crow, "u32"
                ),
                "column_indices": write_array(
                    temporary, "column_indices.u32.bin", column, "u32"
                ),
            }
            edge_path = temporary / "edge_weights.f32.bin"
            fused_edges(
                crow,
                counts,
                trainable["core.edge_gain"],
                edge_path,
            )
            arrays["edge_weights"] = array_entry(
                edge_path, "f32", (edge_count,)
            )

        leak = 0.05 + 0.90 * torch.sigmoid(trainable["core.leak"])
        arrays.update(
            {
                "leak": write_array(temporary, "leak.f32.bin", leak.numpy(), "f32"),
                "sensory_indices": write_array(
                    temporary,
                    "sensory_indices.u32.bin",
                    sensory_indices,
                    "u32",
                ),
                "sensory_feature_ids": write_array(
                    temporary,
                    "sensory_feature_ids.u8.bin",
                    sensory_feature_ids,
                    "u8",
                ),
                "sensory_signs": write_array(
                    temporary, "sensory_signs.i8.bin", sensory_signs, "i8"
                ),
                "motor_indices": write_array(
                    temporary, "motor_indices.u32.bin", motor_indices, "u32"
                ),
                "readout_weight": write_array(
                    temporary,
                    "readout_weight.f32.bin",
                    trainable["readout.weight"].numpy(),
                    "f32",
                ),
                "readout_bias": write_array(
                    temporary,
                    "readout_bias.f32.bin",
                    trainable["readout.bias"].numpy(),
                    "f32",
                ),
            }
        )

        manifest = seal_manifest(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "byte_order": "little",
                "compression": "none",
                "model": {
                    "architecture": "male-cns-rate-v1",
                    "id": f"male-cns-v1-seed-{seed}-selected",
                    "seed": seed,
                },
                "dimensions": {
                    "neurons": neuron_count,
                    "edges": edge_count,
                    "features": len(FEATURE_NAMES),
                    "sensory_neurons": len(sensory_indices),
                    "motor_neurons": len(motor_indices),
                    "outputs": 3,
                },
                "recurrence": {
                    "equation": "state'=(1-leak)*state+leak*tanh(W*state+drive)",
                    "edge_orientation": "row=post,column=pre",
                    "updates_per_game_tick": 1,
                },
                "interface": {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_names": list(FEATURE_NAMES),
                    "sensory_assignment": "seeded-balanced-signed-v1",
                    "sensory_population": checkpoint["sensory_population"],
                    "motor_population": checkpoint["motor_population"],
                },
                "outputs": {
                    "order": ["target_x_logit", "target_y_logit", "strike_logit"],
                    "target_activation": "sigmoid",
                    "strike_activation": "sigmoid",
                    "strike_threshold": float(checkpoint["strike_threshold"]),
                },
                "arrays": arrays,
                "source": {
                    "checkpoint_sha256": checkpoint_sha256,
                    "graph_sha256": graph_sha256,
                    "graph_provenance": checkpoint["graph_provenance"],
                    "selection": checkpoint["selection"],
                    "fusions": [
                        "edge_weight=normalized_synapse_count*(0.05+0.90*sigmoid(edge_gain))",
                        "leak=0.05+0.90*sigmoid(trained_leak)",
                    ],
                },
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.out)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise

    total_bytes = sum(entry["byte_length"] for entry in manifest["arrays"].values())
    print(
        json.dumps(
            {
                "package": str(args.out),
                "package_sha256": manifest["package_sha256"],
                "arrays": len(manifest["arrays"]),
                "bytes": total_bytes,
                "mib": total_bytes / (1024 * 1024),
                "checkpoint_sha256": checkpoint_sha256,
                "graph_sha256": graph_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
