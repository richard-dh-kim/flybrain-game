#!/usr/bin/env python3
"""Compare packed NumPy inference with the selected PyTorch controller."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from flybrain_training.connectome_policy import MaleCnsPolicy
from flybrain_training.packed_connectome import PackedConnectome, sha256_file


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/connectome-v1-selected.pt"),
    )
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    packed = PackedConnectome(args.package)
    graph_directory = resolve_graph_path(args.graph)
    graph_path = graph_directory / "graph.npz"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if packed.manifest["source"]["checkpoint_sha256"] != sha256_file(
        args.checkpoint
    ):
        raise ValueError("package was exported from another checkpoint")
    if packed.manifest["source"]["graph_sha256"] != sha256_file(graph_path):
        raise ValueError("package was exported from another graph")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    nodes = feather.read_table(graph_directory / "nodes.feather")
    superclasses = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_indices = np.flatnonzero(superclasses == "vnc_sensory")
    motor_indices = np.flatnonzero(superclasses == "vnc_motor")
    torch.manual_seed(int(checkpoint["seed"]))
    with np.load(graph_path) as graph:
        policy = MaleCnsPolicy(
            graph,
            sensory_indices,
            motor_indices,
            input_size=packed.feature_count,
            seed=int(checkpoint["seed"]),
        )
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(checkpoint["trainable_state_dict"][name])
    policy = policy.to(device).eval()

    generator = np.random.default_rng(20260920)
    features = generator.uniform(
        low=-1.0,
        high=1.0,
        size=(args.ticks, packed.feature_count),
    ).astype(np.float32)
    packed_state = packed.initial_state()
    torch_state = policy.initial_state(1, device=device)
    maximum_target_error = 0.0
    maximum_strike_logit_error = 0.0
    maximum_state_error = 0.0
    mean_state_error = 0.0
    with torch.no_grad():
        for tick_features in features:
            torch_target, torch_strike, torch_state = policy.step(
                torch.from_numpy(tick_features).unsqueeze(0).to(device),
                torch_state,
            )
            packed_target, packed_strike, packed_state = packed.step(
                tick_features, packed_state
            )
            reference_target = torch_target[0].cpu().numpy()
            reference_strike = float(torch_strike[0].cpu())
            reference_state = torch_state[:, 0].cpu().numpy()
            maximum_target_error = max(
                maximum_target_error,
                float(np.max(np.abs(packed_target - reference_target))),
            )
            maximum_strike_logit_error = max(
                maximum_strike_logit_error,
                abs(float(packed_strike) - reference_strike),
            )
            state_error = np.abs(packed_state - reference_state)
            maximum_state_error = max(maximum_state_error, float(state_error.max()))
            mean_state_error = max(mean_state_error, float(state_error.mean()))

    status = (
        "passed"
        if max(
            maximum_target_error,
            maximum_strike_logit_error,
            maximum_state_error,
        )
        <= args.atol
        else "failed"
    )
    metrics = {
        "status": status,
        "ticks": args.ticks,
        "device": str(device),
        "absolute_tolerance": args.atol,
        "maximum_target_error": maximum_target_error,
        "maximum_strike_logit_error": maximum_strike_logit_error,
        "maximum_state_error": maximum_state_error,
        "maximum_mean_state_error": mean_state_error,
        "package_sha256": packed.manifest["package_sha256"],
    }
    print(json.dumps(metrics, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
