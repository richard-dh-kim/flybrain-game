#!/usr/bin/env python3
"""Compare native Rust packed inference with the NumPy reference runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from flybrain_training.packed_connectome import (
    FORMAT_NAME,
    PackedConnectome,
    package_digest,
    sha256_file,
)


def features_for_ticks(ticks: int, feature_count: int) -> list[np.ndarray]:
    state = 20_260_920
    result = []
    for _ in range(ticks):
        features = np.empty(feature_count, dtype=np.float32)
        for index in range(feature_count):
            state ^= (state << 13) & 0xFFFF_FFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFF_FFFF
            state &= 0xFFFF_FFFF
            upper = (state >> 16) & 0xFFFF
            unit = np.float32(upper) / np.float32(65_535)
            features[index] = unit * np.float32(2) - np.float32(1)
        result.append(features)
    return result


def runtime_weights(
    reference: PackedConnectome, runtime_package: Path
) -> tuple[np.ndarray, dict[str, object]]:
    manifest = json.loads((runtime_package / "manifest.json").read_text())
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError("runtime package uses another format")
    if manifest.get("package_sha256") != package_digest(manifest):
        raise ValueError("runtime package manifest checksum mismatch")
    for name, entry in manifest["arrays"].items():
        path = runtime_package / entry["file"]
        if path.stat().st_size != entry["byte_length"]:
            raise ValueError(f"runtime array byte length mismatch: {name}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"runtime array checksum mismatch: {name}")

    version = int(manifest["format_version"])
    if version == 1:
        if manifest["package_sha256"] != reference.manifest["package_sha256"]:
            raise ValueError("v1 runtime package differs from the reference")
        return np.asarray(reference.edge_weights), manifest
    if version != 2:
        raise ValueError(f"unsupported runtime package version: {version}")
    if (
        manifest["quantization"]["float32_reference_package_sha256"]
        != reference.manifest["package_sha256"]
    ):
        raise ValueError("v2 runtime package names another float32 reference")

    quantized = np.fromfile(
        runtime_package / manifest["arrays"]["edge_weights"]["file"],
        dtype="<u2",
    )
    scales = np.fromfile(
        runtime_package / manifest["arrays"]["edge_scales"]["file"],
        dtype="<f4",
    )
    if quantized.shape != (reference.edge_count,):
        raise ValueError("v2 edge weight count differs from the reference")
    if scales.shape != (reference.neuron_count,):
        raise ValueError("v2 edge scale count differs from the reference")
    offsets = np.asarray(reference.row_offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    dequantized = np.empty(reference.edge_count, dtype=np.float32)
    for row_start in range(0, reference.neuron_count, 4096):
        row_end = min(row_start + 4096, reference.neuron_count)
        edge_start = int(offsets[row_start])
        edge_end = int(offsets[row_end])
        edge_scales = np.repeat(scales[row_start:row_end], lengths[row_start:row_end])
        dequantized[edge_start:edge_end] = (
            quantized[edge_start:edge_end].astype(np.float32) * edge_scales
        )
    return dequantized, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runtime-package", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/examples/benchmark"),
    )
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    packed = PackedConnectome(args.package)
    runtime_package = args.runtime_package or args.package
    packed.edge_weights, runtime_manifest = runtime_weights(packed, runtime_package)
    state = packed.initial_state()
    expected_target = None
    expected_strike_logit = None
    for features in features_for_ticks(args.ticks, packed.feature_count):
        expected_target, expected_strike_logit, state = packed.step(features, state)
    if expected_target is None or expected_strike_logit is None:
        raise ValueError("ticks must be positive")

    with tempfile.TemporaryDirectory() as temporary:
        state_path = Path(temporary) / "rust-state.f32.bin"
        completed = subprocess.run(
            [
                str(args.binary),
                str(runtime_package),
                str(args.ticks),
                str(state_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        native = json.loads(completed.stdout)
        rust_state = np.fromfile(state_path, dtype="<f4")

    if rust_state.shape != state.shape:
        raise ValueError("Rust runtime returned a different state shape")
    rust_target = np.asarray(native["final_target"], dtype=np.float32)
    target_error = float(np.max(np.abs(rust_target - expected_target)))
    strike_logit_error = abs(
        float(native["final_strike_logit"]) - float(expected_strike_logit)
    )
    state_error = np.abs(rust_state - state)
    maximum_state_error = float(state_error.max())
    mean_state_error = float(state_error.mean())
    maximum_error = max(target_error, strike_logit_error, maximum_state_error)
    status = "passed" if maximum_error <= args.atol else "failed"
    metrics = {
        "status": status,
        "ticks": args.ticks,
        "absolute_tolerance": args.atol,
        "maximum_target_error": target_error,
        "strike_logit_error": strike_logit_error,
        "maximum_state_error": maximum_state_error,
        "mean_state_error": mean_state_error,
        "float32_reference_package_sha256": packed.manifest["package_sha256"],
        "runtime_package_sha256": runtime_manifest["package_sha256"],
        "native_benchmark": native,
    }
    print(json.dumps(metrics, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
