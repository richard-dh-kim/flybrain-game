"""Versioned, checksummed inference package for the MaleCNS controller."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_NAME = "flybrain-connectome-packed"
FORMAT_VERSION = 1

_DTYPES = {
    "f32": np.dtype("<f4"),
    "i8": np.dtype("i1"),
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "u32": np.dtype("<u4"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_digest(manifest: dict[str, Any]) -> str:
    """Hash canonical manifest content without the self-referential digest."""

    payload = deepcopy(manifest)
    payload.pop("package_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def seal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(manifest)
    result["package_sha256"] = package_digest(result)
    return result


def write_array(
    directory: Path,
    filename: str,
    values: object,
    dtype: str,
) -> dict[str, Any]:
    """Write a contiguous little-endian array and return its manifest entry."""

    if dtype not in _DTYPES:
        raise ValueError(f"unsupported packed dtype: {dtype}")
    array = np.ascontiguousarray(values, dtype=_DTYPES[dtype])
    path = directory / filename
    array.tofile(path)
    return {
        "file": filename,
        "dtype": dtype,
        "shape": list(array.shape),
        "byte_length": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class PackedConnectome:
    """Memory-mapped NumPy reference runtime for packed format v1."""

    def __init__(self, directory: Path | str, *, verify: bool = True) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != FORMAT_NAME:
            raise ValueError("not a FlyBrain packed connectome package")
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported packed format version: {self.manifest.get('format_version')}"
            )
        expected_package_digest = self.manifest.get("package_sha256")
        if expected_package_digest != package_digest(self.manifest):
            raise ValueError("packed manifest checksum mismatch")
        if verify:
            self._verify_files()

        self.row_offsets = self._array("row_offsets")
        self.column_indices = self._array("column_indices")
        self.edge_weights = self._array("edge_weights")
        self.leak = self._array("leak")
        self.sensory_indices = self._array("sensory_indices")
        self.sensory_feature_ids = self._array("sensory_feature_ids")
        self.sensory_signs = self._array("sensory_signs")
        self.motor_indices = self._array("motor_indices")
        self.readout_weight = self._array("readout_weight")
        self.readout_bias = self._array("readout_bias")

        dimensions = self.manifest["dimensions"]
        self.neuron_count = int(dimensions["neurons"])
        self.edge_count = int(dimensions["edges"])
        self.feature_count = int(dimensions["features"])
        self.strike_threshold = float(self.manifest["outputs"]["strike_threshold"])
        self._validate_layout()

    def _verify_files(self) -> None:
        for name, entry in self.manifest["arrays"].items():
            path = self.directory / entry["file"]
            if not path.is_file():
                raise ValueError(f"packed array is missing: {name}")
            if path.stat().st_size != int(entry["byte_length"]):
                raise ValueError(f"packed array byte length mismatch: {name}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"packed array checksum mismatch: {name}")

    def _array(self, name: str) -> np.memmap:
        entry = self.manifest["arrays"][name]
        dtype_name = entry["dtype"]
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype for {name}: {dtype_name}")
        shape = tuple(int(value) for value in entry["shape"])
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * _DTYPES[
            dtype_name
        ].itemsize
        if expected_bytes != int(entry["byte_length"]):
            raise ValueError(f"manifest shape and byte length disagree: {name}")
        return np.memmap(
            self.directory / entry["file"],
            dtype=_DTYPES[dtype_name],
            mode="r",
            shape=shape,
        )

    def _validate_layout(self) -> None:
        if self.row_offsets.shape != (self.neuron_count + 1,):
            raise ValueError("row offset shape does not match neuron count")
        if int(self.row_offsets[0]) != 0:
            raise ValueError("row offsets must begin at zero")
        if int(self.row_offsets[-1]) != self.edge_count:
            raise ValueError("row offsets do not end at edge count")
        if np.any(self.row_offsets[1:] < self.row_offsets[:-1]):
            raise ValueError("row offsets must be monotonic")
        if self.column_indices.shape != (self.edge_count,):
            raise ValueError("column index shape does not match edge count")
        if self.edge_weights.shape != (self.edge_count,):
            raise ValueError("edge weight shape does not match edge count")
        if self.edge_count and int(self.column_indices.max()) >= self.neuron_count:
            raise ValueError("column index exceeds neuron count")
        if self.leak.shape != (self.neuron_count,):
            raise ValueError("leak shape does not match neuron count")
        if self.sensory_indices.shape != self.sensory_feature_ids.shape:
            raise ValueError("sensory index and feature layouts differ")
        if self.sensory_indices.shape != self.sensory_signs.shape:
            raise ValueError("sensory index and sign layouts differ")
        if self.readout_weight.shape != (3, len(self.motor_indices)):
            raise ValueError("readout shape does not match motor population")
        if self.readout_bias.shape != (3,):
            raise ValueError("readout bias must contain three outputs")
        if len(self.sensory_indices) and int(self.sensory_indices.max()) >= self.neuron_count:
            raise ValueError("sensory index exceeds neuron count")
        if len(self.motor_indices) and int(self.motor_indices.max()) >= self.neuron_count:
            raise ValueError("motor index exceeds neuron count")
        if len(self.sensory_feature_ids) and int(self.sensory_feature_ids.max()) >= self.feature_count:
            raise ValueError("sensory feature index exceeds feature count")

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.neuron_count, dtype=np.float32)

    def step(
        self,
        features: object,
        state: object,
    ) -> tuple[np.ndarray, np.float32, np.ndarray]:
        feature_array = np.asarray(features, dtype=np.float32)
        state_array = np.asarray(state, dtype=np.float32)
        if feature_array.shape != (self.feature_count,):
            raise ValueError(
                f"features must have shape ({self.feature_count},), got {feature_array.shape}"
            )
        if state_array.shape != (self.neuron_count,):
            raise ValueError(
                f"state must have shape ({self.neuron_count},), got {state_array.shape}"
            )

        products = self.edge_weights * state_array[self.column_indices]
        signal = np.zeros(self.neuron_count, dtype=np.float32)
        starts = np.asarray(self.row_offsets[:-1], dtype=np.intp)
        ends = np.asarray(self.row_offsets[1:], dtype=np.intp)
        nonempty = ends > starts
        if np.any(nonempty):
            signal[nonempty] = np.add.reduceat(products, starts[nonempty])
        signal[self.sensory_indices] += (
            feature_array[self.sensory_feature_ids]
            * self.sensory_signs.astype(np.float32)
        )
        next_state = (
            (np.float32(1.0) - self.leak) * state_array
            + self.leak * np.tanh(signal)
        ).astype(np.float32, copy=False)
        raw = (
            self.readout_weight @ next_state[self.motor_indices] + self.readout_bias
        ).astype(np.float32, copy=False)
        target = (
            np.float32(1.0) / (np.float32(1.0) + np.exp(-raw[:2]))
        ).astype(np.float32, copy=False)
        return target, np.float32(raw[2]), next_state
