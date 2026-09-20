#!/usr/bin/env python3
"""Export the validated row-wise u16 Phase 5 candidate as packed format v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from flybrain_training.packed_connectome import (
    FORMAT_NAME,
    PackedConnectome,
    seal_manifest,
    sha256_file,
    write_array,
)
from flybrain_training.quantized_connectome import quantize_rowwise


FORMAT_VERSION = 2
WEIGHT_BITS = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/connectome-packed-v1"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("runs/connectome-v1-quantization.metrics.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/connectome-packed-u16-v2"),
    )
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to replace existing package: {args.out}")
    source = PackedConnectome(args.source)
    validation = json.loads(args.validation.read_text())
    if validation["package_sha256"] != source.manifest["package_sha256"]:
        raise ValueError("quantization validation used another float32 package")
    candidate = validation["candidates"][str(WEIGHT_BITS)]
    if not candidate["preserves_fixed_suite_hits"]:
        raise ValueError("u16 candidate did not preserve fixed-suite hits")
    if candidate["zero_weight_edges"] != 0:
        raise ValueError("u16 candidate removed measured edges")
    if candidate["difference_from_float32"]["strike_decision_disagreements"] != 0:
        raise ValueError("u16 candidate changed held-out strike decisions")

    quantized = quantize_rowwise(
        source.edge_weights, source.row_offsets, WEIGHT_BITS
    )
    if np.count_nonzero(quantized.values == 0):
        raise ValueError("u16 export unexpectedly removed measured edges")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.out.name}-", dir=args.out.parent)
    )
    try:
        arrays = {}
        for name, entry in source.manifest["arrays"].items():
            if name == "edge_weights":
                continue
            destination = temporary / entry["file"]
            shutil.copyfile(args.source / entry["file"], destination)
            arrays[name] = deepcopy(entry)
        arrays["edge_weights"] = write_array(
            temporary,
            "edge_weights.u16.bin",
            quantized.values,
            "u16",
        )
        arrays["edge_scales"] = write_array(
            temporary,
            "edge_scales.f32.bin",
            quantized.scales,
            "f32",
        )

        manifest = deepcopy(source.manifest)
        manifest.pop("package_sha256")
        manifest["format"] = FORMAT_NAME
        manifest["format_version"] = FORMAT_VERSION
        manifest["arrays"] = arrays
        manifest["quantization"] = {
            "scheme": "rowwise-unsigned-max-v1",
            "weight_bits": WEIGHT_BITS,
            "levels": (1 << WEIGHT_BITS) - 1,
            "equation": "edge_weight=edge_scale[row]*edge_weight_u16",
            "zero_weight_edges": 0,
            "float32_reference_package_sha256": source.manifest["package_sha256"],
            "validation_sha256": sha256_file(args.validation),
            "validation_gate": {
                "held_out_strike_decision_disagreements": 0,
                "fixed_suite_hits": 33,
                "fixed_suite_episodes": 33,
            },
        }
        manifest["source"]["float32_reference_package_sha256"] = source.manifest[
            "package_sha256"
        ]
        manifest["source"]["quantization_validation_sha256"] = sha256_file(
            args.validation
        )
        manifest = seal_manifest(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.out)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise

    total_bytes = sum(entry["byte_length"] for entry in arrays.values())
    print(
        json.dumps(
            {
                "package": str(args.out),
                "format_version": FORMAT_VERSION,
                "package_sha256": manifest["package_sha256"],
                "arrays": len(arrays),
                "bytes": total_bytes,
                "mib": total_bytes / (1024 * 1024),
                "float32_reference_package_sha256": source.manifest[
                    "package_sha256"
                ],
                "validation_sha256": sha256_file(args.validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
