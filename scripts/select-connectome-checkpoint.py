#!/usr/bin/env python3
"""Freeze a trained checkpoint with a separately selected live threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_sha256 = sha256_file(args.source)
    sweep = json.loads(args.sweep.read_text())
    validation = json.loads(args.validation.read_text())
    if sweep["checkpoint_sha256"] != source_sha256:
        raise ValueError("threshold sweep used a different source checkpoint")
    if validation["checkpoint_sha256"] != source_sha256:
        raise ValueError("live validation used a different source checkpoint")
    key = str(args.threshold)
    if key not in sweep["trained_conditions"]:
        raise ValueError("selected threshold is absent from the sweep")
    maximum_hits = max(
        condition["hits"] for condition in sweep["trained_conditions"].values()
    )
    if sweep["trained_conditions"][key]["hits"] != maximum_hits:
        raise ValueError("selected threshold did not maximize sweep hits")
    if validation["status"] != "passed" or not validation["beats_initialization"]:
        raise ValueError("selected threshold did not pass live validation")
    if validation["trained_thresholds"] != [args.threshold]:
        raise ValueError("live validation used a different threshold")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    checkpoint["strike_threshold"] = args.threshold
    checkpoint["graph_provenance"] = {
        "dataset": "male-cns:v1.0",
        "license": "CC-BY-4.0",
        "source_page": "https://male-cns.janelia.org/download/",
        "selection": "annotations.status == 'Traced'",
    }
    checkpoint["selection"] = {
        "source_checkpoint_sha256": source_sha256,
        "threshold_sweep_sha256": sha256_file(args.sweep),
        "live_validation_sha256": sha256_file(args.validation),
        "criterion": "maximum hits on 12-episode sweep, then fixed 33-episode validation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(
        json.dumps(
            {
                "checkpoint": str(args.out),
                "checkpoint_sha256": sha256_file(args.out),
                "source_checkpoint_sha256": source_sha256,
                "strike_threshold": args.threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
