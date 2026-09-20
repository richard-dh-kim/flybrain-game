#!/usr/bin/env python3
"""Export the diagnostic GRU checkpoint for Phase 3 browser playback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def nested_list(tensor: torch.Tensor) -> list[object]:
    return tensor.detach().cpu().to(torch.float32).tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "gru":
        raise SystemExit("only GRU checkpoints are supported by this exporter")
    state = checkpoint["state_dict"]
    payload = {
        "format": "flybrain-diagnostic-gru-json-v1",
        "modelId": "gru-v1",
        "checkpointSha256": sha256_file(args.checkpoint),
        "featureSchemaVersion": checkpoint["feature_schema_version"],
        "featureNames": checkpoint["feature_names"],
        "strikeThreshold": checkpoint["strike_threshold"],
        "inputSize": state["recurrent.weight_ih_l0"].shape[1],
        "hiddenSize": state["recurrent.weight_hh_l0"].shape[1],
        "weightInputHidden": nested_list(state["recurrent.weight_ih_l0"]),
        "weightHiddenHidden": nested_list(state["recurrent.weight_hh_l0"]),
        "biasInputHidden": nested_list(state["recurrent.bias_ih_l0"]),
        "biasHiddenHidden": nested_list(state["recurrent.bias_hh_l0"]),
        "weightOutput": nested_list(state["output.weight"]),
        "biasOutput": nested_list(state["output.bias"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Exported {args.checkpoint} to {args.out}")


if __name__ == "__main__":
    main()
