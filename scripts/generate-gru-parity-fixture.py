#!/usr/bin/env python3
"""Generate fixed PyTorch outputs for the TypeScript GRU parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet
import torch

from flybrain_training.features import FEATURE_NAMES, record_features
from flybrain_training.models import GruPolicy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GruPolicy(len(FEATURE_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows = parquet.read_table(args.data).to_pylist()
    first_episode_id = min(int(row["episode_id"]) for row in rows)
    episode = sorted(
        (row for row in rows if int(row["episode_id"]) == first_episode_id),
        key=lambda row: int(row["tick"]),
    )
    features = torch.tensor(
        [record_features(row) for row in episode], dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        recurrent, _ = model.recurrent(features)
        output = model.output(recurrent)

    frames = []
    for index, row in enumerate(episode):
        frames.append(
            {
                "tick": int(row["tick"]),
                "features": features[0, index].tolist(),
                "output": output[0, index].tolist(),
                "activity": recurrent[0, index].tolist(),
            }
        )
    payload = {
        "format": "flybrain-gru-parity-v1",
        "checkpointSha256": sha256_file(args.checkpoint),
        "datasetSha256": sha256_file(args.data),
        "absoluteTolerance": 2e-5,
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(frames)} GRU parity frames to {args.out}")


if __name__ == "__main__":
    main()
