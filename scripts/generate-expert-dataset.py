#!/usr/bin/env python3
"""Generate a versioned Parquet behavior-cloning dataset from the expert."""

from __future__ import annotations

import argparse
from pathlib import Path

from flybrain_training.curriculum import dataset_curriculum
from flybrain_training.expert import InterceptExpertV1
from flybrain_training.rollout import aggregate_metrics, evaluate, write_metrics, write_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--split",
        choices=("evaluation", "train", "validation", "robustness"),
        default="evaluation",
    )
    args = parser.parse_args()

    rollouts = evaluate(dataset_curriculum(args.split), InterceptExpertV1())
    write_parquet(rollouts, args.out)
    metrics = aggregate_metrics(rollouts)
    if args.metrics:
        write_metrics(metrics, args.metrics)
    row_count = sum(len(rollout.rows) for rollout in rollouts)
    print(f"Wrote {row_count} rows across {len(rollouts)} episodes to {args.out}")


if __name__ == "__main__":
    main()
