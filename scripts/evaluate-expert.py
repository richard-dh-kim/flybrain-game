#!/usr/bin/env python3
"""Evaluate the Phase 2 interception expert on deterministic trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from flybrain_training.curriculum import evaluation_curriculum
from flybrain_training.expert import InterceptExpertV1
from flybrain_training.rollout import aggregate_metrics, evaluate, write_metrics, write_replay_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--replay-html", type=Path)
    parser.add_argument("--replay-episode", default="turn-diagonal-60")
    args = parser.parse_args()

    expert = InterceptExpertV1()
    rollouts = evaluate(evaluation_curriculum(), expert)
    metrics = aggregate_metrics(rollouts)
    print(
        f"Expert hit {metrics['hits']}/{metrics['episodes']} episodes; "
        f"simple={metrics['simple_hits']}/{metrics['simple_episodes']}; "
        f"mean hit={metrics['mean_hit_ticks']:.1f} ticks; "
        f"cooldown violations={metrics['cooldown_violations']}."
    )
    for rollout in rollouts:
        summary = rollout.summary
        print(
            f"  {summary.trajectory}: status={summary.status} ticks={summary.elapsed_ticks} "
            f"strikes={summary.accepted_strikes} misses={summary.completed_misses} "
            f"gap={summary.closest_edge_gap_units:.1f}"
        )

    if args.metrics:
        write_metrics(metrics, args.metrics)
    if args.replay_html:
        replay = next(
            (item for item in rollouts if item.summary.trajectory == args.replay_episode),
            None,
        )
        if replay is None:
            raise SystemExit(f"unknown replay episode: {args.replay_episode}")
        write_replay_html(replay, args.replay_html)

    if metrics["simple_hit_rate"] < 0.9:
        raise SystemExit("expert did not reach the 90% simple-trajectory gate")
    if metrics["cooldown_violations"] != 0:
        raise SystemExit("expert issued a strike during an unavailable phase")


if __name__ == "__main__":
    main()
