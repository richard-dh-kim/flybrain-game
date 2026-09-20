"""Shared rules for matched connectome topology experiments."""

from __future__ import annotations

from collections.abc import Mapping
import math


THRESHOLD_SELECTION_RULE = (
    "maximize hits, then minimize mean hit ticks, completed misses, accepted "
    "strikes, and threshold"
)


def select_live_threshold(
    conditions: Mapping[str, Mapping[str, object]],
) -> float:
    """Choose one threshold using the predeclared closed-loop ordering."""

    if not conditions:
        raise ValueError("threshold sweep has no conditions")

    def ranking(item: tuple[str, Mapping[str, object]]) -> tuple[float, ...]:
        threshold_text, metrics = item
        mean_hit_ticks = metrics.get("mean_hit_ticks")
        return (
            -float(metrics["hits"]),
            float(mean_hit_ticks) if mean_hit_ticks is not None else math.inf,
            float(metrics["completed_misses"]),
            float(metrics["accepted_strikes"]),
            float(threshold_text),
        )

    return float(min(conditions.items(), key=ranking)[0])
