#!/usr/bin/env python3
"""Compare row-wise weight quantization with the exact packed controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import flybrain_game
import numpy as np
import pyarrow.parquet as parquet
import torch

from flybrain_training.curriculum import validation_curriculum
from flybrain_training.features import (
    observation_features,
    normalized_target,
    record_features,
    target_from_normalized,
)
from flybrain_training.packed_connectome import PackedConnectome
from flybrain_training.quantized_connectome import (
    FrozenConnectomePolicy,
    quantize_rowwise,
)
from flybrain_training.rollout import aggregate_metrics, run_episode
from flybrain_training.schema import (
    ActionV1,
    ObservationV1,
    synchronized_strike_is_ready,
)


def validation_episodes(
    path: Path,
) -> list[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
    rows = parquet.read_table(path).to_pylist()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    episodes = []
    for episode_id in sorted(grouped):
        episode = sorted(grouped[episode_id], key=lambda row: int(row["tick"]))
        episodes.append(
            (
                torch.tensor(
                    [record_features(row) for row in episode], dtype=torch.float32
                ),
                np.asarray(
                    [
                        normalized_target(
                            (row["action_target_x_units"], row["action_target_y_units"])
                        )
                        for row in episode
                    ],
                    dtype=np.float32,
                ),
                np.asarray(
                    [float(row["action_strike"]) for row in episode],
                    dtype=np.float32,
                ),
            )
        )
    return episodes


def construct_policy(
    packed: PackedConnectome,
    edge_weights: object,
    device: torch.device,
) -> FrozenConnectomePolicy:
    return FrozenConnectomePolicy(
        row_offsets=packed.row_offsets,
        column_indices=packed.column_indices,
        edge_weights=edge_weights,
        leak=packed.leak,
        sensory_indices=packed.sensory_indices,
        sensory_feature_ids=packed.sensory_feature_ids,
        sensory_signs=packed.sensory_signs,
        motor_indices=packed.motor_indices,
        readout_weight=packed.readout_weight,
        readout_bias=packed.readout_bias,
        feature_count=packed.feature_count,
        device=device,
    )


@torch.no_grad()
def offline_outputs(
    policy: FrozenConnectomePolicy,
    episodes: list[tuple[torch.Tensor, np.ndarray, np.ndarray]],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_parts = []
    strike_parts = []
    final_states = []
    expected_target_parts = []
    expected_strike_parts = []
    for index, (features, expected_targets, expected_strikes) in enumerate(episodes):
        targets, strikes, state = policy.forward(features.unsqueeze(0).to(policy.device))
        target_parts.append(targets[0].cpu().numpy())
        strike_parts.append(strikes[0].cpu().numpy())
        final_states.append(state[:, 0].cpu().numpy())
        expected_target_parts.append(expected_targets)
        expected_strike_parts.append(expected_strikes)
        print(
            json.dumps(
                {
                    "stage": "offline",
                    "condition": label,
                    "episode": index + 1,
                    "episodes": len(episodes),
                }
            ),
            flush=True,
        )
    return (
        np.concatenate(target_parts),
        np.concatenate(strike_parts),
        np.stack(final_states),
        np.concatenate(expected_target_parts),
        np.concatenate(expected_strike_parts),
    )


def held_out_metrics(
    targets: np.ndarray,
    strike_logits: np.ndarray,
    expected_targets: np.ndarray,
    expected_strikes: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    width_pixels = (
        flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    height_pixels = (
        flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    errors = (targets - expected_targets) * np.array(
        [width_pixels, height_pixels], dtype=np.float32
    )
    probabilities = 1 / (1 + np.exp(-strike_logits))
    predicted = probabilities >= threshold
    expected = expected_strikes == 1
    true_positive = int(np.logical_and(predicted, expected).sum())
    false_positive = int(np.logical_and(predicted, ~expected).sum())
    false_negative = int(np.logical_and(~predicted, expected).sum())
    return {
        "rows": len(expected_strikes),
        "target_rmse_pixels": float(np.sqrt(np.square(errors).mean())),
        "strike_true_positive": true_positive,
        "strike_false_positive": false_positive,
        "strike_false_negative": false_negative,
        "strike_positive_decisions": int(predicted.sum()),
    }


class LiveFrozenPolicy:
    def __init__(self, policy: FrozenConnectomePolicy, threshold: float) -> None:
        self.policy = policy
        self.threshold = threshold
        self.state: torch.Tensor | None = None
        self.latencies_seconds: list[float] = []

    @torch.no_grad()
    def decide(self, observation: ObservationV1) -> ActionV1:
        started = time.perf_counter()
        features = torch.tensor(
            observation_features(observation),
            dtype=torch.float32,
            device=self.policy.device,
        ).unsqueeze(0)
        if self.state is None:
            self.state = self.policy.initial_state(1)
        target, strike_logit, self.state = self.policy.step(features, self.state)
        target_units = target_from_normalized(target[0].cpu().tolist())
        strike_probability = float(torch.sigmoid(strike_logit[0]).cpu())
        self.latencies_seconds.append(time.perf_counter() - started)
        return ActionV1(
            target=target_units,
            strike=(
                strike_probability >= self.threshold
                and synchronized_strike_is_ready(observation)
            ),
        )


def live_metrics(
    policy: FrozenConnectomePolicy,
    threshold: float,
    maximum_ticks: int,
    label: str,
) -> dict[str, Any]:
    curriculum = validation_curriculum()
    rollouts = []
    latencies = []
    for index, episode in enumerate(curriculum):
        live_policy = LiveFrozenPolicy(policy, threshold)
        rollout = run_episode(
            index,
            episode.name,
            episode.trajectory,
            live_policy,
            maximum_ticks=maximum_ticks,
            initial_player_position=episode.initial_player_position,
        )
        rollouts.append(rollout)
        latencies.extend(live_policy.latencies_seconds)
        print(
            json.dumps(
                {
                    "stage": "live",
                    "condition": label,
                    "episode": index + 1,
                    "episodes": len(curriculum),
                    "hit": rollout.summary.hit,
                    "ticks": rollout.summary.elapsed_ticks,
                }
            ),
            flush=True,
        )
    metrics = aggregate_metrics(rollouts)
    metrics["inference_calls"] = len(latencies)
    metrics["inference_median_ms"] = float(np.median(latencies) * 1000)
    metrics["inference_p95_ms"] = float(np.percentile(latencies, 95) * 1000)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("artifacts/connectome-packed-v1"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("data/behavior-cloning-v1/validation.parquet"),
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--maximum-ticks", type=int, default=1800)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/connectome-v1-quantization.metrics.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full quantization evaluation requires CUDA")

    packed = PackedConnectome(args.package)
    episodes = validation_episodes(args.validation_data)
    threshold = packed.strike_threshold
    exact_policy = construct_policy(packed, packed.edge_weights, device)
    exact_targets, exact_strikes, exact_states, expected_targets, expected_strikes = (
        offline_outputs(exact_policy, episodes, "float32")
    )
    exact_offline = held_out_metrics(
        exact_targets,
        exact_strikes,
        expected_targets,
        expected_strikes,
        threshold,
    )
    del exact_policy
    torch.cuda.empty_cache()

    exact_array_bytes = sum(
        int(entry["byte_length"]) for entry in packed.manifest["arrays"].values()
    )
    edge_float_bytes = int(packed.manifest["arrays"]["edge_weights"]["byte_length"])
    candidates = {}
    for bits in args.bits:
        quantized = quantize_rowwise(packed.edge_weights, packed.row_offsets, bits)
        label = f"rowwise-u{bits}"
        policy = construct_policy(packed, quantized.dequantized, device)
        targets, strikes, states, _, _ = offline_outputs(policy, episodes, label)
        target_difference = np.abs(targets - exact_targets)
        strike_difference = np.abs(strikes - exact_strikes)
        state_difference = np.abs(states - exact_states)
        live = live_metrics(policy, threshold, args.maximum_ticks, label)
        quantized_array_bytes = int(quantized.values.nbytes + quantized.scales.nbytes)
        candidate_package_bytes = (
            exact_array_bytes - edge_float_bytes + quantized_array_bytes
        )
        candidates[str(bits)] = {
            "label": label,
            "weight_storage_bytes": quantized_array_bytes,
            "estimated_package_bytes": candidate_package_bytes,
            "estimated_package_mib": candidate_package_bytes / (1024 * 1024),
            "zero_weight_edges": int(np.count_nonzero(quantized.values == 0)),
            "weight_maximum_absolute_error": float(
                np.max(np.abs(quantized.dequantized - packed.edge_weights))
            ),
            "held_out": held_out_metrics(
                targets,
                strikes,
                expected_targets,
                expected_strikes,
                threshold,
            ),
            "difference_from_float32": {
                "target_maximum_absolute": float(target_difference.max()),
                "target_mean_absolute": float(target_difference.mean()),
                "strike_logit_maximum_absolute": float(strike_difference.max()),
                "strike_logit_mean_absolute": float(strike_difference.mean()),
                "final_state_maximum_absolute": float(state_difference.max()),
                "final_state_mean_absolute": float(state_difference.mean()),
                "strike_decision_disagreements": int(
                    np.count_nonzero(
                        (1 / (1 + np.exp(-strikes)) >= threshold)
                        != (1 / (1 + np.exp(-exact_strikes)) >= threshold)
                    )
                ),
            },
            "live": live,
            "preserves_fixed_suite_hits": int(live["hits"]) == 33,
        }
        del policy, quantized
        torch.cuda.empty_cache()

    metrics = {
        "experiment": "Phase 5 row-wise connectome weight quantization",
        "status": "measured",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "package_sha256": packed.manifest["package_sha256"],
        "threshold": threshold,
        "exact_package_bytes": exact_array_bytes,
        "exact_held_out": exact_offline,
        "reference_live": {
            "source": "docs/experiments/phase-4-male-cns-v1.metrics.json",
            "hits": 33,
            "episodes": 33,
            "mean_hit_ticks": 116.45454545454545,
        },
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
