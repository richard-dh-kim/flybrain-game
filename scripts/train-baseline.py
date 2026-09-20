#!/usr/bin/env python3
"""Train and evaluate a bounded MLP or GRU behavior-cloning baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
from torch import nn
import flybrain_game

from flybrain_training.curriculum import evaluation_curriculum, robustness_curriculum
from flybrain_training.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    normalized_target,
    observation_features,
    record_features,
    target_from_normalized,
)
from flybrain_training.models import GruPolicy, MlpPolicy
from flybrain_training.rollout import aggregate_metrics, run_episode
from flybrain_training.schema import (
    ActionV1,
    ObservationV1,
    synchronized_strike_is_ready,
)


TARGET_LOSS_WEIGHT = 4.0
STRIKE_LOSS_WEIGHT = 0.1


def load_episodes(path: Path) -> list[list[dict[str, Any]]]:
    rows = parquet.read_table(path).to_pylist()
    episodes: list[list[dict[str, Any]]] = []
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(int(row["episode_id"]), []).append(row)
    for episode_id in sorted(by_id):
        episode = sorted(by_id[episode_id], key=lambda row: int(row["tick"]))
        episodes.append(episode)
    return episodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_tensors(
    episodes: list[list[dict[str, Any]]],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    result = []
    for episode in episodes:
        features = torch.tensor([record_features(row) for row in episode], dtype=torch.float32)
        targets = torch.tensor(
            [
                normalized_target(
                    (row["action_target_x_units"], row["action_target_y_units"])
                )
                for row in episode
            ],
            dtype=torch.float32,
        )
        strikes = torch.tensor(
            [float(row["action_strike"]) for row in episode], dtype=torch.float32
        )
        result.append((features, targets, strikes))
    return result


def padded_batch(
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [episodes[index] for index in indices]
    length = max(item[0].shape[0] for item in selected)
    batch_size = len(selected)
    features = torch.zeros(batch_size, length, len(FEATURE_NAMES), device=device)
    targets = torch.zeros(batch_size, length, 2, device=device)
    strikes = torch.zeros(batch_size, length, device=device)
    mask = torch.zeros(batch_size, length, device=device)
    for index, (episode_features, episode_targets, episode_strikes) in enumerate(selected):
        count = episode_features.shape[0]
        features[index, :count] = episode_features.to(device)
        targets[index, :count] = episode_targets.to(device)
        strikes[index, :count] = episode_strikes.to(device)
        mask[index, :count] = 1.0
    return features, targets, strikes, mask


def combined_loss(
    predicted_targets: torch.Tensor,
    strike_logits: torch.Tensor,
    targets: torch.Tensor,
    strikes: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    target_per_step = (predicted_targets - targets).square().mean(dim=-1)
    target_loss = (target_per_step * mask).sum() / mask.sum()
    strike_per_step = nn.functional.binary_cross_entropy_with_logits(
        strike_logits,
        strikes,
        pos_weight=positive_weight,
        reduction="none",
    )
    strike_loss = (strike_per_step * mask).sum() / mask.sum()
    loss = TARGET_LOSS_WEIGHT * target_loss + STRIKE_LOSS_WEIGHT * strike_loss
    return loss, float(target_loss.detach()), float(strike_loss.detach())


def train_mlp(
    model: MlpPolicy,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    positive_weight: torch.Tensor,
) -> list[dict[str, float]]:
    features = torch.cat([item[0] for item in episodes]).to(device)
    targets = torch.cat([item[1] for item in episodes]).to(device)
    strikes = torch.cat([item[2] for item in episodes]).to(device)
    mask = torch.ones_like(strikes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    batch_size = 512
    for epoch in range(epochs):
        permutation = torch.randperm(features.shape[0], device=device)
        totals = [0.0, 0.0, 0.0]
        batches = 0
        model.train()
        for start in range(0, features.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            predicted_targets, strike_logits = model(features[indices])
            loss, target_loss, strike_loss = combined_loss(
                predicted_targets,
                strike_logits,
                targets[indices],
                strikes[indices],
                mask[indices],
                positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[0] += float(loss.detach())
            totals[1] += target_loss
            totals[2] += strike_loss
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": totals[0] / batches,
                "target_loss": totals[1] / batches,
                "strike_loss": totals[2] / batches,
            }
        )
    return history


def train_gru(
    model: GruPolicy,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    positive_weight: torch.Tensor,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    batch_size = 8
    for epoch in range(epochs):
        order = list(range(len(episodes)))
        random.shuffle(order)
        totals = [0.0, 0.0, 0.0]
        batches = 0
        model.train()
        for start in range(0, len(order), batch_size):
            batch = padded_batch(episodes, order[start : start + batch_size], device)
            features, targets, strikes, mask = batch
            predicted_targets, strike_logits, _ = model(features)
            loss, target_loss, strike_loss = combined_loss(
                predicted_targets,
                strike_logits,
                targets,
                strikes,
                mask,
                positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals[0] += float(loss.detach())
            totals[1] += target_loss
            totals[2] += strike_loss
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": totals[0] / batches,
                "target_loss": totals[1] / batches,
                "strike_loss": totals[2] / batches,
            }
        )
    return history


@torch.no_grad()
def offline_predictions(
    model: MlpPolicy | GruPolicy,
    model_kind: str,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted_targets = []
    expected_targets = []
    probabilities = []
    expected_strikes = []
    model.eval()
    for features, targets, strikes in episodes:
        if model_kind == "mlp":
            target_output, logits = model(features.to(device))
        else:
            target_output, logits, _ = model(features.unsqueeze(0).to(device))
            target_output = target_output.squeeze(0)
            logits = logits.squeeze(0)
        predicted_targets.append(target_output.cpu().numpy())
        expected_targets.append(targets.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        expected_strikes.append(strikes.numpy())
    return (
        np.concatenate(predicted_targets),
        np.concatenate(expected_targets),
        np.concatenate(probabilities),
        np.concatenate(expected_strikes),
    )


def choose_threshold(probabilities: np.ndarray, expected: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probabilities >= threshold
        true_positive = int(np.logical_and(predicted, expected == 1).sum())
        false_positive = int(np.logical_and(predicted, expected == 0).sum())
        false_negative = int(np.logical_and(~predicted, expected == 1).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def offline_metrics(
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], threshold: float
) -> dict[str, float | int]:
    predicted_targets, expected_targets, probabilities, expected_strikes = predictions
    width_pixels = (
        flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    height_pixels = (
        flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    target_errors = (predicted_targets - expected_targets) * np.array(
        [width_pixels, height_pixels]
    )
    predicted_strikes = probabilities >= threshold
    positive = expected_strikes == 1
    true_positive = int(np.logical_and(predicted_strikes, positive).sum())
    false_positive = int(np.logical_and(predicted_strikes, ~positive).sum())
    false_negative = int(np.logical_and(~predicted_strikes, positive).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
    return {
        "target_rmse_pixels": float(np.sqrt(np.square(target_errors).mean())),
        "strike_threshold": threshold,
        "strike_true_positive": true_positive,
        "strike_false_positive": false_positive,
        "strike_false_negative": false_negative,
        "strike_precision": precision,
        "strike_recall": recall,
    }


class LearnedPolicy:
    def __init__(
        self,
        model: MlpPolicy | GruPolicy,
        model_kind: str,
        device: torch.device,
        strike_threshold: float,
        carry_state: bool,
    ) -> None:
        self.model = model
        self.model_kind = model_kind
        self.device = device
        self.strike_threshold = strike_threshold
        self.carry_state = carry_state
        self.hidden: torch.Tensor | None = None

    @torch.no_grad()
    def decide(self, observation: ObservationV1) -> ActionV1:
        features = torch.tensor(
            observation_features(observation), dtype=torch.float32, device=self.device
        )
        if self.model_kind == "mlp":
            target, strike_logit = self.model(features.unsqueeze(0))
            target = target[0]
            strike_logit = strike_logit[0]
        else:
            target, strike_logit, hidden = self.model(
                features.reshape(1, 1, -1), self.hidden if self.carry_state else None
            )
            self.hidden = hidden if self.carry_state else None
            target = target[0, 0]
            strike_logit = strike_logit[0, 0]
        return ActionV1(
            target=target_from_normalized(target.cpu().tolist()),
            strike=(
                float(torch.sigmoid(strike_logit)) >= self.strike_threshold
                and synchronized_strike_is_ready(observation)
            ),
        )


def live_metrics(
    model: MlpPolicy | GruPolicy,
    model_kind: str,
    device: torch.device,
    threshold: float,
    carry_state: bool,
    suite: str,
) -> dict[str, Any]:
    rollouts = []
    curriculum = (
        robustness_curriculum() if suite == "robustness" else evaluation_curriculum()
    )
    for index, episode in enumerate(curriculum):
        policy = LearnedPolicy(model, model_kind, device, threshold, carry_state)
        rollouts.append(
            run_episode(
                index,
                episode.name,
                episode.trajectory,
                policy,
                initial_player_position=episode.initial_player_position,
            )
        )
    return aggregate_metrics(rollouts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "gru"), required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--live-suite", choices=("evaluation", "robustness"), default="evaluation"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )

    train_episodes = episode_tensors(load_episodes(args.train_data))
    validation_episodes = episode_tensors(load_episodes(args.validation_data))
    positive_count = sum(float(item[2].sum()) for item in train_episodes)
    row_count = sum(item[2].numel() for item in train_episodes)
    positive_weight = torch.tensor(
        min(100.0, (row_count - positive_count) / max(1.0, positive_count)),
        device=device,
    )
    model: MlpPolicy | GruPolicy
    if args.model == "mlp":
        model = MlpPolicy(len(FEATURE_NAMES))
    else:
        model = GruPolicy(len(FEATURE_NAMES))
    model.to(device)

    training_started = time.perf_counter()
    if args.model == "mlp":
        history = train_mlp(model, train_episodes, device, args.epochs, positive_weight)
    else:
        history = train_gru(model, train_episodes, device, args.epochs, positive_weight)
    training_seconds = time.perf_counter() - training_started

    validation_predictions = offline_predictions(
        model, args.model, validation_episodes, device
    )
    threshold = choose_threshold(validation_predictions[2], validation_predictions[3])
    validation = offline_metrics(validation_predictions, threshold)
    live_carried = live_metrics(
        model, args.model, device, threshold, carry_state=True, suite=args.live_suite
    )
    live_reset = live_metrics(
        model, args.model, device, threshold, carry_state=False, suite=args.live_suite
    )
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "epochs": args.epochs,
        "live_suite": args.live_suite,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "train_rows": row_count,
        "train_strike_labels": int(positive_count),
        "positive_weight": positive_weight.item(),
        "target_loss_weight": TARGET_LOSS_WEIGHT,
        "strike_loss_weight": STRIKE_LOSS_WEIGHT,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "train_dataset_sha256": sha256_file(args.train_data),
        "validation_dataset_sha256": sha256_file(args.validation_data),
        "final_train": history[-1],
        "validation": validation,
        "live_carried_state": live_carried,
        "live_reset_state": live_reset,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "strike_threshold": threshold,
            "seed": args.seed,
        },
        args.checkpoint,
    )
    metrics["checkpoint_sha256"] = sha256_file(args.checkpoint)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.model.upper()} on {device}: target RMSE "
        f"{validation['target_rmse_pixels']:.2f}px, strike P/R "
        f"{validation['strike_precision']:.3f}/{validation['strike_recall']:.3f}, "
        f"live carried {live_carried['hits']}/{live_carried['episodes']}, "
        f"reset {live_reset['hits']}/{live_reset['episodes']}"
    )


if __name__ == "__main__":
    main()
