#!/usr/bin/env python3
"""Run a bounded behavior-cloning smoke test through the full MaleCNS graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import torch
from torch import nn
import flybrain_game

from flybrain_training.connectome_policy import MaleCnsPolicy
from flybrain_training.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    normalized_target,
    record_features,
)


TARGET_LOSS_WEIGHT = 4.0
STRIKE_LOSS_WEIGHT = 0.1
EXPECTED_GRAPH_SHA256 = "eff4093bf53c4dd17d7ee4f2f838f6ae5ede70570f9a91317dd8824cca5c771d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology_digest(crow: np.ndarray, column: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (crow, column):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def resolve_graph_path(requested: Path | None) -> Path:
    candidates = [
        requested,
        Path(os.environ["FLYBRAIN_GRAPH"]) if "FLYBRAIN_GRAPH" in os.environ else None,
        Path("data/graph-traced-v1"),
        Path("/tmp/flybrain-upstream-review/flyhard/data/graph-traced-v1"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "graph.npz").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MaleCNS graph not found; pass --graph or set FLYBRAIN_GRAPH"
    )


def load_episodes(
    path: Path,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows = parquet.read_table(path).to_pylist()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    result = []
    for episode_id in sorted(grouped):
        episode = sorted(grouped[episode_id], key=lambda row: int(row["tick"]))
        result.append(
            (
                torch.tensor(
                    [record_features(row) for row in episode], dtype=torch.float32
                ),
                torch.tensor(
                    [
                        normalized_target(
                            (row["action_target_x_units"], row["action_target_y_units"])
                        )
                        for row in episode
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [float(row["action_strike"]) for row in episode],
                    dtype=torch.float32,
                ),
            )
        )
    if not result:
        raise ValueError(f"no episodes in {path}")
    return result


def combined_loss(
    predicted_targets: torch.Tensor,
    strike_logits: torch.Tensor,
    targets: torch.Tensor,
    strikes: torch.Tensor,
    positive_weight: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_per_tick = (predicted_targets - targets).square().mean(dim=-1)
    strike_per_tick = nn.functional.binary_cross_entropy_with_logits(
        strike_logits,
        strikes,
        pos_weight=positive_weight,
        reduction="none",
    )
    if mask is None:
        target_loss = target_per_tick.mean()
        strike_loss = strike_per_tick.mean()
    else:
        denominator = mask.sum().clamp_min(1)
        target_loss = (target_per_tick * mask).sum() / denominator
        strike_loss = (strike_per_tick * mask).sum() / denominator
    loss = TARGET_LOSS_WEIGHT * target_loss + STRIKE_LOSS_WEIGHT * strike_loss
    return loss, target_loss, strike_loss


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_probe(
    policy: MaleCnsPolicy,
    episode: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    positive_weight: torch.Tensor,
    ticks: int,
    *,
    carry_state: bool,
) -> dict[str, float | int]:
    policy.eval()
    features, targets, strikes = episode
    count = min(ticks, len(features))
    features = features[:count].unsqueeze(0).to(device)
    targets = targets[:count].unsqueeze(0).to(device)
    strikes = strikes[:count].unsqueeze(0).to(device)
    if carry_state:
        predicted_targets, strike_logits, state = policy(features)
    else:
        target_parts = []
        strike_parts = []
        state = policy.initial_state(1, device=device)
        for tick in range(count):
            target, strike, state = policy(features[:, tick : tick + 1])
            target_parts.append(target)
            strike_parts.append(strike)
        predicted_targets = torch.cat(target_parts, dim=1)
        strike_logits = torch.cat(strike_parts, dim=1)
    loss, target_loss, strike_loss = combined_loss(
        predicted_targets, strike_logits, targets, strikes, positive_weight
    )
    return {
        "ticks": count,
        "loss": float(loss),
        "target_loss": float(target_loss),
        "strike_loss": float(strike_loss),
        "state_mean_absolute": float(state.abs().mean()),
        "state_nonzero": int(torch.count_nonzero(state)),
    }


@torch.no_grad()
def evaluate_validation(
    policy: MaleCnsPolicy,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    carry_state: bool,
) -> dict[str, float | int]:
    policy.eval()
    predicted_parts = []
    target_parts = []
    probability_parts = []
    strike_parts = []
    for features, targets, strikes in episodes:
        device_features = features.unsqueeze(0).to(device)
        if carry_state:
            predicted, logits, _ = policy(device_features)
            predicted = predicted.squeeze(0)
            logits = logits.squeeze(0)
        else:
            predicted_ticks = []
            logit_ticks = []
            for tick in range(len(features)):
                tick_target, tick_logit, _ = policy(
                    device_features[:, tick : tick + 1]
                )
                predicted_ticks.append(tick_target[0, 0])
                logit_ticks.append(tick_logit[0, 0])
            predicted = torch.stack(predicted_ticks)
            logits = torch.stack(logit_ticks)
        predicted_parts.append(predicted.cpu().numpy())
        target_parts.append(targets.numpy())
        probability_parts.append(torch.sigmoid(logits).cpu().numpy())
        strike_parts.append(strikes.numpy())

    predicted_targets = np.concatenate(predicted_parts)
    expected_targets = np.concatenate(target_parts)
    probabilities = np.concatenate(probability_parts)
    expected_strikes = np.concatenate(strike_parts)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted_strikes = probabilities >= threshold
        positive = expected_strikes == 1
        true_positive = int(np.logical_and(predicted_strikes, positive).sum())
        false_positive = int(np.logical_and(predicted_strikes, ~positive).sum())
        false_negative = int(np.logical_and(~predicted_strikes, positive).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    predicted_strikes = probabilities >= best_threshold
    positive = expected_strikes == 1
    true_positive = int(np.logical_and(predicted_strikes, positive).sum())
    false_positive = int(np.logical_and(predicted_strikes, ~positive).sum())
    false_negative = int(np.logical_and(~predicted_strikes, positive).sum())
    width_pixels = (
        flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    height_pixels = (
        flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    ) / flybrain_game.UNITS_PER_PIXEL
    pixel_errors = (predicted_targets - expected_targets) * np.array(
        [width_pixels, height_pixels]
    )
    return {
        "rows": len(expected_strikes),
        "target_rmse_pixels": float(np.sqrt(np.square(pixel_errors).mean())),
        "strike_threshold": best_threshold,
        "strike_true_positive": true_positive,
        "strike_false_positive": false_positive,
        "strike_false_negative": false_negative,
        "strike_precision": (
            true_positive / (true_positive + false_positive) if true_positive else 0.0
        ),
        "strike_recall": (
            true_positive / (true_positive + false_negative) if true_positive else 0.0
        ),
    }


def gradient_audit(policy: MaleCnsPolicy) -> dict[str, dict[str, float | int | bool]]:
    result = {}
    for name, parameter in policy.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            result[name] = {"finite": False, "nonzero": 0, "total": parameter.numel()}
            continue
        result[name] = {
            "finite": bool(torch.isfinite(gradient).all()),
            "nonzero": int(torch.count_nonzero(gradient)),
            "total": parameter.numel(),
            "norm": float(gradient.norm()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--graph", type=Path)
    parser.add_argument(
        "--train-data", type=Path, default=Path("data/behavior-cloning-v1/train.parquet")
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("data/behavior-cloning-v1/validation.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/connectome-v1-smoke"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/connectome-v1-smoke.pt"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--probe-ticks", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "The full-graph smoke test requires CUDA by default; use --allow-cpu "
            "only when the much slower CPU run is intentional"
        )

    graph_directory = resolve_graph_path(args.graph)
    graph_path = graph_directory / "graph.npz"
    manifest = json.loads((graph_directory / "manifest.json").read_text())
    graph_sha256 = sha256_file(graph_path)
    if graph_sha256 != manifest["graph_sha256"]:
        raise ValueError("graph.npz does not match its manifest")
    if graph_sha256 != EXPECTED_GRAPH_SHA256:
        raise ValueError(f"unexpected MaleCNS graph: {graph_sha256}")

    started = time.perf_counter()
    nodes = feather.read_table(graph_directory / "nodes.feather")
    superclasses = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(superclasses == "vnc_sensory")
    motor_ids = np.flatnonzero(superclasses == "vnc_motor")
    with np.load(graph_path) as graph:
        body_ids = np.asarray(graph["body_ids"])
        node_body_ids = np.asarray(nodes["bodyId"])
        if not np.array_equal(body_ids, node_body_ids):
            raise ValueError("nodes.feather order does not match graph body_ids")
        topology_before = topology_digest(graph["crow"], graph["col"])
        policy = MaleCnsPolicy(
            graph,
            sensory_ids,
            motor_ids,
            input_size=len(FEATURE_NAMES),
            seed=args.seed,
        )
        neuron_count = policy.core.n
        edge_count = policy.core.column.numel()
    del nodes, superclasses, body_ids, node_body_ids
    resume_sha256 = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["graph_sha256"] != graph_sha256:
            raise ValueError("resume checkpoint uses a different connectome graph")
        if checkpoint["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("resume checkpoint uses a different feature schema")
        if checkpoint["seed"] != args.seed:
            raise ValueError("resume checkpoint uses a different interface seed")
        parameters = dict(policy.named_parameters())
        saved_parameters = checkpoint["trainable_state_dict"]
        if set(parameters) != set(saved_parameters):
            raise ValueError("resume checkpoint trainable parameter layout differs")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved_parameters[name])
        resume_sha256 = sha256_file(args.resume)
    policy.to(device)

    train_episodes = load_episodes(args.train_data)
    validation_episodes = load_episodes(args.validation_data)
    positive_count = sum(float(episode[2].sum()) for episode in train_episodes)
    row_count = sum(episode[2].numel() for episode in train_episodes)
    positive_weight = torch.tensor(
        min(100.0, (row_count - positive_count) / max(1.0, positive_count)),
        device=device,
    )
    initial_probe = {
        "carried_state": evaluate_probe(
            policy,
            validation_episodes[0],
            device,
            positive_weight,
            args.probe_ticks,
            carry_state=True,
        ),
        "reset_each_tick": evaluate_probe(
            policy,
            validation_episodes[0],
            device,
            positive_weight,
            args.probe_ticks,
            carry_state=False,
        ),
    }

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    order = list(range(len(train_episodes)))
    random.shuffle(order)
    order_cursor = 0
    lane_episodes = []
    for _ in range(args.batch):
        lane_episodes.append(order[order_cursor])
        order_cursor = (order_cursor + 1) % len(order)
    lane_ticks = [0] * args.batch
    recurrent_state = policy.initial_state(args.batch, device=device)
    history: list[dict[str, Any]] = []
    gradients: dict[str, dict[str, float | int | bool]] | None = None
    policy.train()
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    for step in range(args.steps):
        if time.perf_counter() - training_started >= args.seconds:
            break
        feature_window = torch.zeros(
            args.batch, args.window, len(FEATURE_NAMES), device=device
        )
        target_window = torch.zeros(args.batch, args.window, 2, device=device)
        strike_window = torch.zeros(args.batch, args.window, device=device)
        mask = torch.zeros(args.batch, args.window, device=device)
        end_ticks = []
        trained_episode_ids = list(lane_episodes)
        for lane, episode_id in enumerate(lane_episodes):
            features, targets, strikes = train_episodes[episode_id]
            start = lane_ticks[lane]
            end = min(start + args.window, len(features))
            count = end - start
            feature_window[lane, :count] = features[start:end].to(device)
            target_window[lane, :count] = targets[start:end].to(device)
            strike_window[lane, :count] = strikes[start:end].to(device)
            mask[lane, :count] = 1
            end_ticks.append(end)

        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        tick = time.perf_counter()
        predicted_targets, strike_logits, next_state = policy(
            feature_window, recurrent_state
        )
        loss, target_loss, strike_loss = combined_loss(
            predicted_targets,
            strike_logits,
            target_window,
            strike_window,
            positive_weight,
            mask,
        )
        synchronize(device)
        forward_seconds = time.perf_counter() - tick
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite connectome objective")
        tick = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - tick
        if gradients is None:
            gradients = gradient_audit(policy)
            if not all(
                audit["finite"] and int(audit["nonzero"]) > 0
                for audit in gradients.values()
            ):
                raise RuntimeError(f"connectome gradient audit failed: {gradients}")
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        recurrent_state = next_state.detach()
        completed_episodes = 0
        for lane, end in enumerate(end_ticks):
            lane_ticks[lane] = end
            if end >= len(train_episodes[lane_episodes[lane]][0]):
                completed_episodes += 1
                if order_cursor == 0:
                    random.shuffle(order)
                lane_episodes[lane] = order[order_cursor]
                order_cursor = (order_cursor + 1) % len(order)
                lane_ticks[lane] = 0
                recurrent_state[:, lane] = 0
        record = {
            "step": step + 1,
            "episodes": trained_episode_ids,
            "episode_end_ticks": list(lane_ticks),
            "completed_episodes": completed_episodes,
            "training_rows": int(mask.sum()),
            "loss": float(loss.detach()),
            "target_loss": float(target_loss.detach()),
            "strike_loss": float(strike_loss.detach()),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "elapsed_seconds": time.perf_counter() - training_started,
        }
        history.append(record)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(json.dumps(record), flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    if not history or gradients is None:
        raise RuntimeError("the time bound expired before any optimizer step completed")
    training_seconds = time.perf_counter() - training_started
    final_probe = {
        "carried_state": evaluate_probe(
            policy,
            validation_episodes[0],
            device,
            positive_weight,
            args.probe_ticks,
            carry_state=True,
        ),
        "reset_each_tick": evaluate_probe(
            policy,
            validation_episodes[0],
            device,
            positive_weight,
            args.probe_ticks,
            carry_state=False,
        ),
    }
    full_validation = None
    if args.full_validation:
        full_validation = {
            "carried_state": evaluate_validation(
                policy, validation_episodes, device, carry_state=True
            ),
            "reset_each_tick": evaluate_validation(
                policy, validation_episodes, device, carry_state=False
            ),
        }
    topology_after = topology_digest(
        policy.core.crow.detach().cpu().numpy(),
        policy.core.column.detach().cpu().numpy(),
    )
    topology_unchanged = topology_before == topology_after
    if not topology_unchanged:
        raise RuntimeError("connectome topology changed during optimization")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": {
                name: parameter.detach().cpu()
                for name, parameter in policy.named_parameters()
            },
            "graph_sha256": graph_sha256,
            "graph_provenance": {
                "dataset": "male-cns:v1.0",
                "license": "CC-BY-4.0",
                "source_page": "https://male-cns.janelia.org/download/",
                "selection": "annotations.status == 'Traced'",
            },
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "sensory_population": "vnc_sensory",
            "motor_population": "vnc_motor",
            "seed": args.seed,
            "optimizer_steps": len(history),
            "strike_threshold": (
                full_validation["carried_state"]["strike_threshold"]
                if full_validation is not None
                else None
            ),
        },
        args.checkpoint,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    steady = history[1:] or history
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    metrics = {
        "experiment": "Phase 4 full MaleCNS behavior cloning",
        "run_label": args.label,
        "status": "passed",
        "claim": (
            "Engineering validation of full-graph task gradients and recurrent "
            "state only; not a trained gameplay result or biological model."
        ),
        "seed": args.seed,
        "resume_checkpoint": str(args.resume) if args.resume is not None else None,
        "resume_checkpoint_sha256": resume_sha256,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "graph_directory": str(graph_directory),
        "graph_sha256": graph_sha256,
        "neurons": neuron_count,
        "edges": edge_count,
        "sensory_population": "vnc_sensory",
        "sensory_neurons": len(sensory_ids),
        "motor_population": "vnc_motor",
        "motor_neurons": len(motor_ids),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_assignment": "frozen balanced random signed mapping",
        "output_interface": "trainable linear readout from vnc_motor neurons",
        "observation_to_action_bypass": False,
        "graph_updates_per_game_tick": 1,
        "recurrent_state_carried_across_windows": True,
        "training_batch_size": args.batch,
        "truncated_backpropagation_window": args.window,
        "trainable_parameters": parameter_count,
        "target_loss_weight": TARGET_LOSS_WEIGHT,
        "strike_loss_weight": STRIKE_LOSS_WEIGHT,
        "positive_weight": float(positive_weight),
        "learning_rate": args.learning_rate,
        "requested_optimizer_steps": args.steps,
        "completed_optimizer_steps": len(history),
        "time_limit_seconds": args.seconds,
        "training_seconds": training_seconds,
        "setup_seconds": setup_seconds,
        "train_rows": row_count,
        "processed_training_rows": sum(
            int(record["training_rows"]) for record in history
        ),
        "approximate_training_epochs": sum(
            int(record["training_rows"]) for record in history
        )
        / row_count,
        "train_strike_labels": int(positive_count),
        "train_dataset_sha256": sha256_file(args.train_data),
        "validation_dataset_sha256": sha256_file(args.validation_data),
        "gradient_audit": gradients,
        "topology_unchanged": topology_unchanged,
        "topology_digest": topology_after,
        "edge_gain_mean_absolute_update": float(policy.core.edge_gain.detach().abs().mean()),
        "leak_mean_absolute_update": float(policy.core.leak.detach().abs().mean()),
        "initial_probe": initial_probe,
        "final_probe": final_probe,
        "full_validation": full_validation,
        "first_training_loss": history[0]["loss"],
        "last_training_loss": history[-1]["loss"],
        "steady_forward_median_seconds": float(
            np.median([record["forward_seconds"] for record in steady])
        ),
        "steady_backward_median_seconds": float(
            np.median([record["backward_seconds"] for record in steady])
        ),
        "peak_gpu_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else None
        ),
        "peak_gpu_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1e9
            if device.type == "cuda"
            else None
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
