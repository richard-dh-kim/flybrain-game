#!/usr/bin/env python3
"""Compare a trained full MaleCNS controller with its initialization in-game."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import torch

from flybrain_training.connectome_policy import MaleCnsPolicy
from flybrain_training.curriculum import (
    evaluation_curriculum,
    robustness_curriculum,
    validation_curriculum,
)
from flybrain_training.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    observation_features,
    record_features,
    target_from_normalized,
)
from flybrain_training.rollout import aggregate_metrics, run_episode
from flybrain_training.schema import (
    ActionV1,
    ObservationV1,
    synchronized_strike_is_ready,
)
from flybrain_training.topology_controls import build_topology_control


EXPECTED_GRAPH_SHA256 = "eff4093bf53c4dd17d7ee4f2f838f6ae5ede70570f9a91317dd8824cca5c771d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
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


def validation_features(path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rows = parquet.read_table(path).to_pylist()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    episodes = []
    for episode_id in sorted(grouped):
        episode = sorted(grouped[episode_id], key=lambda row: int(row["tick"]))
        features = torch.tensor(
            [record_features(row) for row in episode], dtype=torch.float32
        )
        strikes = torch.tensor(
            [float(row["action_strike"]) for row in episode], dtype=torch.float32
        )
        episodes.append((features, strikes))
    return episodes


@torch.no_grad()
def calibrate_threshold(
    policy: MaleCnsPolicy,
    episodes: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, dict[str, float | int]]:
    probability_parts = []
    strike_parts = []
    policy.eval()
    for features, strikes in episodes:
        _, logits, _ = policy(features.unsqueeze(0).to(device))
        probability_parts.append(torch.sigmoid(logits[0]).cpu().numpy())
        strike_parts.append(strikes.numpy())
    probabilities = np.concatenate(probability_parts)
    expected = np.concatenate(strike_parts)
    best = (0.5, -1.0, 0, 0, 0)
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probabilities >= threshold
        positive = expected == 1
        true_positive = int(np.logical_and(predicted, positive).sum())
        false_positive = int(np.logical_and(predicted, ~positive).sum())
        false_negative = int(np.logical_and(~predicted, positive).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best[1]:
            best = (
                float(threshold),
                f1,
                true_positive,
                false_positive,
                false_negative,
            )
    threshold, f1, true_positive, false_positive, false_negative = best
    return threshold, {
        "rows": len(expected),
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


class LiveConnectomePolicy:
    def __init__(
        self,
        policy: MaleCnsPolicy,
        device: torch.device,
        strike_threshold: float,
        carry_state: bool,
    ) -> None:
        self.policy = policy
        self.device = device
        self.strike_threshold = strike_threshold
        self.carry_state = carry_state
        self.state: torch.Tensor | None = None
        self.latencies_seconds: list[float] = []
        self.probabilities: list[float] = []
        self.ready_probabilities: list[float] = []

    @torch.no_grad()
    def decide(self, observation: ObservationV1) -> ActionV1:
        started = time.perf_counter()
        features = torch.tensor(
            observation_features(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.state is None or not self.carry_state:
            self.state = self.policy.initial_state(1, device=self.device)
        target, strike_logit, next_state = self.policy.step(features, self.state)
        self.state = next_state if self.carry_state else None
        target_units = target_from_normalized(target[0].cpu().tolist())
        strike_probability = float(torch.sigmoid(strike_logit[0]))
        self.probabilities.append(strike_probability)
        if synchronized_strike_is_ready(observation):
            self.ready_probabilities.append(strike_probability)
        self.latencies_seconds.append(time.perf_counter() - started)
        return ActionV1(
            target=target_units,
            strike=(
                strike_probability >= self.strike_threshold
                and synchronized_strike_is_ready(observation)
            ),
        )


def construct_policy(
    graph: dict[str, np.ndarray],
    sensory_ids: np.ndarray,
    motor_ids: np.ndarray,
    seed: int,
    device: torch.device,
    trainable_state: dict[str, torch.Tensor] | None = None,
) -> MaleCnsPolicy:
    torch.manual_seed(seed)
    policy = MaleCnsPolicy(
        graph,
        sensory_ids,
        motor_ids,
        input_size=len(FEATURE_NAMES),
        seed=seed,
    )
    if trainable_state is not None:
        parameters = dict(policy.named_parameters())
        if set(parameters) != set(trainable_state):
            raise ValueError("checkpoint trainable parameter layout differs")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(trainable_state[name])
    return policy.to(device).eval()


def live_evaluation(
    policy: MaleCnsPolicy,
    device: torch.device,
    threshold: float,
    suite_name: str,
    maximum_ticks: int,
    carry_state: bool,
) -> dict[str, Any]:
    if suite_name == "robustness":
        curriculum = robustness_curriculum()
    elif suite_name == "validation":
        curriculum = validation_curriculum()
    else:
        curriculum = evaluation_curriculum()
    rollouts = []
    latencies = []
    probabilities = []
    ready_probabilities = []
    for index, episode in enumerate(curriculum):
        live_policy = LiveConnectomePolicy(policy, device, threshold, carry_state)
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
        probabilities.extend(live_policy.probabilities)
        ready_probabilities.extend(live_policy.ready_probabilities)
        print(
            json.dumps(
                {
                    "episode": index + 1,
                    "episodes": len(curriculum),
                    "name": episode.name,
                    "hit": rollout.summary.hit,
                    "ticks": rollout.summary.elapsed_ticks,
                    "accepted_strikes": rollout.summary.accepted_strikes,
                }
            ),
            flush=True,
        )
    metrics = aggregate_metrics(rollouts)
    metrics["inference_calls"] = len(latencies)
    metrics["inference_median_ms"] = float(np.median(latencies) * 1000)
    metrics["inference_p95_ms"] = float(np.percentile(latencies, 95) * 1000)
    metrics["strike_probability_median"] = float(np.median(probabilities))
    metrics["strike_probability_p95"] = float(np.percentile(probabilities, 95))
    metrics["strike_probability_max"] = float(np.max(probabilities))
    metrics["ready_strike_probability_max"] = float(np.max(ready_probabilities))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("data/behavior-cloning-v1/validation.parquet"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/connectome-v1-live.metrics.json")
    )
    parser.add_argument(
        "--suite", choices=("evaluation", "validation", "robustness"), default="validation"
    )
    parser.add_argument("--maximum-ticks", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trained-only", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float)
    parser.add_argument("--initial-reference", type=Path)
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-graph live evaluation requires CUDA")

    graph_directory = resolve_graph_path(args.graph)
    graph_path = graph_directory / "graph.npz"
    graph_sha256 = sha256_file(graph_path)
    if graph_sha256 != EXPECTED_GRAPH_SHA256:
        raise ValueError(f"unexpected MaleCNS graph: {graph_sha256}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["graph_sha256"] != graph_sha256:
        raise ValueError("checkpoint uses a different connectome graph")
    if checkpoint["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint uses a different feature schema")
    seed = int(checkpoint["seed"])
    topology_variant = str(checkpoint.get("topology_variant", "measured"))
    topology_seed = int(checkpoint.get("topology_seed", seed))
    nodes = feather.read_table(graph_directory / "nodes.feather")
    superclasses = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(superclasses == "vnc_sensory")
    motor_ids = np.flatnonzero(superclasses == "vnc_motor")
    with np.load(graph_path) as source_graph:
        body_ids = np.asarray(source_graph["body_ids"])
        node_body_ids = np.asarray(nodes["bodyId"])
        if not np.array_equal(body_ids, node_body_ids):
            raise ValueError("nodes.feather order does not match graph body_ids")
        graph, topology_control = build_topology_control(
            source_graph, topology_variant, topology_seed
        )
    checkpoint_topology_digest = checkpoint.get("topology_digest")
    if (
        checkpoint_topology_digest is not None
        and checkpoint_topology_digest != topology_control["digest"]
    ):
        raise ValueError("checkpoint topology digest does not match reconstructed graph")
    del nodes, superclasses, body_ids, node_body_ids
    validation = validation_features(args.validation_data)

    initial_threshold = None
    initial_calibration = None
    initial_live = None
    initial_reference_sha256 = None
    if args.initial_reference is not None:
        reference = json.loads(args.initial_reference.read_text())
        if reference["graph_sha256"] != graph_sha256:
            raise ValueError("initial reference uses a different connectome graph")
        if reference["seed"] != seed:
            raise ValueError("initial reference uses a different interface seed")
        if reference.get("topology_variant", "measured") != topology_variant:
            raise ValueError("initial reference uses a different topology variant")
        if int(reference.get("topology_seed", seed)) != topology_seed:
            raise ValueError("initial reference uses a different topology seed")
        reference_topology_digest = reference.get("topology_digest")
        if (
            reference_topology_digest is not None
            and reference_topology_digest != topology_control["digest"]
        ):
            raise ValueError("initial reference uses a different topology")
        initial_threshold = reference["initial_threshold"]
        initial_calibration = reference["initial_threshold_calibration"]
        initial_live = reference["initial"]
        initial_reference_sha256 = sha256_file(args.initial_reference)
    elif not args.trained_only:
        initial = construct_policy(
            graph, sensory_ids, motor_ids, seed, device
        )
        initial_threshold, initial_calibration = calibrate_threshold(
            initial, validation, device
        )
        print(
            json.dumps(
                {
                    "condition": "initial",
                    "threshold": initial_threshold,
                    **initial_calibration,
                }
            ),
            flush=True,
        )
        initial_live = live_evaluation(
            initial,
            device,
            initial_threshold,
            args.suite,
            args.maximum_ticks,
            not args.reset_state,
        )
        del initial
        torch.cuda.empty_cache()

    trained = construct_policy(
        graph,
        sensory_ids,
        motor_ids,
        seed,
        device,
        checkpoint["trainable_state_dict"],
    )
    thresholds = args.thresholds or [float(checkpoint["strike_threshold"])]
    trained_conditions = {}
    for threshold in thresholds:
        print(
            json.dumps({"condition": "trained", "threshold": threshold}), flush=True
        )
        trained_conditions[str(threshold)] = live_evaluation(
            trained,
            device,
            threshold,
            args.suite,
            args.maximum_ticks,
            not args.reset_state,
        )
    trained_live = trained_conditions[str(thresholds[0])]
    beats_initialization = None
    if initial_live is not None and len(thresholds) == 1:
        initial_hits = int(initial_live["hits"])
        trained_hits = int(trained_live["hits"])
        beats_initialization = trained_hits > initial_hits or (
            trained_hits == initial_hits
            and trained_hits > 0
            and float(trained_live["mean_hit_ticks"])
            < float(initial_live["mean_hit_ticks"])
        )
    status = (
        "passed"
        if beats_initialization is True
        else "did_not_beat_initialization"
        if beats_initialization is False
        else "diagnostic"
    )
    metrics = {
        "experiment": "Phase 4 trained versus initialized live controller",
        "status": status,
        "claim": (
            "One closed-loop topology condition and seed. Biological-topology "
            "claims require matched control conditions across additional seeds."
        ),
        "suite": args.suite,
        "maximum_ticks": args.maximum_ticks,
        "neural_state_carried": not args.reset_state,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "graph_sha256": graph_sha256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "seed": seed,
        "topology_variant": topology_variant,
        "topology_seed": topology_seed,
        "topology_digest": topology_control["digest"],
        "topology_control": topology_control,
        "initial_threshold": initial_threshold,
        "initial_threshold_calibration": initial_calibration,
        "initial_reference": (
            str(args.initial_reference) if args.initial_reference is not None else None
        ),
        "initial_reference_sha256": initial_reference_sha256,
        "trained_thresholds": thresholds,
        "initial": initial_live,
        "trained": trained_live if len(thresholds) == 1 else None,
        "trained_conditions": trained_conditions,
        "beats_initialization": beats_initialization,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
