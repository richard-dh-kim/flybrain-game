"""Stable normalized feature mapping for conventional learned baselines."""

from __future__ import annotations

from typing import Any, Sequence

import flybrain_game

from .schema import ObservationV1


FEATURE_SCHEMA_VERSION = 1
ROUND_TICKS = 1_800
PLAYER_SPEED_UNITS = 5 * flybrain_game.UNITS_PER_PIXEL
HAND_SPEED_SCALE_UNITS = 32 * flybrain_game.UNITS_PER_PIXEL
ACCELERATION_SCALE_UNITS = 2 * PLAYER_SPEED_UNITS

FEATURE_NAMES = (
    "tick_fraction",
    "player_x",
    "player_y",
    "player_velocity_x",
    "player_velocity_y",
    "player_acceleration_x",
    "player_acceleration_y",
    *(
        f"{side}_{field}"
        for side in ("left", "right")
        for field in (
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "phase_track",
            "phase_wind_up",
            "phase_strike",
            "phase_impact_hold",
            "phase_recover",
            "phase_tick",
            "cooldown",
            "active_contact",
        )
    ),
    "attack_active",
    "round_remaining_fraction",
)


def _normalize_x(value: int) -> float:
    span = flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    return (value - flybrain_game.FLIGHT_MIN_X_UNITS) / span


def _normalize_y(value: int) -> float:
    span = flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    return (value - flybrain_game.FLIGHT_MIN_Y_UNITS) / span


def _hand_features(
    position: tuple[int, int],
    velocity: tuple[int, int],
    phase: int,
    phase_tick: int,
    cooldown: int,
    active_contact: bool,
) -> list[float]:
    return [
        _normalize_x(position[0]),
        _normalize_y(position[1]),
        velocity[0] / HAND_SPEED_SCALE_UNITS,
        velocity[1] / HAND_SPEED_SCALE_UNITS,
        *(1.0 if phase == index else 0.0 for index in range(5)),
        phase_tick / max(
            flybrain_game.HAND_WIND_UP_TICKS,
            flybrain_game.HAND_STRIKE_TICKS,
            flybrain_game.HAND_IMPACT_HOLD_TICKS,
            flybrain_game.HAND_RECOVER_TICKS,
        ),
        cooldown / flybrain_game.HAND_COOLDOWN_TICKS,
        float(active_contact),
    ]


def observation_features(observation: ObservationV1) -> list[float]:
    features = [
        observation.tick / ROUND_TICKS,
        _normalize_x(observation.player_position[0]),
        _normalize_y(observation.player_position[1]),
        observation.player_velocity[0] / PLAYER_SPEED_UNITS,
        observation.player_velocity[1] / PLAYER_SPEED_UNITS,
        observation.player_acceleration[0] / ACCELERATION_SCALE_UNITS,
        observation.player_acceleration[1] / ACCELERATION_SCALE_UNITS,
    ]
    for hand in observation.hands:
        features.extend(
            _hand_features(
                hand.position,
                hand.velocity,
                hand.phase,
                hand.phase_tick,
                hand.cooldown_remaining,
                hand.active_contact,
            )
        )
    features.extend(
        [
            float(observation.attack_active),
            observation.round_remaining_ticks / ROUND_TICKS,
        ]
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"feature layout mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    return features


def record_features(record: dict[str, Any]) -> list[float]:
    features = [
        int(record["tick"]) / ROUND_TICKS,
        _normalize_x(int(record["player_x_units"])),
        _normalize_y(int(record["player_y_units"])),
        int(record["player_velocity_x_units"]) / PLAYER_SPEED_UNITS,
        int(record["player_velocity_y_units"]) / PLAYER_SPEED_UNITS,
        int(record["player_acceleration_x_units"]) / ACCELERATION_SCALE_UNITS,
        int(record["player_acceleration_y_units"]) / ACCELERATION_SCALE_UNITS,
    ]
    for side in ("left", "right"):
        features.extend(
            _hand_features(
                (
                    int(record[f"{side}_hand_x_units"]),
                    int(record[f"{side}_hand_y_units"]),
                ),
                (
                    int(record[f"{side}_hand_velocity_x_units"]),
                    int(record[f"{side}_hand_velocity_y_units"]),
                ),
                int(record[f"{side}_hand_phase"]),
                int(record[f"{side}_hand_phase_tick"]),
                int(record[f"{side}_hand_cooldown_remaining"]),
                bool(record[f"{side}_hand_active_contact"]),
            )
        )
    features.extend(
        [
            float(record["attack_active"]),
            int(record["round_remaining_ticks"]) / ROUND_TICKS,
        ]
    )
    return features


def normalized_target(target: Sequence[int]) -> tuple[float, float]:
    return _normalize_x(int(target[0])), _normalize_y(int(target[1]))


def target_from_normalized(target: Sequence[float]) -> tuple[int, int]:
    x_span = flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    y_span = flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    return (
        round(flybrain_game.FLIGHT_MIN_X_UNITS + min(1.0, max(0.0, target[0])) * x_span),
        round(flybrain_game.FLIGHT_MIN_Y_UNITS + min(1.0, max(0.0, target[1])) * y_span),
    )
