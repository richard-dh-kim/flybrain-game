"""Versioned Python views of the authoritative Rust simulation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import flybrain_game


Vec2: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class HandObservationV1:
    position: Vec2
    velocity: Vec2
    phase: int
    phase_tick: int
    cooldown_remaining: int
    active_contact: bool

    @classmethod
    def from_native(cls, value: tuple[object, ...]) -> HandObservationV1:
        if len(value) != 8:
            raise ValueError(f"hand observation v1 needs 8 values, got {len(value)}")
        return cls(
            position=(int(value[0]), int(value[1])),
            velocity=(int(value[2]), int(value[3])),
            phase=int(value[4]),
            phase_tick=int(value[5]),
            cooldown_remaining=int(value[6]),
            active_contact=bool(value[7]),
        )


@dataclass(frozen=True, slots=True)
class ObservationV1:
    schema_version: int
    tick: int
    player_position: Vec2
    player_velocity: Vec2
    player_acceleration: Vec2
    hands: tuple[HandObservationV1, HandObservationV1]
    attack_active: bool
    round_status: int
    round_elapsed_ticks: int
    round_remaining_ticks: int

    @classmethod
    def from_simulation(cls, simulation: flybrain_game.Simulation) -> ObservationV1:
        value = simulation.observation_v1()
        if len(value) != 11:
            raise ValueError(f"observation v1 needs 11 values, got {len(value)}")
        version = int(value[0])
        if version != flybrain_game.OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation schema {version}; "
                f"expected {flybrain_game.OBSERVATION_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=version,
            tick=int(value[1]),
            player_position=(int(value[2][0]), int(value[2][1])),
            player_velocity=(int(value[3][0]), int(value[3][1])),
            player_acceleration=(int(value[4][0]), int(value[4][1])),
            hands=(
                HandObservationV1.from_native(value[5]),
                HandObservationV1.from_native(value[6]),
            ),
            attack_active=bool(value[7]),
            round_status=int(value[8]),
            round_elapsed_ticks=int(value[9]),
            round_remaining_ticks=int(value[10]),
        )


@dataclass(frozen=True, slots=True)
class ActionV1:
    target: Vec2
    strike: bool
    schema_version: int = flybrain_game.ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != flybrain_game.ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported action schema {self.schema_version}; "
                f"expected {flybrain_game.ACTION_SCHEMA_VERSION}"
            )


def hand_is_ready(hand: HandObservationV1) -> bool:
    """Return whether one hand can accept a synchronized strike command."""

    return hand.phase == 0 and hand.cooldown_remaining == 0


def synchronized_strike_is_ready(observation: ObservationV1) -> bool:
    return not observation.attack_active and all(hand_is_ready(hand) for hand in observation.hands)
