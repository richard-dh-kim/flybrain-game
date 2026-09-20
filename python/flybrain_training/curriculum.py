"""Deterministic destination generators used for expert evaluation and labeling."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Protocol

import flybrain_game

from .schema import ObservationV1, Vec2


class Trajectory(Protocol):
    def destination(self, observation: ObservationV1) -> Vec2: ...


@dataclass(frozen=True, slots=True)
class CurriculumEpisode:
    name: str
    trajectory: Trajectory
    initial_player_position: Vec2 | None = None


def _interior_position(x_numerator: int, y_numerator: int, denominator: int = 10) -> Vec2:
    x_span = flybrain_game.FLIGHT_MAX_X_UNITS - flybrain_game.FLIGHT_MIN_X_UNITS
    y_span = flybrain_game.FLIGHT_MAX_Y_UNITS - flybrain_game.FLIGHT_MIN_Y_UNITS
    return (
        flybrain_game.FLIGHT_MIN_X_UNITS + x_span * x_numerator // denominator,
        flybrain_game.FLIGHT_MIN_Y_UNITS + y_span * y_numerator // denominator,
    )


TRAIN_STARTS = tuple(
    _interior_position(x, y)
    for x, y in ((2, 2), (5, 2), (8, 2), (2, 5), (5, 5), (8, 5), (2, 8), (5, 8), (8, 8))
)
VALIDATION_STARTS = tuple(
    _interior_position(x, y)
    for x, y in ((1, 3), (4, 3), (7, 3), (9, 3), (1, 7), (4, 7), (7, 7), (9, 7))
)


def _episodes_with_starts(
    items: list[tuple[str, Trajectory]], starts: tuple[Vec2, ...]
) -> list[CurriculumEpisode]:
    return [
        CurriculumEpisode(name, trajectory, starts[index % len(starts)])
        for index, (name, trajectory) in enumerate(items)
    ]


def _axis_destination(direction: Vec2) -> Vec2:
    center_x = (flybrain_game.FLIGHT_MIN_X_UNITS + flybrain_game.FLIGHT_MAX_X_UNITS) // 2
    center_y = (flybrain_game.FLIGHT_MIN_Y_UNITS + flybrain_game.FLIGHT_MAX_Y_UNITS) // 2
    return (
        flybrain_game.FLIGHT_MAX_X_UNITS
        if direction[0] > 0
        else flybrain_game.FLIGHT_MIN_X_UNITS
        if direction[0] < 0
        else center_x,
        flybrain_game.FLIGHT_MAX_Y_UNITS
        if direction[1] > 0
        else flybrain_game.FLIGHT_MIN_Y_UNITS
        if direction[1] < 0
        else center_y,
    )


@dataclass(frozen=True, slots=True)
class StationaryTrajectory:
    point: Vec2 = (
        (flybrain_game.FLIGHT_MIN_X_UNITS + flybrain_game.FLIGHT_MAX_X_UNITS) // 2,
        (flybrain_game.FLIGHT_MIN_Y_UNITS + flybrain_game.FLIGHT_MAX_Y_UNITS) // 2,
    )

    def destination(self, observation: ObservationV1) -> Vec2:
        del observation
        return self.point


@dataclass(frozen=True, slots=True)
class ConstantVelocityTrajectory:
    direction: Vec2

    def destination(self, observation: ObservationV1) -> Vec2:
        del observation
        return _axis_destination(self.direction)


@dataclass(frozen=True, slots=True)
class TurningTrajectory:
    directions: tuple[Vec2, ...]
    turn_period_ticks: int
    phase_offset_ticks: int = 0

    def __post_init__(self) -> None:
        if not self.directions:
            raise ValueError("turning trajectory needs at least one direction")
        if self.turn_period_ticks <= 0:
            raise ValueError("turn period must be positive")

    def destination(self, observation: ObservationV1) -> Vec2:
        index = (
            (observation.tick + self.phase_offset_ticks) // self.turn_period_ticks
        ) % len(self.directions)
        return _axis_destination(self.directions[index])


@dataclass(slots=True)
class RandomWaypointTrajectory:
    seed: int
    hold_ticks: int = 90
    _waypoints: list[Vec2] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hold_ticks <= 0:
            raise ValueError("waypoint hold must be positive")
        generator = random.Random(self.seed)
        margin = 36 * flybrain_game.UNITS_PER_PIXEL
        self._waypoints = [
            (
                generator.randint(
                    flybrain_game.FLIGHT_MIN_X_UNITS + margin,
                    flybrain_game.FLIGHT_MAX_X_UNITS - margin,
                ),
                generator.randint(
                    flybrain_game.FLIGHT_MIN_Y_UNITS + margin,
                    flybrain_game.FLIGHT_MAX_Y_UNITS - margin,
                ),
            )
            for _ in range(32)
        ]

    def destination(self, observation: ObservationV1) -> Vec2:
        return self._waypoints[(observation.tick // self.hold_ticks) % len(self._waypoints)]


@dataclass(frozen=True, slots=True)
class EvasiveTrajectory:
    """Bait a target, then reverse direction while the slap is committed."""

    def destination(self, observation: ObservationV1) -> Vec2:
        if observation.attack_active:
            velocity = observation.player_velocity
            reverse = (
                -1 if velocity[0] > 0 else 1 if velocity[0] < 0 else 1,
                -1 if velocity[1] > 0 else 1 if velocity[1] < 0 else -1,
            )
            return _axis_destination(reverse)

        hand_midpoint = (
            (observation.hands[0].position[0] + observation.hands[1].position[0]) // 2,
            (observation.hands[0].position[1] + observation.hands[1].position[1]) // 2,
        )
        player = observation.player_position
        away = (player[0] - hand_midpoint[0], player[1] - hand_midpoint[1])
        if away == (0, 0):
            away = (1, -1)
        return _axis_destination((1 if away[0] >= 0 else -1, 1 if away[1] >= 0 else -1))


@dataclass(frozen=True, slots=True)
class RecordedHumanTrajectory:
    destinations: tuple[Vec2, ...]

    @classmethod
    def from_csv(cls, path: Path) -> RecordedHumanTrajectory:
        destinations: list[Vec2] = []
        with path.open(newline="", encoding="utf-8") as source:
            rows = csv.DictReader(source)
            if rows.fieldnames is None or not {
                "destination_x_units",
                "destination_y_units",
            }.issubset(rows.fieldnames):
                raise ValueError("human trajectory CSV is missing destination unit columns")
            for row in rows:
                destinations.append(
                    (int(row["destination_x_units"]), int(row["destination_y_units"]))
                )
        if not destinations:
            raise ValueError("human trajectory CSV has no rows")
        return cls(tuple(destinations))

    def destination(self, observation: ObservationV1) -> Vec2:
        return self.destinations[min(observation.tick, len(self.destinations) - 1)]


def evaluation_curriculum() -> list[CurriculumEpisode]:
    cardinal = ((1, 0), (0, 1), (-1, 0), (0, -1))
    diagonal = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    return [
        CurriculumEpisode("stationary-center", StationaryTrajectory()),
        CurriculumEpisode("constant-east", ConstantVelocityTrajectory((1, 0))),
        CurriculumEpisode("constant-west", ConstantVelocityTrajectory((-1, 0))),
        CurriculumEpisode("constant-north", ConstantVelocityTrajectory((0, -1))),
        CurriculumEpisode("constant-south", ConstantVelocityTrajectory((0, 1))),
        CurriculumEpisode("turn-cardinal-45", TurningTrajectory(cardinal, 45, 20)),
        CurriculumEpisode("turn-cardinal-60", TurningTrajectory(cardinal, 60, 40)),
        CurriculumEpisode("turn-diagonal-45", TurningTrajectory(diagonal, 45, 10)),
        CurriculumEpisode("turn-diagonal-60", TurningTrajectory(diagonal, 60, 30)),
        CurriculumEpisode(
            "random-waypoint-1701", RandomWaypointTrajectory(1701, hold_ticks=36)
        ),
        CurriculumEpisode(
            "random-waypoint-99017", RandomWaypointTrajectory(99017, hold_ticks=48)
        ),
        CurriculumEpisode("evasive", EvasiveTrajectory()),
    ]


def training_curriculum() -> list[CurriculumEpisode]:
    cardinal = ((1, 0), (0, 1), (-1, 0), (0, -1))
    diagonal = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    curriculum: list[tuple[str, Trajectory]] = [
        ("train-stationary", StationaryTrajectory()),
        *[
            (f"train-constant-{x}-{y}", ConstantVelocityTrajectory((x, y)))
            for x, y in (*cardinal, *diagonal)
        ],
    ]
    for period in (24, 30, 36, 48, 72, 96):
        for offset_index, offset in enumerate((0, period // 3, 2 * period // 3)):
            curriculum.append(
                (
                    f"train-turn-cardinal-{period}-{offset_index}",
                    TurningTrajectory(cardinal, period, offset),
                )
            )
            curriculum.append(
                (
                    f"train-turn-diagonal-{period}-{offset_index}",
                    TurningTrajectory(diagonal, period, offset),
                )
            )
    for index, seed in enumerate(range(2_000, 2_032)):
        hold_ticks = (24, 36, 48, 60)[index % 4]
        curriculum.append(
            (
                f"train-random-{seed}-{hold_ticks}",
                RandomWaypointTrajectory(seed, hold_ticks=hold_ticks),
            )
        )
    curriculum.append(("train-evasive", EvasiveTrajectory()))
    return _episodes_with_starts(curriculum, TRAIN_STARTS)


def validation_curriculum() -> list[CurriculumEpisode]:
    cardinal = ((1, 0), (0, 1), (-1, 0), (0, -1))
    diagonal = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    curriculum: list[tuple[str, Trajectory]] = []
    for period in (27, 42, 54, 84):
        for offset_index, offset in enumerate((period // 4, 3 * period // 4)):
            curriculum.append(
                (
                    f"validation-turn-cardinal-{period}-{offset_index}",
                    TurningTrajectory(cardinal, period, offset),
                )
            )
            curriculum.append(
                (
                    f"validation-turn-diagonal-{period}-{offset_index}",
                    TurningTrajectory(diagonal, period, offset),
                )
            )
    for index, seed in enumerate(range(9_000, 9_016)):
        hold_ticks = (30, 42, 54, 66)[index % 4]
        curriculum.append(
            (
                f"validation-random-{seed}-{hold_ticks}",
                RandomWaypointTrajectory(seed, hold_ticks=hold_ticks),
            )
        )
    curriculum.append(("validation-evasive", EvasiveTrajectory()))
    return _episodes_with_starts(curriculum, VALIDATION_STARTS)


def robustness_curriculum() -> list[CurriculumEpisode]:
    base = evaluation_curriculum()
    episodes: list[CurriculumEpisode] = []
    for start_index, start in enumerate((*TRAIN_STARTS, *VALIDATION_STARTS)):
        for base_episode in base:
            episodes.append(
                CurriculumEpisode(
                    f"robust-{start_index}-{base_episode.name}",
                    base_episode.trajectory,
                    start,
                )
            )
    return episodes


def dataset_curriculum(split: str) -> list[CurriculumEpisode]:
    if split == "evaluation":
        return evaluation_curriculum()
    if split == "train":
        return training_curriculum()
    if split == "validation":
        return validation_curriculum()
    if split == "robustness":
        return robustness_curriculum()
    raise ValueError(f"unknown curriculum split: {split}")
