from pathlib import Path
import tempfile
import unittest

import flybrain_game
import pyarrow.parquet as parquet

from flybrain_training.curriculum import (
    EvasiveTrajectory,
    RandomWaypointTrajectory,
    StationaryTrajectory,
)
from flybrain_training.expert import InterceptExpertV1
from flybrain_training.features import FEATURE_NAMES, observation_features, record_features
from flybrain_training.rollout import run_episode, write_parquet
from flybrain_training.schema import ObservationV1


class Phase2PipelineTest(unittest.TestCase):
    def test_native_observation_exposes_versioned_exact_motion(self) -> None:
        simulation = flybrain_game.Simulation()
        simulation.step_toward(
            flybrain_game.FLIGHT_MAX_X_UNITS,
            flybrain_game.FLIGHT_MIN_Y_UNITS,
        )

        observation = ObservationV1.from_simulation(simulation)
        self.assertEqual(
            observation.schema_version, flybrain_game.OBSERVATION_SCHEMA_VERSION
        )
        self.assertEqual(observation.tick, 1)
        self.assertNotEqual(observation.player_velocity, (0, 0))
        self.assertEqual(observation.player_acceleration, observation.player_velocity)
        self.assertEqual(len(observation.hands), 2)

    def test_seeded_random_waypoints_are_reproducible(self) -> None:
        first = RandomWaypointTrajectory(1701)
        second = RandomWaypointTrajectory(1701)
        different = RandomWaypointTrajectory(99017)
        simulation = flybrain_game.Simulation()
        observation = ObservationV1.from_simulation(simulation)

        self.assertEqual(first.destination(observation), second.destination(observation))
        self.assertNotEqual(first.destination(observation), different.destination(observation))

    def test_expert_hits_simple_motion_and_records_an_evasive_miss(self) -> None:
        expert = InterceptExpertV1()
        stationary = run_episode(0, "stationary", StationaryTrajectory(), expert)
        evasive = run_episode(1, "evasive", EvasiveTrajectory(), expert)

        self.assertTrue(stationary.summary.hit)
        self.assertLess(stationary.summary.elapsed_ticks, 100)
        self.assertTrue(evasive.summary.hit)
        self.assertGreaterEqual(evasive.summary.completed_misses, 1)
        self.assertEqual(evasive.summary.cooldown_violations, 0)

    def test_parquet_dataset_carries_schema_metadata(self) -> None:
        rollout = run_episode(
            0,
            "stationary",
            StationaryTrajectory(),
            InterceptExpertV1(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.parquet"
            write_parquet([rollout], path)
            table = parquet.read_table(path)

        self.assertEqual(table.num_rows, rollout.summary.elapsed_ticks)
        self.assertIn("player_acceleration_x_units", table.column_names)
        self.assertIn("completed_miss", table.column_names)
        metadata = table.schema.metadata or {}
        self.assertEqual(metadata[b"flybrain.dataset_schema_version"], b"1")
        self.assertEqual(metadata[b"flybrain.observation_schema_version"], b"1")
        self.assertEqual(metadata[b"flybrain.action_schema_version"], b"1")

    def test_live_and_dataset_feature_mapping_agree(self) -> None:
        simulation = flybrain_game.Simulation()
        observation = ObservationV1.from_simulation(simulation)
        rollout = run_episode(
            0,
            "stationary",
            StationaryTrajectory(),
            InterceptExpertV1(),
        )

        live = observation_features(observation)
        stored = record_features(rollout.rows[0])
        self.assertEqual(len(live), len(FEATURE_NAMES))
        self.assertEqual(live, stored)


if __name__ == "__main__":
    unittest.main()
