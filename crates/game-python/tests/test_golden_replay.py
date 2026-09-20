import csv
from pathlib import Path
import unittest

from flybrain_game import Simulation


REPLAY = Path(__file__).parents[3] / "tests" / "replays" / "player-movement-v1.csv"
ACTION_REPLAY = Path(__file__).parents[3] / "tests" / "replays" / "action-round-v1.csv"


class GoldenReplayTest(unittest.TestCase):
    def test_movement_replay_matches_golden_checkpoints(self) -> None:
        simulation = Simulation()

        with REPLAY.open(newline="", encoding="utf-8") as replay_file:
            rows = csv.reader(
                line for line in replay_file if line.strip() and not line.startswith("#")
            )
            for row_number, row in enumerate(rows, start=1):
                (
                    horizontal,
                    vertical,
                    ticks,
                    expected_tick,
                    position_x,
                    position_y,
                    velocity_x,
                    velocity_y,
                ) = (int(value) for value in row)

                for _ in range(ticks):
                    simulation.step(horizontal, vertical)

                self.assertEqual(simulation.tick, expected_tick, f"tick at row {row_number}")
                self.assertEqual(
                    (simulation.player_x_units, simulation.player_y_units),
                    (position_x, position_y),
                    f"position at row {row_number}",
                )
                self.assertEqual(
                    (
                        simulation.player_velocity_x_units,
                        simulation.player_velocity_y_units,
                    ),
                    (velocity_x, velocity_y),
                    f"velocity at row {row_number}",
                )

    def test_action_replay_matches_every_gameplay_tick(self) -> None:
        simulation = Simulation()
        current_episode = None
        replay_ticks = 0

        with ACTION_REPLAY.open(newline="", encoding="utf-8") as replay_file:
            rows = csv.reader(
                line for line in replay_file if line.strip() and not line.startswith("#")
            )
            for row_number, row in enumerate(rows, start=1):
                values = [int(value) for value in row]
                self.assertEqual(len(values), 26, f"column count at row {row_number}")
                episode = values[0]
                if current_episode != episode:
                    if current_episode is not None:
                        self.assertEqual(episode, current_episode + 1, "episode sequence")
                        self.assertEqual(simulation.round_status, 2, "survival episode status")
                        simulation.restart()
                    current_episode = episode

                simulation.step_with_action(
                    values[2], values[3], values[4], values[5], values[6] == 1
                )
                replay_ticks += 1
                self.assertEqual(simulation.tick, values[1], f"tick at row {row_number}")
                self.assertEqual(
                    (simulation.player_x_units, simulation.player_y_units),
                    (values[7], values[8]),
                    f"player position at row {row_number}",
                )
                self.assertEqual(
                    (
                        simulation.player_velocity_x_units,
                        simulation.player_velocity_y_units,
                    ),
                    (values[9], values[10]),
                    f"player velocity at row {row_number}",
                )
                self.assertEqual(
                    simulation.round_status, values[11], f"round status at row {row_number}"
                )
                self.assertEqual(
                    simulation.round_elapsed_ticks,
                    values[12],
                    f"round elapsed ticks at row {row_number}",
                )
                self.assertEqual(
                    simulation.attack_active,
                    values[13] == 1,
                    f"attack active at row {row_number}",
                )

                for hand_index, offset in ((0, 14), (1, 20)):
                    self.assertEqual(
                        simulation.hand_state(hand_index),
                        (
                            values[offset],
                            values[offset + 1],
                            values[offset + 2],
                            values[offset + 3],
                            values[offset + 4],
                        ),
                        f"hand {hand_index} state at row {row_number}",
                    )
                    self.assertEqual(
                        simulation.hand_has_active_contact(hand_index),
                        values[offset + 5] == 1,
                        f"hand {hand_index} contact at row {row_number}",
                    )

        self.assertEqual(replay_ticks, 1868)
        self.assertEqual(current_episode, 1)
        self.assertEqual(simulation.round_status, 1, "hit episode status")


if __name__ == "__main__":
    unittest.main()
