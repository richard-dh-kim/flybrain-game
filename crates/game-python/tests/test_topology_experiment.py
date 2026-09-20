import unittest

from flybrain_training.topology_experiment import select_live_threshold


class TopologyExperimentTest(unittest.TestCase):
    def test_selection_prioritizes_hits_before_speed_and_misses(self) -> None:
        conditions = {
            "0.2": {
                "hits": 9,
                "mean_hit_ticks": 40.0,
                "completed_misses": 3,
                "accepted_strikes": 12,
            },
            "0.5": {
                "hits": 10,
                "mean_hit_ticks": 200.0,
                "completed_misses": 20,
                "accepted_strikes": 30,
            },
        }

        self.assertEqual(select_live_threshold(conditions), 0.5)

    def test_selection_breaks_ties_in_predeclared_order(self) -> None:
        conditions = {
            "0.05": {
                "hits": 9,
                "mean_hit_ticks": 80.0,
                "completed_misses": 2,
                "accepted_strikes": 11,
            },
            "0.2": {
                "hits": 9,
                "mean_hit_ticks": 70.0,
                "completed_misses": 8,
                "accepted_strikes": 20,
            },
            "0.35": {
                "hits": 9,
                "mean_hit_ticks": 70.0,
                "completed_misses": 7,
                "accepted_strikes": 20,
            },
            "0.5": {
                "hits": 9,
                "mean_hit_ticks": 70.0,
                "completed_misses": 7,
                "accepted_strikes": 19,
            },
            "0.65": {
                "hits": 9,
                "mean_hit_ticks": 70.0,
                "completed_misses": 7,
                "accepted_strikes": 19,
            },
        }

        self.assertEqual(select_live_threshold(conditions), 0.5)

    def test_selection_handles_no_hits(self) -> None:
        conditions = {
            "0.2": {
                "hits": 0,
                "mean_hit_ticks": None,
                "completed_misses": 4,
                "accepted_strikes": 5,
            },
            "0.5": {
                "hits": 0,
                "mean_hit_ticks": None,
                "completed_misses": 2,
                "accepted_strikes": 3,
            },
        }

        self.assertEqual(select_live_threshold(conditions), 0.5)

    def test_selection_rejects_empty_sweep(self) -> None:
        with self.assertRaises(ValueError):
            select_live_threshold({})


if __name__ == "__main__":
    unittest.main()
