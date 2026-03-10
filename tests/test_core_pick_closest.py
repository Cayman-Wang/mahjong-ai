import unittest

from mahjong_ai.core.engine import GameEngine


class TestPickClosest(unittest.TestCase):
    def test_pick_closest_to_player(self):
        engine = GameEngine(enable_events=False)

        cases = [
            (0, [1, 2, 3], 1),
            (0, [2, 3], 2),
            (1, [2, 3, 0], 2),
            (3, [0, 1, 2], 0),
            (1, [3], 3),
            (2, [], None),
        ]
        for center, cands, expected in cases:
            self.assertEqual(engine._pick_closest_to_player(center, cands), expected)


if __name__ == "__main__":
    unittest.main()

