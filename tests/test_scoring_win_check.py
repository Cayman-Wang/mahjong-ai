import unittest

from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.scoring.win_check import detect_win


class TestWinCheck(unittest.TestCase):
    def test_standard_win(self):
        c = counts_empty()
        # 111W 234W 567W 234T 99T
        for _ in range(3):
            c[tile_id(0, 1)] += 1
        for r in (2, 3, 4):
            c[tile_id(0, r)] += 1
        for r in (5, 6, 7):
            c[tile_id(0, r)] += 1
        for r in (2, 3, 4):
            c[tile_id(1, r)] += 1
        for _ in range(2):
            c[tile_id(1, 9)] += 1

        res = detect_win(c, meld_count=0, dingque_suit=None)
        self.assertTrue(res.ok)
        self.assertEqual(res.kind, "standard")

    def test_qidui(self):
        c = counts_empty()
        # 7 pairs in W suit.
        for r in (1, 2, 3, 4, 5, 6, 7):
            c[tile_id(0, r)] += 2
        res = detect_win(c, meld_count=0, dingque_suit=None)
        self.assertTrue(res.ok)
        self.assertEqual(res.kind, "qidui")


if __name__ == "__main__":
    unittest.main()
