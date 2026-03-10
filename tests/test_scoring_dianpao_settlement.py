import unittest

from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.settlement import settle_hu


class TestDianpaoSettlement(unittest.TestCase):
    def test_dianpao_winner_only(self):
        rules = RulesConfig(dianpao_pays_all_alive=False, base_points=1)
        d = settle_hu(
            winner=1,
            from_player=0,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=2,
        )
        self.assertEqual(d, [-1, 1, 0, 0])
        self.assertEqual(sum(d), 0)

    def test_dianpao_pays_all_alive(self):
        rules = RulesConfig(dianpao_pays_all_alive=True, base_points=1, enable_dealer_multiplier=False)
        d = settle_hu(
            winner=1,
            from_player=0,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [-3, 3, 0, 0])
        self.assertEqual(sum(d), 0)

    def test_yipao_duoxiang_discards_pays_each_winner(self):
        rules = RulesConfig(dianpao_pays_all_alive=False, base_points=1)
        d1 = settle_hu(
            winner=1,
            from_player=0,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=3,
        )
        d2 = settle_hu(
            winner=2,
            from_player=0,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=3,
        )
        d = [d1[i] + d2[i] for i in range(4)]
        self.assertEqual(d, [-2, 1, 1, 0])
        self.assertEqual(sum(d), 0)


if __name__ == "__main__":
    unittest.main()
