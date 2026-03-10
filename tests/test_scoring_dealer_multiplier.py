import unittest

from mahjong_ai.core.state import MeldKind
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.settlement import settle_gang, settle_hu


class TestDealerMultiplier(unittest.TestCase):
    def test_dealer_self_draw_doubles_all_payers(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2)
        d = settle_hu(
            winner=0,
            from_player=None,
            self_draw=True,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [6, -2, -2, -2])
        self.assertEqual(sum(d), 0)

    def test_non_dealer_self_draw_only_dealer_pays_double(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2)
        d = settle_hu(
            winner=1,
            from_player=None,
            self_draw=True,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [-2, 4, -1, -1])
        self.assertEqual(sum(d), 0)

    def test_dianpao_dealer_discards_pays_double(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2)
        d = settle_hu(
            winner=1,
            from_player=0,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [-2, 2, 0, 0])
        self.assertEqual(sum(d), 0)

    def test_dianpao_to_dealer_wins_double(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2)
        d = settle_hu(
            winner=0,
            from_player=1,
            self_draw=False,
            fan_total=0,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [2, -2, 0, 0])
        self.assertEqual(sum(d), 0)

    def test_gang_dealer_actor_gets_double_from_each_payer(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2, gang_an_pay=2)
        d = settle_gang(
            actor=0,
            gang_kind=MeldKind.GANG_AN,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [12, -4, -4, -4])
        self.assertEqual(sum(d), 0)

    def test_gang_only_dealer_payer_pays_double(self):
        rules = RulesConfig(base_points=1, enable_dealer_multiplier=True, dealer_multiplier=2, gang_ming_pay=1)
        d = settle_gang(
            actor=1,
            gang_kind=MeldKind.GANG_MING,
            alive=[0, 1, 2, 3],
            rules=rules,
            dealer=0,
        )
        self.assertEqual(d, [-2, 4, -1, -1])
        self.assertEqual(sum(d), 0)


if __name__ == "__main__":
    unittest.main()
