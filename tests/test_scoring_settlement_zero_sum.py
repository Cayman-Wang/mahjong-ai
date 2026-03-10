import random
import unittest

from mahjong_ai.core.state import PlayerState
from mahjong_ai.core.tiles import all_tiles, counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.settlement import settle_hua_zhu_and_cha_jiao
from mahjong_ai.scoring.ting import is_ting


class TestSettlementZeroSum(unittest.TestCase):
    def test_hua_zhu_zero_sum(self):
        rules = RulesConfig(enable_hua_zhu=True, enable_cha_jiao=False, hua_zhu_penalty=16)

        players = [PlayerState() for _ in range(4)]
        # player0 is hua_zhu (dingque=wan but still has wan).
        players[0].dingque_suit = 0
        players[0].hand[tile_id(0, 1)] = 1

        # player1 is not hua_zhu.
        players[1].dingque_suit = 0
        players[1].hand[tile_id(1, 1)] = 1

        alive = [0, 1]
        d = settle_hua_zhu_and_cha_jiao(players=players, alive=alive, rules=rules)
        self.assertEqual(sum(d), 0)
        self.assertEqual(d[0], -16)
        self.assertEqual(d[1], 16)

    def test_cha_jiao_zero_sum(self):
        rules = RulesConfig(enable_hua_zhu=False, enable_cha_jiao=True, cha_jiao_penalty=8)

        players = [PlayerState() for _ in range(4)]

        # player0: a known ting hand (waiting for 1W).
        c0 = counts_empty()
        c0[tile_id(0, 1)] = 2
        for r in (2, 3, 4):
            c0[tile_id(0, r)] += 1
        for r in (5, 6, 7):
            c0[tile_id(0, r)] += 1
        for r in (2, 3, 4):
            c0[tile_id(1, r)] += 1
        c0[tile_id(1, 9)] += 2
        players[0].hand = c0
        players[0].dingque_suit = None
        self.assertTrue(is_ting(players[0].hand, meld_count=0, dingque_suit=None))

        # player1: find a deterministic non-ting hand.
        rng = random.Random(0)
        wall = all_tiles()
        c1 = None
        for _ in range(500):
            rng.shuffle(wall)
            hand = wall[:13]
            c = counts_empty()
            for t in hand:
                c[t] += 1
            if not is_ting(c, meld_count=0, dingque_suit=None):
                c1 = c
                break
        self.assertIsNotNone(c1)
        players[1].hand = c1  # type: ignore[assignment]
        players[1].dingque_suit = None

        alive = [0, 1]
        d = settle_hua_zhu_and_cha_jiao(players=players, alive=alive, rules=rules)
        self.assertEqual(sum(d), 0)
        # non-ting pays ting.
        self.assertEqual(d[0], 8)
        self.assertEqual(d[1], -8)


if __name__ == "__main__":
    unittest.main()
