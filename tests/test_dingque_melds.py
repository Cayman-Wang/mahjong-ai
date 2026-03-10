import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import Meld, MeldKind, Phase, PlayerState
from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.settlement import settle_hua_zhu_and_cha_jiao


class TestDingqueMelds(unittest.TestCase):
    def test_dingque_tiles_in_melds_disallow_self_draw_hu(self):
        rules = RulesConfig(
            swap_enabled=False,
            dingque_enabled=False,
            enable_hua_zhu=False,
            enable_cha_jiao=False,
        )
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)

        # Hand is a standard win given 1 open meld (11 concealed tiles),
        # but the open meld itself violates dingque suit.
        p0 = state.players[0]
        p0.dingque_suit = 0
        p0.melds = [Meld(MeldKind.PENG, tile_id(0, 1), from_player=2)]

        c = counts_empty()
        for r in (2, 3, 4):
            c[tile_id(1, r)] += 1
        for r in (5, 6, 7):
            c[tile_id(1, r)] += 1
        for r in (2, 3, 4):
            c[tile_id(2, r)] += 1
        c[tile_id(2, 9)] += 2
        p0.hand = c

        state.phase = Phase.TURN_ACTION
        state.current_player = 0
        state.turn_has_drawn = True

        mask = engine.legal_action_mask(state, 0)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 0)

        with self.assertRaises(ValueError):
            engine.step(state, {0: encode_action(ActionKind.HU)})

    def test_hua_zhu_counts_dingque_tiles_in_melds(self):
        rules = RulesConfig(
            swap_enabled=False,
            dingque_enabled=False,
            enable_hua_zhu=True,
            enable_cha_jiao=False,
            hua_zhu_penalty=16,
        )

        players = [PlayerState() for _ in range(4)]
        for pid in range(4):
            players[pid].hand = counts_empty()
            players[pid].melds = []
            players[pid].dingque_suit = 0

        # player0 violates dingque via an open meld, even though the concealed hand has no suit0 tiles.
        players[0].melds = [Meld(MeldKind.PENG, tile_id(0, 1), from_player=2)]
        players[1].hand[tile_id(1, 1)] = 1
        players[2].hand[tile_id(1, 2)] = 1
        players[3].hand[tile_id(2, 3)] = 1

        d = settle_hua_zhu_and_cha_jiao(players=players, alive=[0, 1, 2, 3], rules=rules)
        self.assertEqual(sum(d), 0)
        self.assertEqual(d[0], -48)
        self.assertEqual(d[1], 16)
        self.assertEqual(d[2], 16)
        self.assertEqual(d[3], 16)


if __name__ == "__main__":
    unittest.main()

