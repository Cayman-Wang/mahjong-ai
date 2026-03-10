import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig


class TestWallEmptyContract(unittest.TestCase):
    def test_wall_empty_settlement_is_applied_once_via_step(self):
        rules = RulesConfig(
            swap_enabled=False,
            dingque_enabled=True,
            enable_hua_zhu=True,
            enable_cha_jiao=False,
            hua_zhu_penalty=16,
        )
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=123)

        # Make the wall empty before the first draw, so the first auto-advance will settle.
        state.wall_pos = state.wall_end

        # Construct a deterministic hua_zhu situation after dingque:
        # player1 will be hua_zhu (dingque suit=0 and holds a suit0 tile), others are not.
        for pid in range(4):
            state.players[pid].hand = counts_empty()
            state.players[pid].dingque_suit = None

        state.players[0].hand[tile_id(1, 1)] = 1  # suit1
        state.players[1].hand[tile_id(0, 1)] = 1  # suit0 -> hua_zhu when dingque=0
        state.players[2].hand[tile_id(1, 2)] = 1  # suit1
        state.players[3].hand[tile_id(2, 3)] = 1  # suit2

        actions = {pid: encode_action(ActionKind.DINGQUE, 0) for pid in range(4)}
        res = engine.step(state, actions)

        self.assertTrue(res.done)
        self.assertEqual(sum(res.score_delta), 0)

        # Scores should match the delta exactly (applied once), not be doubled.
        self.assertEqual(state.scores, res.score_delta)


if __name__ == "__main__":
    unittest.main()

