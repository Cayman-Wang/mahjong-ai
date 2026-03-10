import unittest

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.rng import RNG
from mahjong_ai.core.state import Phase
from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig


class TestWallEmpty(unittest.TestCase):
    def test_wall_empty_settlement_zero_sum_and_alive_only(self):
        rules = RulesConfig(
            swap_enabled=False,
            dingque_enabled=False,
            enable_hua_zhu=True,
            enable_cha_jiao=False,
            hua_zhu_penalty=16,
        )
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=123)

        # Mark player0 as already won; wall-empty settlement should only involve alive players.
        state.players[0].won = True

        # Construct a deterministic hua_zhu situation among alive players [1,2,3]:
        # player1 is hua_zhu (still has dingque suit tiles), players2/3 are not.
        for pid in (1, 2, 3):
            state.players[pid].hand = counts_empty()
            state.players[pid].dingque_suit = 0

        state.players[1].hand[tile_id(0, 1)] = 1  # hua_zhu
        state.players[2].hand[tile_id(1, 1)] = 1  # non-hua
        state.players[3].hand[tile_id(2, 1)] = 1  # non-hua

        # Force wall empty and run auto-advance.
        state.phase = Phase.TURN_DRAW
        state.wall_pos = state.wall_end

        rng = RNG.from_seed(state.seed)
        rng.setstate(state.rng_state)
        events = []
        d = engine._auto_advance(state, events, rng=rng)

        self.assertEqual(state.phase, Phase.ROUND_END)
        self.assertEqual(sum(d), 0)

        # player1 pays 16 to each of (2,3).
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], -32)
        self.assertEqual(d[2], 16)
        self.assertEqual(d[3], 16)


if __name__ == "__main__":
    unittest.main()
