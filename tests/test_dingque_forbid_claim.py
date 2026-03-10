import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import PendingDiscard, Phase
from mahjong_ai.core.tiles import counts_empty, tile_id


class TestDingqueForbidClaim(unittest.TestCase):
    def test_cannot_peng_or_ming_gang_dingque_suit(self):
        engine = GameEngine(enable_events=False)
        state = engine.reset(seed=1)

        state.phase = Phase.RESPONSE
        t = tile_id(0, 1)  # suit0
        state.pending_discard = PendingDiscard(from_player=0, tile=t, from_last_tile_draw=False)

        c = counts_empty()
        c[t] = 3  # enough for PENG and GANG_MING
        state.players[1].hand = c
        state.players[1].dingque_suit = 0  # cannot claim suit0
        state.players[2].dingque_suit = 1
        state.players[3].dingque_suit = 1

        mask = engine.legal_action_mask(state, 1)
        self.assertEqual(mask[encode_action(ActionKind.PENG, t)], 0)
        self.assertEqual(mask[encode_action(ActionKind.GANG_MING, t)], 0)
        self.assertEqual(mask[encode_action(ActionKind.PASS)], 1)

        with self.assertRaises(ValueError):
            engine.step(
                state,
                {
                    1: encode_action(ActionKind.PENG, t),
                    2: encode_action(ActionKind.PASS),
                    3: encode_action(ActionKind.PASS),
                },
            )


if __name__ == "__main__":
    unittest.main()

