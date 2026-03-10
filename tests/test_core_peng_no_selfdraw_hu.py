import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import PendingDiscard, Phase
from mahjong_ai.core.tiles import counts_empty, tile_id


class TestPengNoSelfDrawHu(unittest.TestCase):
    def test_hu_is_illegal_after_peng_claim_turn(self):
        engine = GameEngine(enable_events=False)
        state = engine.reset(seed=1)

        # Force a discard-response situation.
        state.phase = Phase.RESPONSE
        t = tile_id(0, 1)
        state.pending_discard = PendingDiscard(from_player=0, tile=t, from_last_tile_draw=False)
        state.players[0].discards = [t]

        # Player1 can PENG the discard and would otherwise look like a self-draw winner
        # (11-tile concealed hand + 1 meld), which must NOT be treated as self-draw HU.
        c = counts_empty()
        c[tile_id(0, 1)] = 2
        for r in (2, 3, 4):
            c[tile_id(0, r)] += 1
        for r in (5, 6, 7):
            c[tile_id(0, r)] += 1
        for r in (2, 3, 4):
            c[tile_id(1, r)] += 1
        c[tile_id(1, 9)] += 2
        state.players[1].hand = c

        # Choose a dingque suit that does not block the hand.
        for pid in (1, 2, 3):
            state.players[pid].dingque_suit = 2

        engine.step(
            state,
            {
                1: encode_action(ActionKind.PENG, t),
                2: encode_action(ActionKind.PASS),
                3: encode_action(ActionKind.PASS),
            },
        )

        self.assertEqual(state.phase, Phase.TURN_ACTION)
        self.assertEqual(state.current_player, 1)
        self.assertFalse(state.turn_has_drawn)

        mask = engine.legal_action_mask(state, 1)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 0)

        with self.assertRaises(ValueError):
            engine.step(state, {1: encode_action(ActionKind.HU)})


if __name__ == "__main__":
    unittest.main()

