import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import PendingDiscard, Phase
from mahjong_ai.core.tiles import counts_empty, tile_id


def _set_hand(player, counts):
    player.hand = counts


class TestResponseMultiHu(unittest.TestCase):
    def test_yipao_duoxiang_two_winners(self):
        engine = GameEngine(enable_events=False)
        state = engine.reset(seed=1)

        # Force a simple response situation: player0 discards 1W.
        t = tile_id(0, 1)
        discarder = 0

        state.pending_discard = PendingDiscard(from_player=discarder, tile=t, from_last_tile_draw=False)
        state.phase = Phase.RESPONSE

        # Winner1 (pid=1): 111W 234W 567W 234T 99T, missing one 1W.
        c1 = counts_empty()
        c1[tile_id(0, 1)] = 2  # missing one to make triplet
        for r in (2, 3, 4):
            c1[tile_id(0, r)] += 1
        for r in (5, 6, 7):
            c1[tile_id(0, r)] += 1
        for r in (2, 3, 4):
            c1[tile_id(1, r)] += 1
        c1[tile_id(1, 9)] += 2
        _set_hand(state.players[1], c1)
        state.players[1].dingque_suit = 2

        # Winner2 (pid=2): 123W 456W 789W 222T 55T, missing one 1W.
        c2 = counts_empty()
        # sequences W (missing 1W in 123W)
        c2[tile_id(0, 2)] += 1
        c2[tile_id(0, 3)] += 1
        for r in (4, 5, 6):
            c2[tile_id(0, r)] += 1
        for r in (7, 8, 9):
            c2[tile_id(0, r)] += 1
        c2[tile_id(1, 2)] += 3
        c2[tile_id(1, 5)] += 2
        _set_hand(state.players[2], c2)
        state.players[2].dingque_suit = 2

        # Player3 irrelevant.
        state.players[3].dingque_suit = 2

        actions = {
            1: encode_action(ActionKind.HU),
            2: encode_action(ActionKind.HU),
            3: encode_action(ActionKind.PASS),
        }
        res = engine.step(state, actions)

        self.assertTrue(state.players[1].won)
        self.assertTrue(state.players[2].won)
        self.assertEqual(sum(res.score_delta), 0)


if __name__ == "__main__":
    unittest.main()
