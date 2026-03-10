import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import Meld, MeldKind, PendingKong, Phase
from mahjong_ai.core.tiles import NUM_TILE_TYPES, counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig


def _total_tiles(state) -> int:
    wall_remaining = state.wall_end - state.wall_pos
    hands = sum(sum(p.hand) for p in state.players)
    melds = sum(sum(m.size for m in p.melds) for p in state.players)
    discards = sum(len(p.discards) for p in state.players)
    swap = sum(len(x) for x in state.swap_picks)
    return int(wall_remaining + hands + melds + discards + swap)


class TestResponseQianggang(unittest.TestCase):
    def test_illegal_hu_action_raises(self):
        rules = RulesConfig(max_round_wins=1)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)

        actor = 0
        robber = 1
        t = tile_id(0, 1)

        state.players[actor].melds = [Meld(MeldKind.PENG, t, from_player=3)]
        state.players[actor].hand = counts_empty()
        state.players[actor].hand[t] = 1

        # Robber definitely cannot HU: their hand is empty so +1 tile can't make a valid 14-tile win.
        state.players[robber].hand = counts_empty()

        state.phase = Phase.RESPONSE_QIANGGANG
        state.pending_kong = PendingKong(actor=actor, tile=t, meld_index=0)

        mask = engine.legal_action_mask(state, robber)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 0)

        actions = {
            robber: encode_action(ActionKind.HU),  # illegal
            2: encode_action(ActionKind.PASS),
            3: encode_action(ActionKind.PASS),
        }
        with self.assertRaises(ValueError):
            engine.step(state, actions)

    def test_rob_kong_tile_is_public_and_conserved(self):
        rules = RulesConfig(max_round_wins=1)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=2)

        actor = 0
        winner = 1
        t = tile_id(0, 1)

        state.players[actor].melds = [Meld(MeldKind.PENG, t, from_player=3)]
        state.players[actor].hand = counts_empty()
        state.players[actor].hand[t] = 1

        # Build a 13-tile hand for winner that wins on +1W.
        c = counts_empty()
        c[tile_id(0, 1)] = 2
        for r in (2, 3, 4):
            c[tile_id(0, r)] += 1
        for r in (5, 6, 7):
            c[tile_id(0, r)] += 1
        for r in (2, 3, 4):
            c[tile_id(1, r)] += 1
        c[tile_id(1, 9)] += 2
        self.assertEqual(sum(c), 13)
        state.players[winner].hand = c
        state.players[winner].dingque_suit = 2

        # Ensure other players don't accidentally HU.
        for pid in (2, 3):
            state.players[pid].hand = [0] * NUM_TILE_TYPES
            state.players[pid].dingque_suit = 2

        state.phase = Phase.RESPONSE_QIANGGANG
        state.pending_kong = PendingKong(actor=actor, tile=t, meld_index=0)

        mask = engine.legal_action_mask(state, winner)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 1)

        before = _total_tiles(state)
        before_discards = len(state.players[actor].discards)

        res = engine.step(
            state,
            {
                winner: encode_action(ActionKind.HU),
                2: encode_action(ActionKind.PASS),
                3: encode_action(ActionKind.PASS),
            },
        )

        self.assertTrue(res.done)
        self.assertIsNone(state.pending_kong)
        self.assertTrue(state.players[winner].won)

        # Actor loses the claimed tile from hand, but keeps their original PENG meld.
        self.assertEqual(state.players[actor].hand[t], 0)
        self.assertEqual(len(state.players[actor].melds), 1)
        self.assertEqual(state.players[actor].melds[0].kind, MeldKind.PENG)
        self.assertEqual(state.players[actor].melds[0].tile, t)

        # The robbed tile is recorded as public information.
        self.assertEqual(len(state.players[actor].discards), before_discards + 1)
        self.assertEqual(state.players[actor].discards[-1], t)

        after = _total_tiles(state)
        self.assertEqual(before, after, "rob-kong HU must conserve total tiles in state")
        self.assertEqual(sum(res.score_delta), 0)


if __name__ == "__main__":
    unittest.main()

