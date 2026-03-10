from __future__ import annotations

import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import MeldKind, PendingDiscard, Phase
from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig


def _total_tiles(state) -> int:
    wall_remaining = state.wall_end - state.wall_pos
    hands = sum(sum(p.hand) for p in state.players)
    melds = sum(sum(m.size for m in p.melds) for p in state.players)
    discards = sum(len(p.discards) for p in state.players)
    swap = sum(len(x) for x in state.swap_picks)
    return int(wall_remaining + hands + melds + discards + swap)


def _public_visible_count(state, tid: int) -> int:
    discards = sum(1 for p in state.players for x in p.discards if x == tid)
    melds = 0
    for p in state.players:
        for m in p.melds:
            if m.tile == tid:
                melds += m.size
    return discards + melds


class TestResponseClaimConservation(unittest.TestCase):
    def _base_state(self, *, seed: int, claimed_tid: int) -> tuple[GameEngine, object]:
        engine = GameEngine(rules=RulesConfig(), enable_events=False)
        state = engine.reset(seed=seed)

        for pid in range(4):
            state.players[pid].hand = counts_empty()
            state.players[pid].melds = []
            state.players[pid].discards = []
            state.players[pid].dingque_suit = None
            state.players[pid].won = False

        state.phase = Phase.RESPONSE
        state.pending_discard = PendingDiscard(from_player=0, tile=claimed_tid, from_last_tile_draw=False)
        state.players[0].discards = [tile_id(2, 2), claimed_tid]
        return engine, state

    def test_peng_claim_pops_discard_and_keeps_tile_count(self) -> None:
        tid = tile_id(0, 1)
        engine, state = self._base_state(seed=3, claimed_tid=tid)

        claimer = 1
        state.players[claimer].hand[tid] = 2

        before = _total_tiles(state)
        res = engine.step(
            state,
            {
                claimer: encode_action(ActionKind.PENG, tid),
                2: encode_action(ActionKind.PASS),
                3: encode_action(ActionKind.PASS),
            },
        )
        after = _total_tiles(state)

        self.assertFalse(res.done)
        self.assertEqual(before, after)
        self.assertEqual(state.players[0].discards, [tile_id(2, 2)])
        self.assertEqual(state.players[claimer].melds[-1].kind, MeldKind.PENG)
        self.assertLessEqual(_public_visible_count(state, tid), 4)

    def test_ming_gang_claim_pops_discard_and_keeps_tile_count(self) -> None:
        tid = tile_id(0, 1)
        engine, state = self._base_state(seed=4, claimed_tid=tid)

        claimer = 1
        state.players[claimer].hand[tid] = 3

        before = _total_tiles(state)
        res = engine.step(
            state,
            {
                claimer: encode_action(ActionKind.GANG_MING, tid),
                2: encode_action(ActionKind.PASS),
                3: encode_action(ActionKind.PASS),
            },
        )
        after = _total_tiles(state)

        self.assertFalse(res.done)
        self.assertEqual(before, after)
        self.assertEqual(state.players[0].discards, [tile_id(2, 2)])
        self.assertEqual(state.players[claimer].melds[-1].kind, MeldKind.GANG_MING)
        self.assertLessEqual(_public_visible_count(state, tid), 4)

    def test_claim_rejects_discard_stack_mismatch(self) -> None:
        tid = tile_id(0, 1)
        engine, state = self._base_state(seed=5, claimed_tid=tid)

        state.players[0].discards = [tile_id(2, 2)]
        state.players[1].hand[tid] = 2

        with self.assertRaisesRegex(ValueError, "top discard"):
            engine.step(
                state,
                {
                    1: encode_action(ActionKind.PENG, tid),
                    2: encode_action(ActionKind.PASS),
                    3: encode_action(ActionKind.PASS),
                },
            )


if __name__ == "__main__":
    unittest.main()
