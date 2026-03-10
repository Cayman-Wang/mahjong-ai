import unittest

from mahjong_ai.core.actions import ActionKind, decode_action, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import Phase
from mahjong_ai.core.tiles import NUM_TILE_TYPES, tile_suit


def _total_wall_and_hands_and_swap(state) -> int:
    wall_remaining = state.wall_end - state.wall_pos
    hands = sum(sum(p.hand) for p in state.players)
    swap = sum(len(x) for x in state.swap_picks)
    return int(wall_remaining + hands + swap)


class TestSwapAndDingque(unittest.TestCase):
    def test_swap_conservation(self):
        engine = GameEngine(enable_events=False)
        state = engine.reset(seed=123)
        self.assertEqual(state.phase, Phase.SWAP_PICK_1)
        self.assertEqual(_total_wall_and_hands_and_swap(state), 108)

        for _ in range(3):
            required = engine.required_players(state)
            actions = {}
            for pid in required:
                tid = next(i for i, n in enumerate(state.players[pid].hand) if n > 0)
                actions[pid] = encode_action(ActionKind.SWAP_PICK, tid)
            _ = engine.step(state, actions)
            self.assertEqual(_total_wall_and_hands_and_swap(state), 108)

        self.assertEqual(state.phase, Phase.DINGQUE)

    def test_dingque_forced_discard_mask(self):
        engine = GameEngine(enable_events=False)
        state = engine.reset(seed=7)

        # Finish 3 swap pick rounds quickly.
        for _ in range(3):
            required = engine.required_players(state)
            actions = {}
            for pid in required:
                tid = next(i for i, n in enumerate(state.players[pid].hand) if n > 0)
                actions[pid] = encode_action(ActionKind.SWAP_PICK, tid)
            _ = engine.step(state, actions)

        self.assertEqual(state.phase, Phase.DINGQUE)

        dealer = state.current_player
        # Pick a dingque suit that the dealer actually has tiles of.
        suit_counts = [0, 0, 0]
        for tid, n in enumerate(state.players[dealer].hand):
            if n:
                suit_counts[tile_suit(tid)] += n
        dingque = next(s for s in range(3) if suit_counts[s] > 0)

        actions = {pid: encode_action(ActionKind.DINGQUE, 0) for pid in engine.required_players(state)}
        actions[dealer] = encode_action(ActionKind.DINGQUE, dingque)
        _ = engine.step(state, actions)

        # After dingque, the engine auto-draws and should land on a TURN_ACTION decision.
        self.assertEqual(state.phase, Phase.TURN_ACTION)
        self.assertEqual(state.current_player, dealer)

        mask = engine.legal_action_mask(state, dealer)
        discard_tiles = []
        for a, m in enumerate(mask):
            if m != 1:
                continue
            da = decode_action(a)
            if da.kind == ActionKind.DISCARD:
                assert da.arg is not None
                discard_tiles.append(da.arg)

        self.assertTrue(discard_tiles)
        # All legal discards must be of the dingque suit while the player still has dingque tiles.
        for tid in discard_tiles:
            self.assertEqual(tile_suit(tid), dingque)


if __name__ == "__main__":
    unittest.main()
