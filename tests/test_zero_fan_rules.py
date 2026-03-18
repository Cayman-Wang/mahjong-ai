import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import PendingDiscard, Phase, PlayerState
from mahjong_ai.core.tiles import NUM_TILE_TYPES, counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.settlement import settle_hua_zhu_and_cha_jiao
from mahjong_ai.scoring.ting import is_ting


def _plain_zero_fan_waiting_hand():
    counts = counts_empty()
    counts[tile_id(0, 1)] = 2
    for rank in (2, 3, 4):
        counts[tile_id(0, rank)] += 1
    for rank in (5, 6, 7):
        counts[tile_id(0, rank)] += 1
    for rank in (2, 3, 4):
        counts[tile_id(1, rank)] += 1
    counts[tile_id(1, 9)] += 2
    return counts


def _plain_zero_fan_win_hand():
    counts = _plain_zero_fan_waiting_hand()
    counts[tile_id(0, 1)] += 1
    return counts


def _fanful_waiting_hand():
    counts = counts_empty()
    counts[tile_id(0, 1)] = 1
    for rank in (2, 3, 4, 5, 6, 7):
        counts[tile_id(0, rank)] = 2
    return counts


class TestZeroFanRules(unittest.TestCase):
    def test_allow_zero_fan_false_blocks_self_draw_hu(self):
        rules = RulesConfig(swap_enabled=False, dingque_enabled=False, allow_zero_fan=False)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)

        winner = state.current_player
        state.players[winner].hand = _plain_zero_fan_win_hand()
        state.players[winner].dingque_suit = 2
        state.phase = Phase.TURN_ACTION
        state.turn_has_drawn = True

        mask = engine.legal_action_mask(state, winner)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 0)

        with self.assertRaises(ValueError):
            engine.step(state, {winner: encode_action(ActionKind.HU)})

    def test_allow_zero_fan_false_blocks_discard_hu(self):
        rules = RulesConfig(allow_zero_fan=False)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)

        state.phase = Phase.RESPONSE
        state.pending_discard = PendingDiscard(from_player=0, tile=tile_id(0, 1), from_last_tile_draw=False)
        state.players[1].hand = _plain_zero_fan_waiting_hand()
        state.players[1].dingque_suit = 2
        state.players[2].dingque_suit = 2
        state.players[3].dingque_suit = 2

        mask = engine.legal_action_mask(state, 1)
        self.assertEqual(mask[encode_action(ActionKind.HU)], 0)

        with self.assertRaises(ValueError):
            engine.step(
                state,
                {
                    1: encode_action(ActionKind.HU),
                    2: encode_action(ActionKind.PASS),
                    3: encode_action(ActionKind.PASS),
                },
            )

    def test_allow_zero_fan_false_only_settles_fanful_winner_in_multi_hu_window(self):
        rules = RulesConfig(allow_zero_fan=False, allow_yipao_duoxiang=True, max_round_wins=3)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)

        discard_tile = tile_id(0, 1)
        state.phase = Phase.RESPONSE
        state.pending_discard = PendingDiscard(from_player=0, tile=discard_tile, from_last_tile_draw=False)

        state.players[1].hand = _plain_zero_fan_waiting_hand()
        state.players[1].dingque_suit = 2

        state.players[2].hand = _fanful_waiting_hand()
        state.players[2].dingque_suit = None

        state.players[3].hand = [0] * NUM_TILE_TYPES
        state.players[3].dingque_suit = 2

        mask_zero_fan = engine.legal_action_mask(state, 1)
        mask_fanful = engine.legal_action_mask(state, 2)
        self.assertEqual(mask_zero_fan[encode_action(ActionKind.HU)], 0)
        self.assertEqual(mask_fanful[encode_action(ActionKind.HU)], 1)

        result = engine.step(
            state,
            {
                1: encode_action(ActionKind.PASS),
                2: encode_action(ActionKind.HU),
                3: encode_action(ActionKind.PASS),
            },
        )

        self.assertFalse(state.players[1].won)
        self.assertTrue(state.players[2].won)
        self.assertEqual(sum(result.score_delta), 0)

    def test_cha_jiao_requires_a_non_zero_fan_path_when_zero_fan_is_disabled(self):
        rules = RulesConfig(enable_hua_zhu=False, enable_cha_jiao=True, cha_jiao_penalty=8, allow_zero_fan=False)

        players = [PlayerState() for _ in range(4)]

        players[0].hand = _plain_zero_fan_waiting_hand()
        players[1].hand = counts_empty()
        players[1].hand[tile_id(0, 1)] = 1
        players[1].hand[tile_id(1, 1)] = 1
        players[1].hand[tile_id(2, 1)] = 1
        players[2].hand = _fanful_waiting_hand()

        self.assertTrue(is_ting(players[0].hand, meld_count=0, dingque_suit=None))
        self.assertTrue(is_ting(players[2].hand, meld_count=0, dingque_suit=None))

        delta = settle_hua_zhu_and_cha_jiao(players=players, alive=[0, 1, 2], rules=rules)
        self.assertEqual(sum(delta), 0)
        self.assertEqual(delta, [-8, -8, 16, 0])


if __name__ == "__main__":
    unittest.main()
