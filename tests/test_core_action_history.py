from __future__ import annotations

import unittest

from mahjong_ai.core.actions import ActionKind, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import RECENT_ACTION_HISTORY_LIMIT
from mahjong_ai.rules.schema import RulesConfig


def _first_legal(mask: list[int]) -> int:
    for i, v in enumerate(mask):
        if v == 1:
            return i
    raise AssertionError("no legal action in mask")


class TestCoreActionHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GameEngine(rules=RulesConfig(), enable_events=False)
        self.state = self.engine.reset(seed=1)

    def test_swap_pick_actions_are_not_recorded_publicly(self) -> None:
        required = self.engine.required_players(self.state)
        actions = {
            pid: _first_legal(self.engine.legal_action_mask(self.state, pid))
            for pid in required
        }

        self.engine.step(self.state, actions)

        self.assertEqual(self.state.phase.value, "swap_pick_2")
        self.assertEqual(self.state.recent_actions, [])

    def test_recent_actions_recorded_after_public_turn_action(self) -> None:
        engine = GameEngine(rules=RulesConfig(swap_enabled=False, dingque_enabled=False), enable_events=False)
        state = engine.reset(seed=2)

        required = engine.required_players(state)
        self.assertEqual(len(required), 1)
        pid = required[0]
        action = _first_legal(engine.legal_action_mask(state, pid))

        engine.step(state, {pid: action})

        self.assertEqual(len(state.recent_actions), 1)
        self.assertEqual(state.recent_actions[0].actor, pid)
        self.assertEqual(state.recent_actions[0].action_id, action)

    def test_recent_actions_history_is_capped(self) -> None:
        pass_action = encode_action(ActionKind.PASS)
        total = RECENT_ACTION_HISTORY_LIMIT + 5
        for i in range(total):
            self.engine._record_recent_action(self.state, pid=i % 4, action_id=pass_action)

        self.assertEqual(len(self.state.recent_actions), RECENT_ACTION_HISTORY_LIMIT)
        expected_first_actor = (total - RECENT_ACTION_HISTORY_LIMIT) % 4
        self.assertEqual(self.state.recent_actions[0].actor, expected_first_actor)
        self.assertTrue(all(t.action_id == pass_action for t in self.state.recent_actions))


if __name__ == "__main__":
    unittest.main()
