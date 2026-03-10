import unittest

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import Phase
from mahjong_ai.rules.schema import RulesConfig


class TestResetConfigs(unittest.TestCase):
    def test_reset_without_swap_and_dingque(self):
        rules = RulesConfig(swap_enabled=False, dingque_enabled=False)
        engine = GameEngine(rules=rules, enable_events=False)
        state = engine.reset(seed=1)
        self.assertEqual(state.phase, Phase.TURN_ACTION)


if __name__ == "__main__":
    unittest.main()

