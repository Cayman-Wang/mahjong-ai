import random
import unittest

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import Phase


def _total_tiles(state) -> int:
    wall_remaining = state.wall_end - state.wall_pos
    hands = sum(sum(player.hand) for player in state.players)
    melds = sum(sum(meld.size for meld in player.melds) for player in state.players)
    discards = sum(len(player.discards) for player in state.players)
    swap = sum(len(picks) for picks in state.swap_picks)
    return int(wall_remaining + hands + melds + discards + swap)


class TestCoreRandomizedInvariants(unittest.TestCase):
    def test_random_legal_games_preserve_invariants(self):
        rng = random.Random(0)
        engine = GameEngine(enable_events=False)

        for seed in range(50):
            with self.subTest(seed=seed):
                state = engine.reset(seed=seed)
                self.assertEqual(_total_tiles(state), 108)
                self.assertEqual(sum(state.scores), 0)

                for step_idx in range(2000):
                    if state.phase == Phase.ROUND_END:
                        break

                    required = engine.required_players(state)
                    self.assertTrue(required, f"no required players at seed={seed}, step={step_idx}")
                    self.assertEqual(len(required), len(set(required)))

                    actions = {}
                    for pid in required:
                        mask = engine.legal_action_mask(state, pid)
                        legal = [action_id for action_id, allowed in enumerate(mask) if allowed]
                        self.assertTrue(
                            legal,
                            f"no legal actions at seed={seed}, step={step_idx}, pid={pid}, phase={state.phase}",
                        )
                        actions[pid] = rng.choice(legal)

                    result = engine.step(state, actions)
                    self.assertEqual(sum(result.score_delta), 0)
                    self.assertEqual(sum(state.scores), 0)
                    self.assertEqual(_total_tiles(state), 108)

                    if state.phase == Phase.ROUND_END:
                        self.assertTrue(result.done)
                        self.assertIsNone(state.pending_discard)
                        self.assertIsNone(state.pending_kong)
                        self.assertEqual(engine.required_players(state), [])
                        break
                else:
                    self.fail(f"game did not end within step budget for seed={seed}")


if __name__ == "__main__":
    unittest.main()
