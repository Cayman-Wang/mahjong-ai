from __future__ import annotations

import unittest

from mahjong_ai.core.actions import ActionKind, N_ACTIONS, encode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import ActionTrace, Phase
from mahjong_ai.env import obs_vector_encoder as vec
from mahjong_ai.rules.schema import RulesConfig


@unittest.skipUnless(vec.HAS_NUMPY, "numpy is required for vector observation tests")
class TestObsVectorEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GameEngine(rules=RulesConfig(), enable_events=False)
        self.state = self.engine.reset(seed=1)

    def test_vector_shape_and_dtype(self) -> None:
        v = vec.encode_observation_vector(self.state, pid=0)
        self.assertEqual(v.shape, (vec.OBS_VECTOR_DIM,))
        self.assertEqual(str(v.dtype), "float32")

    def test_phase_encoding_is_one_hot(self) -> None:
        for phase in Phase:
            self.state.phase = phase
            v = vec.encode_observation_vector(self.state, pid=0)
            phase_vec = v[vec.PHASE_SLICE]
            self.assertAlmostEqual(float(phase_vec.sum()), 1.0)
            self.assertEqual(int(phase_vec.argmax()), vec.phase_to_index(phase))

    def test_pending_none_encoded_in_last_slot(self) -> None:
        self.state.pending_discard = None
        self.state.pending_kong = None
        v = vec.encode_observation_vector(self.state, pid=0)

        pending_discard_tile = v[vec.PENDING_DISCARD_TILE_SLICE]
        pending_discard_from = v[vec.PENDING_DISCARD_FROM_SLICE]
        pending_kong_tile = v[vec.PENDING_KONG_TILE_SLICE]
        pending_kong_actor = v[vec.PENDING_KONG_ACTOR_SLICE]

        self.assertEqual(int(pending_discard_tile.argmax()), len(pending_discard_tile) - 1)
        self.assertEqual(int(pending_discard_from.argmax()), len(pending_discard_from) - 1)
        self.assertEqual(int(pending_kong_tile.argmax()), len(pending_kong_tile) - 1)
        self.assertEqual(int(pending_kong_actor.argmax()), len(pending_kong_actor) - 1)

    def test_phase_progress_is_one_hot(self) -> None:
        v = vec.encode_observation_vector(self.state, pid=0)
        phase_progress = v[vec.PHASE_PROGRESS_SLICE]
        self.assertAlmostEqual(float(phase_progress.sum()), 1.0)
        self.assertEqual(int((phase_progress > 0).sum()), 1)

    def test_new_public_stat_features_are_normalized(self) -> None:
        v = vec.encode_observation_vector(self.state, pid=0)

        self.assertGreaterEqual(float(v[vec.TILES_PLAYED_RATIO_IDX]), 0.0)
        self.assertLessEqual(float(v[vec.TILES_PLAYED_RATIO_IDX]), 1.0)

        self.assertGreaterEqual(float(v[vec.ALIVE_COUNT_IDX]), 0.0)
        self.assertLessEqual(float(v[vec.ALIVE_COUNT_IDX]), 1.0)
        self.assertAlmostEqual(float(v[vec.ALIVE_COUNT_IDX]), 1.0)

        discard_counts = v[vec.DISCARD_COUNT_SLICE]
        meld_tile_counts = v[vec.MELD_TILE_COUNT_SLICE]
        public_hist = v[vec.PUBLIC_TILE_HIST_SLICE]

        self.assertTrue((discard_counts >= 0.0).all())
        self.assertTrue((discard_counts <= 1.0).all())
        self.assertTrue((meld_tile_counts >= 0.0).all())
        self.assertTrue((meld_tile_counts <= 1.0).all())
        self.assertTrue((public_hist >= 0.0).all())
        self.assertTrue((public_hist <= 1.0).all())

        # At reset there is no public discard/meld information yet.
        self.assertAlmostEqual(float(discard_counts.sum()), 0.0)
        self.assertAlmostEqual(float(meld_tile_counts.sum()), 0.0)
        self.assertAlmostEqual(float(public_hist.sum()), 0.0)

    def test_recent_action_history_default_is_padded(self) -> None:
        v = vec.encode_observation_vector(self.state, pid=0)
        recent = v[vec.RECENT_ACTION_SLICE].reshape(vec.RECENT_ACTION_SLOTS, vec.RECENT_ACTION_SLOT_DIM)
        actor_dim = vec.RECENT_ACTION_ACTOR_DIM
        kind_dim = vec.RECENT_ACTION_KIND_DIM

        for slot in recent:
            actor_part = slot[:actor_dim]
            kind_part = slot[actor_dim : actor_dim + kind_dim]
            self.assertEqual(int(actor_part.argmax()), actor_dim - 1)
            self.assertEqual(int(kind_part.argmax()), vec.RECENT_ACTION_KIND_PAD_IDX)
            self.assertAlmostEqual(float(slot[-1]), 0.0)

    def test_recent_action_history_encodes_latest_slots(self) -> None:
        pass_action = encode_action(ActionKind.PASS)
        dingque_action = encode_action(ActionKind.DINGQUE, 1)
        self.state.recent_actions = [
            ActionTrace(actor=1, action_id=pass_action),
            ActionTrace(actor=2, action_id=dingque_action),
        ]

        v = vec.encode_observation_vector(self.state, pid=0)
        recent = v[vec.RECENT_ACTION_SLICE].reshape(vec.RECENT_ACTION_SLOTS, vec.RECENT_ACTION_SLOT_DIM)

        actor_dim = vec.RECENT_ACTION_ACTOR_DIM
        kind_dim = vec.RECENT_ACTION_KIND_DIM

        prev_slot = recent[-2]
        prev_actor = prev_slot[:actor_dim]
        prev_kind = prev_slot[actor_dim : actor_dim + kind_dim]
        self.assertEqual(int(prev_actor.argmax()), 1)
        self.assertEqual(int(prev_kind.argmax()), vec.ACTION_KIND_TO_INDEX[ActionKind.PASS])
        self.assertAlmostEqual(float(prev_slot[-1]), float(pass_action) / float(N_ACTIONS - 1))

        last_slot = recent[-1]
        last_actor = last_slot[:actor_dim]
        last_kind = last_slot[actor_dim : actor_dim + kind_dim]
        self.assertEqual(int(last_actor.argmax()), 2)
        self.assertEqual(int(last_kind.argmax()), vec.ACTION_KIND_TO_INDEX[ActionKind.DINGQUE])
        self.assertAlmostEqual(float(last_slot[-1]), float(dingque_action) / float(N_ACTIONS - 1))

    def test_recent_action_history_does_not_leak_swap_pick_tiles(self) -> None:
        required = self.engine.required_players(self.state)
        actions = {
            pid: next(i for i, allowed in enumerate(self.engine.legal_action_mask(self.state, pid)) if allowed == 1)
            for pid in required
        }
        self.engine.step(self.state, actions)

        # Swap picks are hidden information and should not appear in shared history.
        self.assertEqual(self.state.recent_actions, [])

        v = vec.encode_observation_vector(self.state, pid=1)
        recent = v[vec.RECENT_ACTION_SLICE].reshape(vec.RECENT_ACTION_SLOTS, vec.RECENT_ACTION_SLOT_DIM)
        actor_dim = vec.RECENT_ACTION_ACTOR_DIM
        kind_dim = vec.RECENT_ACTION_KIND_DIM

        kind_indices = [
            int(slot[actor_dim : actor_dim + kind_dim].argmax())
            for slot in recent
        ]
        self.assertNotIn(vec.ACTION_KIND_TO_INDEX[ActionKind.SWAP_PICK], kind_indices)


if __name__ == "__main__":
    unittest.main()
