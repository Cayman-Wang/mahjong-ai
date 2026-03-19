from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.training.self_play_replay import (
    build_self_play_replay_trace,
    render_replay_text,
    write_replay_artifacts,
)


def _first_legal_action(_: int, __, action_mask: list[int], ___: str) -> int:
    for action_id, allowed in enumerate(action_mask):
        if int(allowed) == 1:
            return int(action_id)
    raise RuntimeError("action mask has no legal action")


class _IllegalThenFallback:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _: int, __, action_mask: list[int], ___: str) -> int:
        self.calls += 1
        if self.calls == 1:
            return -1
        for action_id, allowed in enumerate(action_mask):
            if int(allowed) == 1:
                return int(action_id)
        raise RuntimeError("action mask has no legal action")


class TestSelfPlayReplay(unittest.TestCase):
    def test_render_omniscient_and_seat_views(self) -> None:
        trace = build_self_play_replay_trace(
            rules=RulesConfig(),
            seed=7,
            policy_assignments={pid: "shared_policy" for pid in range(4)},
            action_selector=_first_legal_action,
            max_steps=5000,
            strict_illegal_action=True,
            rules_path="configs/rules/sichuan_xuezhan_default.yaml",
        )

        omniscient_text = render_replay_text(trace, view="omniscient")
        seat_text = render_replay_text(trace, view="seat", seat=0)

        self.assertIn("self_play_replay view=omniscient", omniscient_text)
        self.assertIn("rules_path=configs/rules/sichuan_xuezhan_default.yaml", omniscient_text)
        self.assertIn("P1 hand=", omniscient_text)
        self.assertNotIn("<hidden>", omniscient_text)

        self.assertIn("self_play_replay view=seat0", seat_text)
        self.assertIn("P0 hand=", seat_text)
        self.assertIn("P1 hand=<hidden>", seat_text)
        self.assertIn("action=<hidden_swap_pick>", seat_text)
        self.assertIn("tile=<hidden_draw>", seat_text)
        self.assertIn("tile=<hidden_swap_pick>", seat_text)

    def test_write_replay_artifacts_writes_expected_files(self) -> None:
        trace = build_self_play_replay_trace(
            rules=RulesConfig(),
            seed=11,
            policy_assignments={pid: "shared_policy" for pid in range(4)},
            action_selector=_first_legal_action,
            max_steps=5000,
            strict_illegal_action=True,
        )

        with tempfile.TemporaryDirectory() as td:
            written = write_replay_artifacts(
                output_dir=Path(td),
                trace=trace,
                include_omniscient=True,
                seat_views=[0],
            )

            self.assertEqual(
                [path.name for path in written],
                ["seed_11_omniscient.txt", "seed_11_seat0.txt"],
            )
            for path in written:
                self.assertTrue(path.exists())
                self.assertIn("self_play_replay", path.read_text(encoding="utf-8"))

    def test_illegal_action_is_recorded_when_not_strict(self) -> None:
        selector = _IllegalThenFallback()
        trace = build_self_play_replay_trace(
            rules=RulesConfig(),
            seed=13,
            policy_assignments={pid: "shared_policy" for pid in range(4)},
            action_selector=selector,
            max_steps=5000,
            strict_illegal_action=False,
        )

        self.assertEqual(trace.frames[0].invalid_action_ids, {0: -1})
        rendered = render_replay_text(trace, view="omniscient")
        self.assertIn("invalid_selected=-1", rendered)


if __name__ == "__main__":
    unittest.main()
