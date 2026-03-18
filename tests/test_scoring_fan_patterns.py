import unittest

from mahjong_ai.core.state import PlayerState
from mahjong_ai.core.tiles import counts_empty, tile_id
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.fan_patterns import WinContext
from mahjong_ai.scoring.settlement import compute_fan


def _player_for(counts):
    return PlayerState(hand=counts.copy())


class TestScoringFanPatterns(unittest.TestCase):
    def test_qidui_default_fan(self):
        counts = counts_empty()
        for rank in (1, 2, 3, 4, 5, 6, 7):
            counts[tile_id(0, rank)] = 2

        fan = compute_fan(
            counts=counts,
            player=_player_for(counts),
            ctx=WinContext(winner=0, winning_tile=tile_id(0, 1), self_draw=False, from_player=1),
            rules=RulesConfig(),
        )

        self.assertEqual(fan.fan_total, 4)
        self.assertEqual(fan.patterns, {"qidui", "qingyise"})

    def test_qingyise_default_fan(self):
        counts = counts_empty()
        for rank in (1, 2, 3):
            counts[tile_id(0, rank)] += 1
        for rank in (2, 3, 4):
            counts[tile_id(0, rank)] += 1
        for rank in (5, 6, 7):
            counts[tile_id(0, rank)] += 1
        for rank in (7, 8, 9):
            counts[tile_id(0, rank)] += 1
        counts[tile_id(0, 9)] += 2

        fan = compute_fan(
            counts=counts,
            player=_player_for(counts),
            ctx=WinContext(winner=0, winning_tile=tile_id(0, 9), self_draw=False, from_player=1),
            rules=RulesConfig(),
        )

        self.assertEqual(fan.fan_total, 2)
        self.assertEqual(fan.patterns, {"qingyise"})

    def test_pengpenghu_default_fan(self):
        counts = counts_empty()
        for _ in range(3):
            counts[tile_id(0, 1)] += 1
            counts[tile_id(0, 2)] += 1
            counts[tile_id(1, 3)] += 1
            counts[tile_id(2, 4)] += 1
        counts[tile_id(1, 5)] = 2

        fan = compute_fan(
            counts=counts,
            player=_player_for(counts),
            ctx=WinContext(winner=0, winning_tile=tile_id(1, 5), self_draw=False, from_player=1),
            rules=RulesConfig(),
        )

        self.assertEqual(fan.fan_total, 2)
        self.assertEqual(fan.patterns, {"pengpenghu"})

    def test_contextual_bonus_patterns_are_scored(self):
        counts = counts_empty()
        counts[tile_id(0, 1)] = 3
        for rank in (2, 3, 4):
            counts[tile_id(0, rank)] += 1
        for rank in (5, 6, 7):
            counts[tile_id(0, rank)] += 1
        for rank in (2, 3, 4):
            counts[tile_id(1, rank)] += 1
        counts[tile_id(1, 9)] = 2

        player = _player_for(counts)
        rules = RulesConfig()

        gangshanghua = compute_fan(
            counts=counts,
            player=player,
            ctx=WinContext(winner=0, winning_tile=tile_id(1, 9), self_draw=True, from_player=None, after_kong=True),
            rules=rules,
        )
        qiangganghu = compute_fan(
            counts=counts,
            player=player,
            ctx=WinContext(winner=0, winning_tile=tile_id(0, 1), self_draw=False, from_player=1, rob_kong=True),
            rules=rules,
        )
        haidilaoyue = compute_fan(
            counts=counts,
            player=player,
            ctx=WinContext(winner=0, winning_tile=tile_id(1, 9), self_draw=True, from_player=None, last_tile_draw=True),
            rules=rules,
        )
        haidipao = compute_fan(
            counts=counts,
            player=player,
            ctx=WinContext(winner=0, winning_tile=tile_id(0, 1), self_draw=False, from_player=1, last_tile_discard=True),
            rules=rules,
        )

        self.assertEqual(gangshanghua.fan_total, 1)
        self.assertEqual(gangshanghua.patterns, {"gangshanghua"})
        self.assertEqual(qiangganghu.fan_total, 1)
        self.assertEqual(qiangganghu.patterns, {"qiangganghu"})
        self.assertEqual(haidilaoyue.fan_total, 1)
        self.assertEqual(haidilaoyue.patterns, {"haidilaoyue"})
        self.assertEqual(haidipao.fan_total, 1)
        self.assertEqual(haidipao.patterns, {"haidipao"})

    def test_fan_cap_clamps_total(self):
        counts = counts_empty()
        for rank in (1, 2, 3, 4, 5, 6, 7):
            counts[tile_id(0, rank)] = 2

        fan = compute_fan(
            counts=counts,
            player=_player_for(counts),
            ctx=WinContext(winner=0, winning_tile=tile_id(0, 7), self_draw=False, from_player=1),
            rules=RulesConfig(fan_cap=3),
        )

        self.assertEqual(fan.fan_total, 3)
        self.assertEqual(fan.patterns, {"qidui", "qingyise"})


if __name__ == "__main__":
    unittest.main()
