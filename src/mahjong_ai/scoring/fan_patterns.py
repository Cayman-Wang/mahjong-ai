from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.core.state import Meld
from mahjong_ai.core.tiles import NUM_TILE_TYPES, tile_suit


@dataclass(frozen=True, slots=True)
class WinContext:
    winner: int
    winning_tile: int

    self_draw: bool
    from_player: int | None

    after_kong: bool = False
    rob_kong: bool = False

    # For last-tile bonuses.
    last_tile_draw: bool = False
    last_tile_discard: bool = False


def _all_suits_used(counts: list[int], melds: list[Meld]) -> set[int]:
    suits: set[int] = set()
    for tid, n in enumerate(counts):
        if n:
            suits.add(tile_suit(tid))
    for m in melds:
        suits.add(tile_suit(m.tile))
    return suits


def is_qingyise(counts: list[int], melds: list[Meld]) -> bool:
    suits = _all_suits_used(counts, melds)
    return len(suits) == 1


def is_pengpenghu(counts: list[int]) -> bool:
    # True if the concealed portion can be decomposed into (triplets only) + pair.
    for tid in range(NUM_TILE_TYPES):
        if counts[tid] >= 2:
            tmp = counts.copy()
            tmp[tid] -= 2
            if all((c % 3) == 0 for c in tmp):
                return True
    return False


def detect_fan_patterns(
    *,
    counts: list[int],
    melds: list[Meld],
    win_kind: str,
    ctx: WinContext,
) -> set[str]:
    patterns: set[str] = set()

    if win_kind == "qidui":
        patterns.add("qidui")

    if is_qingyise(counts, melds):
        patterns.add("qingyise")

    if win_kind == "standard" and is_pengpenghu(counts):
        patterns.add("pengpenghu")

    if ctx.after_kong and ctx.self_draw:
        patterns.add("gangshanghua")

    if ctx.rob_kong:
        patterns.add("qiangganghu")

    if ctx.self_draw and ctx.last_tile_draw:
        patterns.add("haidilaoyue")

    if (not ctx.self_draw) and ctx.last_tile_discard:
        patterns.add("haidipao")

    return patterns
