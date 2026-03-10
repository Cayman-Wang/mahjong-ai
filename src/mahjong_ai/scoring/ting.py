from __future__ import annotations

from mahjong_ai.core.tiles import NUM_TILE_TYPES, tile_suit

from .win_check import detect_win


def is_ting(counts: list[int], *, meld_count: int, dingque_suit: int | None) -> bool:
    """Return True if the hand is in a listening state (one tile away from winning)."""
    # `detect_win` already checks dingque suitability, but we short-circuit here.
    if dingque_suit is not None:
        for tid, n in enumerate(counts):
            if n and tile_suit(tid) == dingque_suit:
                return False

    for tid in range(NUM_TILE_TYPES):
        if counts[tid] >= 4:
            continue
        tmp = counts.copy()
        tmp[tid] += 1
        if detect_win(tmp, meld_count=meld_count, dingque_suit=dingque_suit).ok:
            return True
    return False
