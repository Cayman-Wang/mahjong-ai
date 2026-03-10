from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from mahjong_ai.core.tiles import NUM_TILE_TYPES, RANKS_PER_SUIT, tile_suit


@dataclass(frozen=True, slots=True)
class WinResult:
    ok: bool
    kind: str | None = None  # "qidui" | "standard"


def _counts_total(counts: list[int]) -> int:
    return sum(counts)


def _has_dingque_tiles(counts: list[int], dingque_suit: int | None) -> bool:
    if dingque_suit is None:
        return False
    for tid, n in enumerate(counts):
        if n and tile_suit(tid) == dingque_suit:
            return True
    return False


def is_qidui(counts: list[int]) -> bool:
    if _counts_total(counts) != 14:
        return False
    pairs = 0
    for c in counts:
        pairs += c // 2
    return pairs == 7


@lru_cache(maxsize=200_000)
def _can_meld_suit(counts9: tuple[int, ...]) -> bool:
    # All zeros -> success.
    for c in counts9:
        if c:
            break
    else:
        return True

    # Find first non-zero index.
    i = 0
    while i < RANKS_PER_SUIT and counts9[i] == 0:
        i += 1

    c0 = counts9[i]
    # Try triplet.
    if c0 >= 3:
        lst = list(counts9)
        lst[i] -= 3
        if _can_meld_suit(tuple(lst)):
            return True

    # Try sequence i,i+1,i+2.
    if i <= 6 and counts9[i + 1] > 0 and counts9[i + 2] > 0:
        lst = list(counts9)
        lst[i] -= 1
        lst[i + 1] -= 1
        lst[i + 2] -= 1
        if _can_meld_suit(tuple(lst)):
            return True

    return False


def _can_all_melds(counts: list[int]) -> bool:
    # No honors -> suits independent.
    for suit in range(3):
        base = suit * RANKS_PER_SUIT
        key = tuple(counts[base : base + RANKS_PER_SUIT])
        if not _can_meld_suit(key):
            return False
    return True


def is_standard_win(counts: list[int]) -> bool:
    total = _counts_total(counts)
    if total % 3 != 2:
        return False

    for tid in range(NUM_TILE_TYPES):
        if counts[tid] >= 2:
            tmp = counts.copy()
            tmp[tid] -= 2
            if _can_all_melds(tmp):
                return True
    return False


def detect_win(
    counts: list[int],
    *,
    meld_count: int,
    dingque_suit: int | None,
    dingque_in_melds: bool = False,
) -> WinResult:
    """Detect whether a player can win with the given concealed counts.

    The melds are assumed to be already fixed (open melds are *not* included in `counts`).
    """
    # Defensive: in most Sichuan rulesets, dingque suit tiles cannot appear in open melds.
    if dingque_in_melds:
        return WinResult(False, None)

    if _has_dingque_tiles(counts, dingque_suit):
        return WinResult(False, None)

    total = _counts_total(counts)

    # Seven pairs cannot coexist with open melds.
    if meld_count == 0 and total == 14 and is_qidui(counts):
        return WinResult(True, "qidui")

    # Standard hand: remaining tiles must match the number of melds needed.
    needed = 2 + 3 * (4 - meld_count)
    if total != needed:
        return WinResult(False, None)

    if is_standard_win(counts):
        return WinResult(True, "standard")

    return WinResult(False, None)
