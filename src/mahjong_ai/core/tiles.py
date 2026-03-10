from __future__ import annotations

from dataclasses import dataclass

NUM_SUITS = 3
RANKS_PER_SUIT = 9
NUM_TILE_TYPES = NUM_SUITS * RANKS_PER_SUIT  # 27
COPIES_PER_TILE = 4

SUIT_NAMES = ("wan", "tiao", "tong")
SUIT_SHORT = ("W", "T", "B")


def tile_id(suit: int, rank: int) -> int:
    """Return tile id for suit in [0..2] and rank in [1..9]."""
    if suit < 0 or suit >= NUM_SUITS:
        raise ValueError(f"invalid suit: {suit}")
    if rank < 1 or rank > RANKS_PER_SUIT:
        raise ValueError(f"invalid rank: {rank}")
    return suit * RANKS_PER_SUIT + (rank - 1)


def tile_suit(tid: int) -> int:
    if tid < 0 or tid >= NUM_TILE_TYPES:
        raise ValueError(f"invalid tile id: {tid}")
    return tid // RANKS_PER_SUIT


def tile_rank(tid: int) -> int:
    if tid < 0 or tid >= NUM_TILE_TYPES:
        raise ValueError(f"invalid tile id: {tid}")
    return (tid % RANKS_PER_SUIT) + 1


def pretty_tile(tid: int) -> str:
    """Human-readable tile string."""
    return f"{tile_rank(tid)}{SUIT_SHORT[tile_suit(tid)]}"


def all_tiles() -> list[int]:
    """Return a full 108-tile wall (shuffled by the caller)."""
    wall: list[int] = []
    for tid in range(NUM_TILE_TYPES):
        wall.extend([tid] * COPIES_PER_TILE)
    return wall


def counts_empty() -> list[int]:
    return [0] * NUM_TILE_TYPES


def counts_total(counts: list[int]) -> int:
    return sum(counts)


def counts_add(counts: list[int], tid: int, n: int = 1) -> None:
    counts[tid] += n


def counts_remove(counts: list[int], tid: int, n: int = 1) -> None:
    if counts[tid] < n:
        raise ValueError(f"cannot remove {n} of tile {tid} from counts")
    counts[tid] -= n


@dataclass(frozen=True, slots=True)
class TileCount:
    tid: int
    n: int
