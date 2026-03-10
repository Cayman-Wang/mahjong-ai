from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tiles import NUM_TILE_TYPES, tile_suit


class MeldKind(str, Enum):
    PENG = "peng"
    GANG_MING = "gang_ming"  # exposed kong from discard
    GANG_AN = "gang_an"      # concealed kong
    GANG_BU = "gang_bu"      # supplemental kong (upgrade from peng)


@dataclass(slots=True)
class Meld:
    kind: MeldKind
    tile: int
    from_player: int | None = None  # for exposed melds from a discard

    @property
    def size(self) -> int:
        return 4 if self.kind in (MeldKind.GANG_MING, MeldKind.GANG_AN, MeldKind.GANG_BU) else 3


@dataclass(slots=True)
class PlayerState:
    hand: list[int] = field(default_factory=lambda: [0] * NUM_TILE_TYPES)  # concealed counts
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    dingque_suit: int | None = None
    won: bool = False

    def has_dingque_tiles(self) -> bool:
        """Return True if concealed hand still contains the dingque suit.

        Note: this intentionally checks *hand only* (not melds). It is used for
        "must discard dingque first" constraints; melds cannot be discarded.
        """
        if self.dingque_suit is None:
            return False
        s = self.dingque_suit
        for tid, n in enumerate(self.hand):
            if n and tile_suit(tid) == s:
                return True
        return False

    def has_dingque_tiles_in_melds(self) -> bool:
        """Return True if any open meld contains the dingque suit (illegal in most rulesets)."""
        if self.dingque_suit is None:
            return False
        s = self.dingque_suit
        return any(tile_suit(m.tile) == s for m in self.melds)

    def has_dingque_tiles_anywhere(self) -> bool:
        """Return True if the player has dingque-suit tiles in hand or melds."""
        return self.has_dingque_tiles() or self.has_dingque_tiles_in_melds()

    def meld_count(self) -> int:
        return len(self.melds)


class Phase(str, Enum):
    SWAP_PICK_1 = "swap_pick_1"
    SWAP_PICK_2 = "swap_pick_2"
    SWAP_PICK_3 = "swap_pick_3"
    SWAP_RESOLVE = "swap_resolve"
    DINGQUE = "dingque"

    TURN_DRAW = "turn_draw"           # internal auto phase
    TURN_ACTION = "turn_action"       # current player chooses (hu/gang/discard)

    RESPONSE = "response"             # discard response (hu/gang/peng/pass)
    RESPONSE_QIANGGANG = "response_qianggang"  # rob a supplemental kong (hu/pass)

    ROUND_END = "round_end"


@dataclass(slots=True)
class PendingDiscard:
    from_player: int
    tile: int
    from_last_tile_draw: bool = False


@dataclass(slots=True)
class PendingKong:
    actor: int
    tile: int
    meld_index: int


@dataclass(slots=True)
class ActionTrace:
    actor: int
    action_id: int


RECENT_ACTION_HISTORY_LIMIT = 16


@dataclass(slots=True)
class GameState:
    seed: int
    rng_state: object

    wall: list[int]
    wall_pos: int
    wall_end: int

    dealer: int
    current_player: int
    phase: Phase

    players: list[PlayerState]
    scores: list[int]

    swap_picks: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    pending_discard: PendingDiscard | None = None
    pending_kong: PendingKong | None = None

    # Context flags for the current player (set on draw, cleared on turn end).
    last_draw_after_kong: bool = False
    last_draw_was_last_tile: bool = False
    # True only if the current player has drawn a tile for the current TURN_ACTION.
    # This distinguishes normal self-draw turns from "claim turns" (e.g., after PENG),
    # where HU must not be treated as a self-draw win.
    turn_has_drawn: bool = False

    # Debug / training trace
    events: list[object] = field(default_factory=list)
    # Recent player decisions (public actions only), newest at the tail.
    recent_actions: list[ActionTrace] = field(default_factory=list)

    def alive_players(self) -> list[int]:
        return [pid for pid, p in enumerate(self.players) if not p.won]

    def num_won(self) -> int:
        return sum(1 for p in self.players if p.won)

    def wall_remaining(self) -> int:
        return max(0, self.wall_end - self.wall_pos)
