from __future__ import annotations

from typing import Any, Final

from mahjong_ai.core.actions import ActionKind, N_ACTIONS, decode_action
from mahjong_ai.core.state import GameState, MeldKind, Phase
from mahjong_ai.core.tiles import COPIES_PER_TILE, NUM_SUITS, NUM_TILE_TYPES

try:  # pragma: no cover - optional training dependency
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = e
else:  # pragma: no cover
    _NUMPY_IMPORT_ERROR = None

N_PLAYERS: Final[int] = 4
DINGQUE_STATES: Final[int] = NUM_SUITS + 1  # none + suit id
TILE_OR_NONE: Final[int] = NUM_TILE_TYPES + 1
PLAYER_OR_NONE: Final[int] = N_PLAYERS + 1
TOTAL_TILES: Final[int] = NUM_TILE_TYPES * COPIES_PER_TILE
SCORE_NORM: Final[float] = 1000.0
PHASE_PROGRESS_BUCKETS: Final[int] = 4
RECENT_ACTION_SLOTS: Final[int] = 6
RECENT_ACTION_ACTOR_DIM: Final[int] = N_PLAYERS + 1  # plus padding slot
ACTION_KINDS: Final[tuple[ActionKind, ...]] = tuple(ActionKind)
RECENT_ACTION_KIND_DIM: Final[int] = len(ACTION_KINDS) + 1  # plus padding slot
RECENT_ACTION_KIND_PAD_IDX: Final[int] = len(ACTION_KINDS)
RECENT_ACTION_SLOT_DIM: Final[int] = RECENT_ACTION_ACTOR_DIM + RECENT_ACTION_KIND_DIM + 1

PHASES: Final[tuple[Phase, ...]] = tuple(Phase)
PHASE_TO_INDEX: Final[dict[Phase, int]] = {p: i for i, p in enumerate(PHASES)}
ACTION_KIND_TO_INDEX: Final[dict[ActionKind, int]] = {a: i for i, a in enumerate(ACTION_KINDS)}

_HAND_START = 0
_HAND_END = _HAND_START + NUM_TILE_TYPES
_DISCARDS_START = _HAND_END
_DISCARDS_END = _DISCARDS_START + N_PLAYERS * NUM_TILE_TYPES
_MELDS_START = _DISCARDS_END
_MELDS_END = _MELDS_START + N_PLAYERS * NUM_TILE_TYPES
_DINGQUE_START = _MELDS_END
_DINGQUE_END = _DINGQUE_START + N_PLAYERS * DINGQUE_STATES
_WON_START = _DINGQUE_END
_WON_END = _WON_START + N_PLAYERS
_CURRENT_PLAYER_START = _WON_END
_CURRENT_PLAYER_END = _CURRENT_PLAYER_START + N_PLAYERS
_PID_START = _CURRENT_PLAYER_END
_PID_END = _PID_START + N_PLAYERS
_PHASE_START = _PID_END
_PHASE_END = _PHASE_START + len(PHASES)
_PENDING_DISCARD_TILE_START = _PHASE_END
_PENDING_DISCARD_TILE_END = _PENDING_DISCARD_TILE_START + TILE_OR_NONE
_PENDING_DISCARD_FROM_START = _PENDING_DISCARD_TILE_END
_PENDING_DISCARD_FROM_END = _PENDING_DISCARD_FROM_START + PLAYER_OR_NONE
_PENDING_KONG_TILE_START = _PENDING_DISCARD_FROM_END
_PENDING_KONG_TILE_END = _PENDING_KONG_TILE_START + TILE_OR_NONE
_PENDING_KONG_ACTOR_START = _PENDING_KONG_TILE_END
_PENDING_KONG_ACTOR_END = _PENDING_KONG_ACTOR_START + PLAYER_OR_NONE
_LAST_DRAW_AFTER_KONG_IDX = _PENDING_KONG_ACTOR_END
_LAST_DRAW_WAS_LAST_TILE_IDX = _LAST_DRAW_AFTER_KONG_IDX + 1
_WALL_REMAINING_IDX = _LAST_DRAW_WAS_LAST_TILE_IDX + 1
_SCORES_START = _WALL_REMAINING_IDX + 1
_SCORES_END = _SCORES_START + N_PLAYERS
_TILES_PLAYED_RATIO_IDX = _SCORES_END
_ALIVE_COUNT_IDX = _TILES_PLAYED_RATIO_IDX + 1
_PHASE_PROGRESS_START = _ALIVE_COUNT_IDX + 1
_PHASE_PROGRESS_END = _PHASE_PROGRESS_START + PHASE_PROGRESS_BUCKETS
_DISCARD_COUNT_START = _PHASE_PROGRESS_END
_DISCARD_COUNT_END = _DISCARD_COUNT_START + N_PLAYERS
_MELD_TILE_COUNT_START = _DISCARD_COUNT_END
_MELD_TILE_COUNT_END = _MELD_TILE_COUNT_START + N_PLAYERS
_PUBLIC_TILE_HIST_START = _MELD_TILE_COUNT_END
_PUBLIC_TILE_HIST_END = _PUBLIC_TILE_HIST_START + NUM_TILE_TYPES
_RECENT_ACTION_START = _PUBLIC_TILE_HIST_END
_RECENT_ACTION_END = _RECENT_ACTION_START + RECENT_ACTION_SLOTS * RECENT_ACTION_SLOT_DIM

OBS_VECTOR_DIM: Final[int] = _RECENT_ACTION_END

HAND_SLICE = slice(_HAND_START, _HAND_END)
DISCARDS_SLICE = slice(_DISCARDS_START, _DISCARDS_END)
MELDS_SLICE = slice(_MELDS_START, _MELDS_END)
DINGQUE_SLICE = slice(_DINGQUE_START, _DINGQUE_END)
WON_SLICE = slice(_WON_START, _WON_END)
CURRENT_PLAYER_SLICE = slice(_CURRENT_PLAYER_START, _CURRENT_PLAYER_END)
PID_SLICE = slice(_PID_START, _PID_END)
PHASE_SLICE = slice(_PHASE_START, _PHASE_END)
PENDING_DISCARD_TILE_SLICE = slice(_PENDING_DISCARD_TILE_START, _PENDING_DISCARD_TILE_END)
PENDING_DISCARD_FROM_SLICE = slice(_PENDING_DISCARD_FROM_START, _PENDING_DISCARD_FROM_END)
PENDING_KONG_TILE_SLICE = slice(_PENDING_KONG_TILE_START, _PENDING_KONG_TILE_END)
PENDING_KONG_ACTOR_SLICE = slice(_PENDING_KONG_ACTOR_START, _PENDING_KONG_ACTOR_END)
SCORES_SLICE = slice(_SCORES_START, _SCORES_END)
PHASE_PROGRESS_SLICE = slice(_PHASE_PROGRESS_START, _PHASE_PROGRESS_END)
DISCARD_COUNT_SLICE = slice(_DISCARD_COUNT_START, _DISCARD_COUNT_END)
MELD_TILE_COUNT_SLICE = slice(_MELD_TILE_COUNT_START, _MELD_TILE_COUNT_END)
PUBLIC_TILE_HIST_SLICE = slice(_PUBLIC_TILE_HIST_START, _PUBLIC_TILE_HIST_END)
RECENT_ACTION_SLICE = slice(_RECENT_ACTION_START, _RECENT_ACTION_END)
TILES_PLAYED_RATIO_IDX = _TILES_PLAYED_RATIO_IDX
ALIVE_COUNT_IDX = _ALIVE_COUNT_IDX

HAS_NUMPY: Final[bool] = _NUMPY_IMPORT_ERROR is None


def _require_numpy() -> None:
    if _NUMPY_IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError(
            "NumPy is required for vector observations. Install optional deps: pip install -e \".[rl]\""
        ) from _NUMPY_IMPORT_ERROR


def _one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _phase_progress_bucket(tiles_played_ratio: float) -> int:
    if tiles_played_ratio < 0.25:
        return 0
    if tiles_played_ratio < 0.5:
        return 1
    if tiles_played_ratio < 0.75:
        return 2
    return 3


def _action_id_norm(action_id: int | None) -> float:
    if action_id is None or N_ACTIONS <= 1:
        return 0.0
    clipped = max(0, min(int(action_id), N_ACTIONS - 1))
    return float(clipped) / float(N_ACTIONS - 1)


def _recent_action_slot_features(*, actor: int | None, action_id: int | None) -> list[float]:
    actor_idx = N_PLAYERS
    kind_idx = RECENT_ACTION_KIND_PAD_IDX

    if actor is not None and 0 <= int(actor) < N_PLAYERS:
        actor_idx = int(actor)

    if action_id is not None:
        try:
            da = decode_action(int(action_id))
            kind_idx = ACTION_KIND_TO_INDEX.get(da.kind, RECENT_ACTION_KIND_PAD_IDX)
        except Exception:
            kind_idx = RECENT_ACTION_KIND_PAD_IDX

    out: list[float] = []
    out.extend(_one_hot(actor_idx, RECENT_ACTION_ACTOR_DIM))
    out.extend(_one_hot(kind_idx, RECENT_ACTION_KIND_DIM))
    out.append(_action_id_norm(action_id))
    return out


def phase_to_index(phase: Phase) -> int:
    return PHASE_TO_INDEX[phase]


def encode_observation_vector(state: GameState, pid: int) -> "np.ndarray":
    """Encode an information-set observation into a fixed-length float vector."""

    _require_numpy()

    p = state.players[pid]

    discards_counts: list[list[float]] = [[0.0] * NUM_TILE_TYPES for _ in range(N_PLAYERS)]
    melds_counts: list[list[float]] = [[0.0] * NUM_TILE_TYPES for _ in range(N_PLAYERS)]
    meld_tile_counts: list[float] = [0.0] * N_PLAYERS
    for i in range(N_PLAYERS):
        for tid in state.players[i].discards:
            discards_counts[i][tid] += 1.0
        for m in state.players[i].melds:
            tiles_in_meld = 4.0 if m.kind in (MeldKind.GANG_AN, MeldKind.GANG_BU, MeldKind.GANG_MING) else 3.0
            melds_counts[i][m.tile] += tiles_in_meld
            meld_tile_counts[i] += tiles_in_meld

    pending_discard_tile_index = NUM_TILE_TYPES
    pending_discard_from_index = N_PLAYERS
    if state.pending_discard is not None:
        pending_discard_tile_index = int(state.pending_discard.tile)
        pending_discard_from_index = int(state.pending_discard.from_player)

    pending_kong_tile_index = NUM_TILE_TYPES
    pending_kong_actor_index = N_PLAYERS
    if state.pending_kong is not None:
        pending_kong_tile_index = int(state.pending_kong.tile)
        pending_kong_actor_index = int(state.pending_kong.actor)

    wall_remaining = float(state.wall_remaining())
    wall_remaining_ratio = wall_remaining / float(TOTAL_TILES)
    tiles_played_ratio = max(0.0, min(1.0, 1.0 - wall_remaining_ratio))
    phase_progress_bucket = _phase_progress_bucket(tiles_played_ratio)
    alive_count = sum(1 for sp in state.players if not sp.won)

    public_tile_hist = [0.0] * NUM_TILE_TYPES
    for tid in range(NUM_TILE_TYPES):
        visible = 0.0
        for seat in range(N_PLAYERS):
            visible += discards_counts[seat][tid]
            visible += melds_counts[seat][tid]
        public_tile_hist[tid] = visible / float(COPIES_PER_TILE)

    recent_actions = list(getattr(state, "recent_actions", []))
    recent_window = recent_actions[-RECENT_ACTION_SLOTS:]

    features: list[float] = []

    features.extend(float(v) for v in p.hand)

    for i in range(N_PLAYERS):
        features.extend(discards_counts[i])

    for i in range(N_PLAYERS):
        features.extend(melds_counts[i])

    for i in range(N_PLAYERS):
        dq = state.players[i].dingque_suit
        features.extend(_one_hot(0 if dq is None else int(dq) + 1, DINGQUE_STATES))

    for i in range(N_PLAYERS):
        features.append(1.0 if state.players[i].won else 0.0)

    features.extend(_one_hot(int(state.current_player), N_PLAYERS))
    features.extend(_one_hot(int(pid), N_PLAYERS))
    features.extend(_one_hot(phase_to_index(state.phase), len(PHASES)))

    features.extend(_one_hot(pending_discard_tile_index, TILE_OR_NONE))
    features.extend(_one_hot(pending_discard_from_index, PLAYER_OR_NONE))

    features.extend(_one_hot(pending_kong_tile_index, TILE_OR_NONE))
    features.extend(_one_hot(pending_kong_actor_index, PLAYER_OR_NONE))

    features.append(float(state.last_draw_after_kong))
    features.append(float(state.last_draw_was_last_tile))
    features.append(wall_remaining_ratio)

    features.extend(float(s) / SCORE_NORM for s in state.scores)

    # Extra context for more stable policy learning.
    features.append(tiles_played_ratio)
    features.append(float(alive_count) / float(N_PLAYERS))
    features.extend(_one_hot(phase_progress_bucket, PHASE_PROGRESS_BUCKETS))
    features.extend(float(len(state.players[i].discards)) / float(TOTAL_TILES) for i in range(N_PLAYERS))
    features.extend(float(meld_tile_counts[i]) / float(TOTAL_TILES) for i in range(N_PLAYERS))
    features.extend(public_tile_hist)

    padding_slots = RECENT_ACTION_SLOTS - len(recent_window)
    for _ in range(max(0, padding_slots)):
        features.extend(_recent_action_slot_features(actor=None, action_id=None))

    for trace in recent_window:
        actor = getattr(trace, "actor", None)
        action_id = getattr(trace, "action_id", None)
        features.extend(_recent_action_slot_features(actor=actor, action_id=action_id))

    if len(features) != OBS_VECTOR_DIM:
        raise AssertionError(f"observation vector dim mismatch: got {len(features)}, expect {OBS_VECTOR_DIM}")

    return np.asarray(features, dtype=np.float32)
