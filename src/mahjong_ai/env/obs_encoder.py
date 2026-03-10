from __future__ import annotations

from mahjong_ai.core.state import GameState, MeldKind
from mahjong_ai.env.obs_vector_encoder import phase_to_index
from mahjong_ai.core.tiles import NUM_TILE_TYPES


def encode_observation(state: GameState, pid: int) -> dict[str, object]:
    """Encode an information-set observation for `pid`.

    This encoder only uses public information + the player's own concealed hand.
    It intentionally does not leak other players' concealed tiles or wall order.
    """

    p = state.players[pid]

    discards_counts: list[list[int]] = [[0] * NUM_TILE_TYPES for _ in range(4)]
    melds_counts: list[list[int]] = [[0] * NUM_TILE_TYPES for _ in range(4)]
    for i in range(4):
        for tid in state.players[i].discards:
            discards_counts[i][tid] += 1
        for m in state.players[i].melds:
            melds_counts[i][m.tile] += 4 if m.kind in (MeldKind.GANG_AN, MeldKind.GANG_BU, MeldKind.GANG_MING) else 3

    dingque = [(-1 if state.players[i].dingque_suit is None else int(state.players[i].dingque_suit)) for i in range(4)]
    won = [1 if state.players[i].won else 0 for i in range(4)]

    pending_tile = -1
    pending_from = -1
    if state.pending_discard is not None:
        pending_tile = int(state.pending_discard.tile)
        pending_from = int(state.pending_discard.from_player)

    pending_kong_tile = -1
    pending_kong_actor = -1
    if state.pending_kong is not None:
        pending_kong_tile = int(state.pending_kong.tile)
        pending_kong_actor = int(state.pending_kong.actor)

    obs: dict[str, object] = {
        "pid": pid,
        "phase": phase_to_index(state.phase),
        "current_player": state.current_player,
        "wall_remaining": state.wall_remaining(),
        "hand": p.hand.copy(),
        "discards": discards_counts,
        "melds": melds_counts,
        "dingque": dingque,
        "won": won,
        "pending_discard_tile": pending_tile,
        "pending_discard_from": pending_from,
        "pending_kong_tile": pending_kong_tile,
        "pending_kong_actor": pending_kong_actor,
        "last_draw_after_kong": int(state.last_draw_after_kong),
        "last_draw_was_last_tile": int(state.last_draw_was_last_tile),
        "scores": state.scores.copy(),
    }
    return obs
