from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mahjong_ai.core.actions import ActionKind, decode_action
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.events import Event
from mahjong_ai.core.state import GameState
from mahjong_ai.core.tiles import SUIT_NAMES, pretty_tile
from mahjong_ai.rules.schema import RulesConfig

PolicyActionSelector = Callable[[int, GameState, list[int], str], int]


@dataclass(slots=True)
class ReplayFrame:
    index: int
    required_players: list[int]
    legal_action_ids: dict[int, list[int]]
    chosen_action_ids: dict[int, int]
    invalid_action_ids: dict[int, int]
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    events: list[dict[str, Any]]
    score_delta: list[int]
    done: bool


@dataclass(slots=True)
class ReplayTrace:
    seed: int
    rules_path: str
    generated_at_utc: str
    policy_assignments: dict[int, str]
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    frames: list[ReplayFrame]


@dataclass(slots=True)
class _ReplayEpisodeRef:
    episode_id: str


def _snapshot_player(player: Any) -> dict[str, Any]:
    return {
        "hand": [int(v) for v in player.hand],
        "melds": [
            {
                "kind": str(m.kind.value),
                "tile": int(m.tile),
                "from_player": None if m.from_player is None else int(m.from_player),
            }
            for m in player.melds
        ],
        "discards": [int(t) for t in player.discards],
        "dingque_suit": None if player.dingque_suit is None else int(player.dingque_suit),
        "won": bool(player.won),
    }


def snapshot_state(state: GameState) -> dict[str, Any]:
    pending_discard = None
    if state.pending_discard is not None:
        pending_discard = {
            "from_player": int(state.pending_discard.from_player),
            "tile": int(state.pending_discard.tile),
            "from_last_tile_draw": bool(state.pending_discard.from_last_tile_draw),
        }

    pending_kong = None
    if state.pending_kong is not None:
        pending_kong = {
            "actor": int(state.pending_kong.actor),
            "tile": int(state.pending_kong.tile),
            "meld_index": int(state.pending_kong.meld_index),
        }

    return {
        "dealer": int(state.dealer),
        "current_player": int(state.current_player),
        "phase": str(state.phase.value),
        "wall_remaining": int(state.wall_remaining()),
        "scores": [int(v) for v in state.scores],
        "pending_discard": pending_discard,
        "pending_kong": pending_kong,
        "players": [_snapshot_player(player) for player in state.players],
    }


def _normalize_event(event: Event) -> dict[str, Any]:
    return {
        "type": str(event.type),
        "actor": None if event.actor is None else int(event.actor),
        "tile": None if event.tile is None else int(event.tile),
        "meta": copy.deepcopy(event.meta),
    }


def _format_scores(scores: list[int]) -> str:
    return " ".join(f"P{pid}={int(score)}" for pid, score in enumerate(scores))


def _format_tile_sequence(tiles: list[int]) -> str:
    if not tiles:
        return "-"
    return " ".join(pretty_tile(int(tile)) for tile in tiles)


def _format_hand_counts(counts: list[int]) -> str:
    parts: list[str] = []
    for tid, count in enumerate(counts):
        if int(count) <= 0:
            continue
        token = pretty_tile(int(tid))
        if int(count) > 1:
            token = f"{token}x{int(count)}"
        parts.append(token)
    return " ".join(parts) if parts else "-"


def _format_melds(melds: list[dict[str, Any]]) -> str:
    if not melds:
        return "-"
    parts: list[str] = []
    for meld in melds:
        token = f"{meld['kind']}({pretty_tile(int(meld['tile']))}"
        from_player = meld.get("from_player")
        if from_player is not None:
            token += f",from=P{int(from_player)}"
        token += ")"
        parts.append(token)
    return " ".join(parts)


def _format_suit(suit: int | None) -> str:
    if suit is None:
        return "-"
    return SUIT_NAMES[int(suit)]


def _format_action(action_id: int) -> str:
    decoded = decode_action(int(action_id))
    if decoded.kind == ActionKind.DINGQUE and decoded.arg is not None:
        return f"{decoded.kind.value}({_format_suit(int(decoded.arg))})"
    if decoded.arg is not None:
        return f"{decoded.kind.value}({pretty_tile(int(decoded.arg))})"
    return decoded.kind.value


def _extract_legal_action_ids(mask: list[int]) -> list[int]:
    return [int(i) for i, allowed in enumerate(mask) if int(allowed) == 1]


def _round_end_reason(trace: ReplayTrace) -> str:
    for frame in reversed(trace.frames):
        for event in reversed(frame.events):
            if event["type"] == "round_end":
                meta = event.get("meta") or {}
                reason = meta.get("reason")
                if reason:
                    return str(reason)
    return "-"


def build_self_play_replay_trace(
    *,
    rules: RulesConfig,
    seed: int,
    policy_assignments: dict[int, str],
    action_selector: PolicyActionSelector,
    max_steps: int,
    strict_illegal_action: bool,
    rules_path: str = "",
) -> ReplayTrace:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    replay_engine = GameEngine(rules=copy.deepcopy(rules), enable_events=False)
    state = replay_engine.reset(seed=int(seed))
    initial_state = snapshot_state(state)
    frames: list[ReplayFrame] = []
    steps = 0

    while True:
        if state.phase.value == "round_end":
            break
        if steps >= int(max_steps):
            raise RuntimeError("self-play replay exceeded step limit; possible engine bug")

        pre_state = snapshot_state(state)
        required_players = replay_engine.required_players(state)
        legal_action_ids: dict[int, list[int]] = {}
        chosen_action_ids: dict[int, int] = {}
        invalid_action_ids: dict[int, int] = {}

        for pid in required_players:
            mask = replay_engine.legal_action_mask(state, pid)
            legal_action_ids[pid] = _extract_legal_action_ids(mask)
            policy_id = str(policy_assignments.get(pid, "shared_policy"))
            chosen = int(action_selector(pid, state, mask, policy_id))
            if chosen < 0 or chosen >= len(mask) or mask[chosen] != 1:
                invalid_action_ids[pid] = int(chosen)
                if strict_illegal_action:
                    raise RuntimeError(
                        f"self-play replay produced illegal action {chosen} for pid={pid} in phase={state.phase.value}"
                    )
                if not legal_action_ids[pid]:
                    raise RuntimeError(f"action mask has no legal action for pid={pid} in phase={state.phase.value}")
                chosen = int(legal_action_ids[pid][0])
            chosen_action_ids[pid] = int(chosen)

        result = replay_engine.step(state, chosen_action_ids)
        steps += 1
        frames.append(
            ReplayFrame(
                index=steps,
                required_players=[int(pid) for pid in required_players],
                legal_action_ids=legal_action_ids,
                chosen_action_ids=chosen_action_ids,
                invalid_action_ids=invalid_action_ids,
                pre_state=pre_state,
                post_state=snapshot_state(state),
                events=[_normalize_event(event) for event in result.events],
                score_delta=[int(v) for v in result.score_delta],
                done=bool(result.done),
            )
        )
        if result.done:
            break

    return ReplayTrace(
        seed=int(seed),
        rules_path=str(rules_path or ""),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        policy_assignments={int(pid): str(policy_id) for pid, policy_id in policy_assignments.items()},
        initial_state=initial_state,
        final_state=snapshot_state(state),
        frames=frames,
    )


def _render_pending(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    pending_discard = snapshot.get("pending_discard")
    if pending_discard is None:
        lines.append("pending_discard=-")
    else:
        lines.append(
            "pending_discard="
            f"tile={pretty_tile(int(pending_discard['tile']))} from=P{int(pending_discard['from_player'])} "
            f"from_last_tile_draw={bool(pending_discard['from_last_tile_draw'])}"
        )

    pending_kong = snapshot.get("pending_kong")
    if pending_kong is None:
        lines.append("pending_kong=-")
    else:
        lines.append(
            "pending_kong="
            f"tile={pretty_tile(int(pending_kong['tile']))} actor=P{int(pending_kong['actor'])} "
            f"meld_index={int(pending_kong['meld_index'])}"
        )
    return lines


def _render_players(snapshot: dict[str, Any], *, omniscient: bool, seat: int | None) -> list[str]:
    lines: list[str] = []
    for pid, player in enumerate(snapshot["players"]):
        if omniscient or pid == seat:
            hand_repr = _format_hand_counts(player["hand"])
        else:
            hand_repr = "<hidden>"
        lines.append(
            f"P{pid} hand={hand_repr} dingque={_format_suit(player['dingque_suit'])} "
            f"melds={_format_melds(player['melds'])} discards={_format_tile_sequence(player['discards'])} "
            f"won={bool(player['won'])}"
        )
    return lines


def _render_legal_actions(action_ids: list[int], *, omniscient: bool, pid: int, seat: int | None) -> str:
    if omniscient or pid == seat:
        return ", ".join(_format_action(action_id) for action_id in action_ids) if action_ids else "-"
    return "<hidden>"


def _render_action_for_view(action_id: int, *, phase: str, omniscient: bool, pid: int, seat: int | None) -> str:
    if omniscient or pid == seat:
        return _format_action(action_id)
    if phase in {"swap_pick_1", "swap_pick_2", "swap_pick_3"}:
        return "<hidden_swap_pick>"
    return _format_action(action_id)


def _render_event(event: dict[str, Any], *, omniscient: bool, seat: int | None) -> str:
    parts = [str(event["type"])]
    actor = event.get("actor")
    if actor is not None:
        parts.append(f"actor=P{int(actor)}")

    tile = event.get("tile")
    if tile is not None:
        if not omniscient and actor is not None and int(actor) != int(seat):
            if event["type"] == "swap_pick":
                parts.append("tile=<hidden_swap_pick>")
            elif event["type"] == "draw":
                parts.append("tile=<hidden_draw>")
            else:
                parts.append(f"tile={pretty_tile(int(tile))}")
        else:
            parts.append(f"tile={pretty_tile(int(tile))}")

    meta = event.get("meta") or {}
    if meta:
        parts.append(f"meta={meta}")
    return " ".join(parts)


def render_replay_text(trace: ReplayTrace, *, view: str, seat: int | None = None) -> str:
    if view not in {"omniscient", "seat"}:
        raise ValueError("view must be 'omniscient' or 'seat'")
    if view == "seat" and seat is None:
        raise ValueError("seat view requires a seat id")

    omniscient = view == "omniscient"
    view_label = "omniscient" if omniscient else f"seat{int(seat)}"

    lines = [
        f"self_play_replay view={view_label}",
        f"seed={trace.seed}",
        f"generated_at_utc={trace.generated_at_utc}",
        f"rules_path={trace.rules_path or '-'}",
        "policy_assignments=" + " ".join(
            f"P{pid}={trace.policy_assignments.get(pid, '-')}" for pid in range(4)
        ),
        "",
        "[initial_state]",
    ]

    initial = trace.initial_state
    lines.append(
        f"dealer=P{int(initial['dealer'])} current_player=P{int(initial['current_player'])} "
        f"phase={initial['phase']} wall_remaining={int(initial['wall_remaining'])}"
    )
    lines.append(f"scores={_format_scores(initial['scores'])}")
    lines.extend(_render_pending(initial))
    lines.extend(_render_players(initial, omniscient=omniscient, seat=seat))

    for frame in trace.frames:
        lines.append("")
        lines.append(f"[step {frame.index}]")
        lines.append(
            f"phase={frame.pre_state['phase']} current_player=P{int(frame.pre_state['current_player'])} "
            f"required={frame.required_players} wall_remaining={int(frame.pre_state['wall_remaining'])}"
        )
        lines.append(f"scores_before={_format_scores(frame.pre_state['scores'])}")
        lines.extend(_render_pending(frame.pre_state))
        lines.extend(_render_players(frame.pre_state, omniscient=omniscient, seat=seat))

        for pid in frame.required_players:
            legal = _render_legal_actions(frame.legal_action_ids.get(pid, []), omniscient=omniscient, pid=pid, seat=seat)
            action = _render_action_for_view(
                frame.chosen_action_ids[pid],
                phase=str(frame.pre_state['phase']),
                omniscient=omniscient,
                pid=pid,
                seat=seat,
            )
            line = f"P{pid} policy={trace.policy_assignments.get(pid, '-')} legal=[{legal}] action={action}"
            if pid in frame.invalid_action_ids:
                line += f" invalid_selected={frame.invalid_action_ids[pid]}"
            lines.append(line)

        if frame.events:
            lines.append("events:")
            for event in frame.events:
                lines.append(f"- {_render_event(event, omniscient=omniscient, seat=seat)}")
        else:
            lines.append("events: <none>")

        lines.append(f"score_delta={_format_scores(frame.score_delta)}")
        lines.append(f"scores_after={_format_scores(frame.post_state['scores'])}")
        lines.append(f"wall_remaining_after={int(frame.post_state['wall_remaining'])}")

    lines.append("")
    lines.append("[final_state]")
    final = trace.final_state
    lines.append(
        f"dealer=P{int(final['dealer'])} current_player=P{int(final['current_player'])} "
        f"phase={final['phase']} wall_remaining={int(final['wall_remaining'])}"
    )
    lines.append(f"scores={_format_scores(final['scores'])}")
    lines.append(f"round_end_reason={_round_end_reason(trace)}")
    lines.extend(_render_pending(final))
    lines.extend(_render_players(final, omniscient=omniscient, seat=seat))
    return "\n".join(lines) + "\n"


def write_replay_artifacts(
    *,
    output_dir: Path,
    trace: ReplayTrace,
    include_omniscient: bool,
    seat_views: list[int],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if include_omniscient:
        path = output_dir / f"seed_{trace.seed}_omniscient.txt"
        path.write_text(render_replay_text(trace, view="omniscient"), encoding="utf-8")
        written.append(path)

    for seat in seat_views:
        path = output_dir / f"seed_{trace.seed}_seat{int(seat)}.txt"
        path.write_text(render_replay_text(trace, view="seat", seat=int(seat)), encoding="utf-8")
        written.append(path)

    return written


__all__ = [
    "PolicyActionSelector",
    "ReplayFrame",
    "ReplayTrace",
    "_ReplayEpisodeRef",
    "build_self_play_replay_trace",
    "render_replay_text",
    "snapshot_state",
    "write_replay_artifacts",
]
