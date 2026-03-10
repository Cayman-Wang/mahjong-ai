from __future__ import annotations

import random

from mahjong_ai.agents.base import Agent, AgentContext
from mahjong_ai.core.actions import ActionKind, decode_action, encode_action
from mahjong_ai.core.state import GameState, Phase
from mahjong_ai.core.tiles import NUM_TILE_TYPES, RANKS_PER_SUIT, tile_rank, tile_suit


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, *, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        # Always take HU if available.
        hu_id = encode_action(ActionKind.HU)
        if action_mask[hu_id] == 1:
            return hu_id

        if state.phase in (Phase.SWAP_PICK_1, Phase.SWAP_PICK_2, Phase.SWAP_PICK_3):
            return self._act_swap(ctx, state, action_mask)

        if state.phase == Phase.DINGQUE:
            return self._act_dingque(ctx, state, action_mask)

        if state.phase == Phase.RESPONSE:
            # Prefer exposed kong, then peng, otherwise pass.
            for kind in (ActionKind.GANG_MING, ActionKind.PENG):
                for a, m in enumerate(action_mask):
                    if m != 1:
                        continue
                    da = decode_action(a)
                    if da.kind == kind:
                        return a
            return encode_action(ActionKind.PASS)

        if state.phase == Phase.RESPONSE_QIANGGANG:
            return encode_action(ActionKind.PASS)

        if state.phase == Phase.TURN_ACTION:
            # Prefer declaring a kong before discarding.
            for kind in (ActionKind.GANG_AN, ActionKind.GANG_BU):
                for a, m in enumerate(action_mask):
                    if m != 1:
                        continue
                    da = decode_action(a)
                    if da.kind == kind:
                        return a

            return self._act_discard(ctx, state, action_mask)

        # Fallback to first legal.
        for a, m in enumerate(action_mask):
            if m == 1:
                return a
        raise RuntimeError(f"no legal actions for player {ctx.seat} in phase {state.phase}")

    def _act_dingque(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        p = state.players[ctx.seat]
        suit_counts = [0, 0, 0]
        for tid, n in enumerate(p.hand):
            if n:
                suit_counts[tile_suit(tid)] += n
        # Choose the suit with the fewest tiles.
        best = min(range(3), key=lambda s: (suit_counts[s], s))
        a = encode_action(ActionKind.DINGQUE, best)
        if action_mask[a] == 1:
            return a
        # Shouldn't happen, but fall back.
        legal = [i for i, m in enumerate(action_mask) if m == 1]
        return self._rng.choice(legal)

    def _act_swap(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        # Simple strategy: guess the future dingque as the least-populated suit and swap those tiles out.
        p = state.players[ctx.seat]
        suit_counts = [0, 0, 0]
        tiles_by_suit: list[list[int]] = [[], [], []]
        for tid, n in enumerate(p.hand):
            if n <= 0:
                continue
            s = tile_suit(tid)
            suit_counts[s] += n
            tiles_by_suit[s].extend([tid] * n)

        dingque_guess = min(range(3), key=lambda s: (suit_counts[s], s))
        candidates = tiles_by_suit[dingque_guess]
        self._rng.shuffle(candidates)
        for tid in candidates:
            a = encode_action(ActionKind.SWAP_PICK, tid)
            if action_mask[a] == 1:
                return a

        legal = [i for i, m in enumerate(action_mask) if m == 1]
        return self._rng.choice(legal)

    def _act_discard(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        p = state.players[ctx.seat]

        legal_discards: list[int] = []
        for a, m in enumerate(action_mask):
            if m != 1:
                continue
            da = decode_action(a)
            if da.kind == ActionKind.DISCARD:
                legal_discards.append(a)

        if not legal_discards:
            # Shouldn't happen; fall back.
            legal = [i for i, m in enumerate(action_mask) if m == 1]
            return self._rng.choice(legal)

        # If we still have dingque tiles, discard the most frequent one.
        if p.dingque_suit is not None and p.has_dingque_tiles():
            best_a = None
            best_n = -1
            for a in legal_discards:
                tid = decode_action(a).arg  # type: ignore[assignment]
                assert tid is not None
                n = p.hand[tid]
                if n > best_n:
                    best_n = n
                    best_a = a
            assert best_a is not None
            return best_a

        # Otherwise, discard the least \"useful\" tile (rough isolation heuristic).
        def usefulness(tid: int) -> float:
            c = p.hand[tid]
            if c <= 0:
                return 1e9
            s = tile_suit(tid)
            r0 = tile_rank(tid) - 1  # 0..8
            base = s * RANKS_PER_SUIT

            neighbors = 0
            if r0 - 1 >= 0:
                neighbors += p.hand[base + (r0 - 1)]
            if r0 + 1 < RANKS_PER_SUIT:
                neighbors += p.hand[base + (r0 + 1)]

            # Prefer keeping pairs/triplets and tiles with neighbors.
            return 2.0 * (c - 1) + 1.0 * neighbors

        best = None
        best_u = 1e9
        for a in legal_discards:
            tid = decode_action(a).arg
            assert tid is not None
            u = usefulness(tid)
            if u < best_u:
                best_u = u
                best = a

        if best is None:
            return self._rng.choice(legal_discards)
        return best

