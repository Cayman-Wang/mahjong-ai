from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.core.actions import (
    ActionKind,
    N_ACTIONS,
    decode_action,
    encode_action,
)
from mahjong_ai.core.events import Event
from mahjong_ai.core.rng import RNG
from mahjong_ai.core.state import (
    ActionTrace,
    GameState,
    Meld,
    MeldKind,
    PendingDiscard,
    PendingKong,
    Phase,
    PlayerState,
    RECENT_ACTION_HISTORY_LIMIT,
)
from mahjong_ai.core.tiles import (
    NUM_TILE_TYPES,
    all_tiles,
    counts_add,
    counts_remove,
    tile_suit,
)
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.scoring.fan_patterns import WinContext
from mahjong_ai.scoring.settlement import (
    compute_fan,
    settle_gang,
    settle_hu,
    settle_hua_zhu_and_cha_jiao,
)
from mahjong_ai.scoring.win_check import detect_win


@dataclass(slots=True)
class StepResult:
    score_delta: list[int]
    events: list[Event]
    done: bool
    reason: str | None = None


class GameEngine:
    """Sichuan Mahjong (Xuezhan Daodi) engine.

    Design goals:
    - Deterministic given seed + action sequence
    - Explicit simultaneous decision phases (swap/dingque/response)
    - Core engine has no third-party dependencies
    """

    def __init__(self, rules: RulesConfig | None = None, *, enable_events: bool = True):
        self.rules = rules or RulesConfig()
        self.rules.validate()
        if self.rules.allow_chi:
            raise NotImplementedError("allow_chi=True is not supported yet (CHI is not implemented).")
        self.enable_events = enable_events

    @staticmethod
    def _pick_closest_to_player(center_pid: int, candidates: list[int]) -> int | None:
        """Pick the closest candidate in clockwise order from `center_pid`."""
        if not candidates:
            return None
        for i in range(1, 5):
            pid = (center_pid + i) % 4
            if pid in candidates:
                return pid
        return candidates[0]

    @staticmethod
    def _record_recent_action(state: GameState, *, pid: int, action_id: int) -> None:
        state.recent_actions.append(ActionTrace(actor=int(pid), action_id=int(action_id)))
        if len(state.recent_actions) > RECENT_ACTION_HISTORY_LIMIT:
            del state.recent_actions[:-RECENT_ACTION_HISTORY_LIMIT]

    @staticmethod
    def _consume_last_discard_for_claim(state: GameState, *, discarder: int, tile: int) -> None:
        discards = state.players[discarder].discards
        if not discards:
            raise ValueError("cannot claim discard: discarder has no public discards")
        if discards[-1] != tile:
            raise ValueError(
                f"cannot claim discard tile {tile}: top discard is {discards[-1]} for player {discarder}"
            )
        discards.pop()

    def reset(self, *, seed: int) -> GameState:
        rng = RNG.from_seed(seed)

        wall = all_tiles()
        rng.shuffle(wall)

        dealer = rng.randint(0, 3)

        players = [PlayerState() for _ in range(4)]
        wall_pos = 0
        wall_end = len(wall)

        # Deal 13 tiles to each player.
        for _ in range(13):
            for pid in range(4):
                tid = wall[wall_pos]
                wall_pos += 1
                players[pid].hand[tid] += 1

        phase = Phase.SWAP_PICK_1 if self.rules.swap_enabled else Phase.DINGQUE
        if not self.rules.dingque_enabled:
            phase = Phase.TURN_DRAW

        state = GameState(
            seed=seed,
            rng_state=rng.getstate(),
            wall=wall,
            wall_pos=wall_pos,
            wall_end=wall_end,
            dealer=dealer,
            current_player=dealer,
            phase=phase,
            players=players,
            scores=[0, 0, 0, 0],
        )

        events: list[Event] = [Event("reset", meta={"dealer": dealer})]

        if phase == Phase.TURN_DRAW:
            # Auto-draw to reach the first decision.
            d = self._auto_advance(state, events, rng=rng)
            for i in range(4):
                state.scores[i] += d[i]

        if self.enable_events:
            state.events.extend(events)

        return state

    def required_players(self, state: GameState) -> list[int]:
        alive = [pid for pid in range(4) if not state.players[pid].won]

        if state.phase in (Phase.SWAP_PICK_1, Phase.SWAP_PICK_2, Phase.SWAP_PICK_3):
            return alive
        if state.phase == Phase.DINGQUE:
            return [pid for pid in alive if state.players[pid].dingque_suit is None]
        if state.phase == Phase.TURN_ACTION:
            pid = state.current_player
            return [pid] if (pid in alive) else []
        if state.phase == Phase.RESPONSE:
            if state.pending_discard is None:
                return []
            discarder = state.pending_discard.from_player
            return [pid for pid in alive if pid != discarder]
        if state.phase == Phase.RESPONSE_QIANGGANG:
            if state.pending_kong is None:
                return []
            actor = state.pending_kong.actor
            return [pid for pid in alive if pid != actor]
        return []

    def legal_action_mask(self, state: GameState, pid: int) -> list[int]:
        mask = [0] * N_ACTIONS
        if pid not in self.required_players(state):
            return mask

        p = state.players[pid]

        if state.phase in (Phase.SWAP_PICK_1, Phase.SWAP_PICK_2, Phase.SWAP_PICK_3):
            for tid in range(NUM_TILE_TYPES):
                if p.hand[tid] > 0:
                    mask[encode_action(ActionKind.SWAP_PICK, tid)] = 1
            return mask

        if state.phase == Phase.DINGQUE:
            for suit in range(3):
                mask[encode_action(ActionKind.DINGQUE, suit)] = 1
            return mask

        if state.phase == Phase.TURN_ACTION:
            # HU (self-draw). Only legal on a turn that actually drew a tile.
            if state.turn_has_drawn:
                win = detect_win(
                    p.hand,
                    meld_count=p.meld_count(),
                    dingque_suit=p.dingque_suit,
                    dingque_in_melds=p.has_dingque_tiles_in_melds(),
                )
                if win.ok:
                    ctx = WinContext(
                        winner=pid,
                        winning_tile=-1,
                        self_draw=True,
                        from_player=None,
                        after_kong=state.last_draw_after_kong,
                        rob_kong=False,
                        last_tile_draw=state.last_draw_was_last_tile,
                        last_tile_discard=False,
                    )
                    fan = compute_fan(counts=p.hand, player=p, ctx=ctx, rules=self.rules)
                    if self.rules.allow_zero_fan or fan.fan_total > 0:
                        mask[encode_action(ActionKind.HU)] = 1

            # GANG_AN
            for tid in range(NUM_TILE_TYPES):
                if p.hand[tid] >= 4:
                    mask[encode_action(ActionKind.GANG_AN, tid)] = 1

            # GANG_BU
            peng_tiles = {m.tile for m in p.melds if m.kind == MeldKind.PENG}
            for tid in peng_tiles:
                if p.hand[tid] >= 1:
                    if p.dingque_suit is not None and tile_suit(tid) == p.dingque_suit:
                        continue
                    mask[encode_action(ActionKind.GANG_BU, tid)] = 1

            # DISCARD
            must_suit = p.dingque_suit if p.has_dingque_tiles() else None
            for tid in range(NUM_TILE_TYPES):
                if p.hand[tid] <= 0:
                    continue
                if must_suit is not None and tile_suit(tid) != must_suit:
                    continue
                mask[encode_action(ActionKind.DISCARD, tid)] = 1

            return mask

        if state.phase == Phase.RESPONSE:
            assert state.pending_discard is not None
            disc = state.pending_discard
            t = disc.tile

            mask[encode_action(ActionKind.PASS)] = 1

            # HU on discard.
            tmp = p.hand.copy()
            tmp[t] += 1
            win = detect_win(
                tmp,
                meld_count=p.meld_count(),
                dingque_suit=p.dingque_suit,
                dingque_in_melds=p.has_dingque_tiles_in_melds(),
            )
            if win.ok:
                ctx = WinContext(
                    winner=pid,
                    winning_tile=t,
                    self_draw=False,
                    from_player=disc.from_player,
                    after_kong=False,
                    rob_kong=False,
                    last_tile_draw=False,
                    last_tile_discard=disc.from_last_tile_draw,
                )
                fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
                if self.rules.allow_zero_fan or fan.fan_total > 0:
                    mask[encode_action(ActionKind.HU)] = 1

            # GANG_MING
            if p.dingque_suit is None or tile_suit(t) != p.dingque_suit:
                if p.hand[t] >= 3:
                    mask[encode_action(ActionKind.GANG_MING, t)] = 1

            # PENG
            if p.dingque_suit is None or tile_suit(t) != p.dingque_suit:
                if p.hand[t] >= 2:
                    mask[encode_action(ActionKind.PENG, t)] = 1

            return mask

        if state.phase == Phase.RESPONSE_QIANGGANG:
            assert state.pending_kong is not None
            t = state.pending_kong.tile

            mask[encode_action(ActionKind.PASS)] = 1

            tmp = p.hand.copy()
            tmp[t] += 1
            win = detect_win(
                tmp,
                meld_count=p.meld_count(),
                dingque_suit=p.dingque_suit,
                dingque_in_melds=p.has_dingque_tiles_in_melds(),
            )
            if win.ok:
                ctx = WinContext(
                    winner=pid,
                    winning_tile=t,
                    self_draw=False,
                    from_player=state.pending_kong.actor,
                    after_kong=False,
                    rob_kong=True,
                    last_tile_draw=False,
                    last_tile_discard=False,
                )
                fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
                if self.rules.allow_zero_fan or fan.fan_total > 0:
                    mask[encode_action(ActionKind.HU)] = 1

            return mask

        return mask

    def step(self, state: GameState, actions: dict[int, int]) -> StepResult:
        rng = RNG.from_seed(state.seed)
        rng.setstate(state.rng_state)

        events: list[Event] = []
        score_delta = [0, 0, 0, 0]

        required = self.required_players(state)
        if set(actions.keys()) != set(required):
            raise ValueError(
                f"actions must be provided for required players {required}, got {sorted(actions.keys())}"
            )

        def add_delta(d: list[int]) -> None:
            for i in range(4):
                score_delta[i] += d[i]
                state.scores[i] += d[i]

        if state.phase in (Phase.SWAP_PICK_1, Phase.SWAP_PICK_2, Phase.SWAP_PICK_3):
            for pid in required:
                da = decode_action(actions[pid])
                if da.kind != ActionKind.SWAP_PICK or da.arg is None:
                    raise ValueError("swap phase requires SWAP_PICK(tile)")
                tid = da.arg
                if state.players[pid].hand[tid] <= 0:
                    raise ValueError("cannot swap-pick a tile not in hand")
                # Swap picks are private hidden information; do not leak them into
                # the shared recent_actions history used by vector observations.
                counts_remove(state.players[pid].hand, tid, 1)
                state.swap_picks[pid].append(tid)
                events.append(Event("swap_pick", actor=pid, tile=tid))

            if state.phase == Phase.SWAP_PICK_1:
                state.phase = Phase.SWAP_PICK_2
            elif state.phase == Phase.SWAP_PICK_2:
                state.phase = Phase.SWAP_PICK_3
            else:
                state.phase = Phase.SWAP_RESOLVE
                add_delta(self._resolve_swap(state, rng, events))

            state.rng_state = rng.getstate()
            if self.enable_events:
                state.events.extend(events)
            return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

        if state.phase == Phase.DINGQUE:
            for pid in required:
                da = decode_action(actions[pid])
                if da.kind != ActionKind.DINGQUE or da.arg is None:
                    raise ValueError("dingque phase requires DINGQUE(suit)")
                suit = da.arg
                self._record_recent_action(state, pid=pid, action_id=actions[pid])
                state.players[pid].dingque_suit = suit
                events.append(Event("dingque", actor=pid, meta={"suit": suit}))

            state.phase = Phase.TURN_DRAW
            state.turn_has_drawn = False
            add_delta(self._auto_advance(state, events, rng=rng))

            state.rng_state = rng.getstate()
            if self.enable_events:
                state.events.extend(events)
            return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

        if state.phase == Phase.TURN_ACTION:
            pid = required[0]
            da = decode_action(actions[pid])

            if da.kind == ActionKind.HU:
                if not state.turn_has_drawn:
                    raise ValueError("illegal HU (no draw on this turn)")
                win = detect_win(
                    state.players[pid].hand,
                    meld_count=state.players[pid].meld_count(),
                    dingque_suit=state.players[pid].dingque_suit,
                    dingque_in_melds=state.players[pid].has_dingque_tiles_in_melds(),
                )
                if not win.ok:
                    raise ValueError("illegal HU")

                ctx = WinContext(
                    winner=pid,
                    winning_tile=-1,
                    self_draw=True,
                    from_player=None,
                    after_kong=state.last_draw_after_kong,
                    rob_kong=False,
                    last_tile_draw=state.last_draw_was_last_tile,
                    last_tile_discard=False,
                )
                fan = compute_fan(counts=state.players[pid].hand, player=state.players[pid], ctx=ctx, rules=self.rules)
                if (not self.rules.allow_zero_fan) and fan.fan_total <= 0:
                    raise ValueError("HU requires >=1 fan in this ruleset")

                self._record_recent_action(state, pid=pid, action_id=actions[pid])
                alive = [i for i in range(4) if not state.players[i].won]
                add_delta(
                    settle_hu(
                        winner=pid,
                        from_player=None,
                        self_draw=True,
                        fan_total=fan.fan_total,
                        alive=alive,
                        rules=self.rules,
                        dealer=state.dealer,
                    )
                )
                state.players[pid].won = True
                events.append(
                    Event(
                        "hu",
                        actor=pid,
                        meta={"self_draw": True, "fan": fan.fan_total, "patterns": sorted(fan.patterns)},
                    )
                )

                if state.num_won() >= self.rules.max_round_wins:
                    state.phase = Phase.ROUND_END
                    events.append(Event("round_end", meta={"reason": "max_wins"}))
                else:
                    state.current_player = self._next_alive_after(state, pid)
                    state.phase = Phase.TURN_DRAW
                    state.turn_has_drawn = False
                    add_delta(self._auto_advance(state, events, rng=rng))

            elif da.kind == ActionKind.GANG_AN and da.arg is not None:
                tid = da.arg
                if state.players[pid].hand[tid] < 4:
                    raise ValueError("illegal GANG_AN")

                self._record_recent_action(state, pid=pid, action_id=actions[pid])
                counts_remove(state.players[pid].hand, tid, 4)
                state.players[pid].melds.append(Meld(MeldKind.GANG_AN, tid, from_player=None))

                alive = [i for i in range(4) if not state.players[i].won]
                add_delta(settle_gang(actor=pid, gang_kind=MeldKind.GANG_AN, alive=alive, rules=self.rules, dealer=state.dealer))
                events.append(Event("gang", actor=pid, tile=tid, meta={"kind": "an"}))

                state.phase = Phase.TURN_DRAW
                state.turn_has_drawn = False
                add_delta(
                    self._auto_advance(
                        state,
                        events,
                        rng=rng,
                        draw_from_dead_wall=True,
                        after_kong=True,
                    )
                )

            elif da.kind == ActionKind.GANG_BU and da.arg is not None:
                tid = da.arg
                midx = None
                for i, m in enumerate(state.players[pid].melds):
                    if m.kind == MeldKind.PENG and m.tile == tid:
                        midx = i
                        break
                if midx is None or state.players[pid].hand[tid] < 1:
                    raise ValueError("illegal GANG_BU")
                if (
                    state.players[pid].dingque_suit is not None
                    and tile_suit(tid) == state.players[pid].dingque_suit
                ):
                    raise ValueError("illegal GANG_BU (dingque suit)")

                self._record_recent_action(state, pid=pid, action_id=actions[pid])
                state.pending_kong = PendingKong(actor=pid, tile=tid, meld_index=midx)
                state.phase = Phase.RESPONSE_QIANGGANG
                events.append(Event("gang_bu_declared", actor=pid, tile=tid))

            elif da.kind == ActionKind.DISCARD and da.arg is not None:
                tid = da.arg
                p = state.players[pid]
                if p.hand[tid] <= 0:
                    raise ValueError("illegal DISCARD")
                if p.has_dingque_tiles() and p.dingque_suit is not None and tile_suit(tid) != p.dingque_suit:
                    raise ValueError("must discard a dingque-suit tile first")

                self._record_recent_action(state, pid=pid, action_id=actions[pid])
                counts_remove(p.hand, tid, 1)
                p.discards.append(tid)
                state.pending_discard = PendingDiscard(
                    from_player=pid,
                    tile=tid,
                    from_last_tile_draw=state.last_draw_was_last_tile,
                )

                # Turn context ends after a discard.
                state.last_draw_after_kong = False
                state.last_draw_was_last_tile = False
                state.turn_has_drawn = False

                state.phase = Phase.RESPONSE
                events.append(Event("discard", actor=pid, tile=tid))

            else:
                raise ValueError(f"unsupported action in TURN_ACTION: {da.kind}")

            state.rng_state = rng.getstate()
            if self.enable_events:
                state.events.extend(events)
            return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

        if state.phase == Phase.RESPONSE:
            assert state.pending_discard is not None
            disc = state.pending_discard
            discarder = disc.from_player
            t = disc.tile

            hu: list[int] = []
            gang: list[int] = []
            peng: list[int] = []

            for pid in required:
                aid = actions[pid]
                da = decode_action(aid)
                if da.kind == ActionKind.PASS:
                    continue

                mask = self.legal_action_mask(state, pid)
                if mask[aid] != 1:
                    raise ValueError(
                        f"Player {pid} chose {da.kind} in RESPONSE but action_mask forbids it "
                        f"(discarder={discarder}, tile={t})"
                    )

                if da.kind == ActionKind.HU:
                    hu.append(pid)
                elif da.kind == ActionKind.GANG_MING:
                    gang.append(pid)
                elif da.kind == ActionKind.PENG:
                    peng.append(pid)
                else:
                    raise ValueError(f"illegal response action: {da.kind}")

            for pid in required:
                self._record_recent_action(state, pid=pid, action_id=actions[pid])

            alive_before = [i for i in range(4) if not state.players[i].won]

            # HU has highest priority.
            if hu:
                if not self.rules.allow_yipao_duoxiang:
                    winner = self._pick_closest_to_player(discarder, hu)
                    hu = [winner] if winner is not None else []

                for winner in hu:
                    p = state.players[winner]
                    tmp = p.hand.copy()
                    tmp[t] += 1
                    ctx = WinContext(
                        winner=winner,
                        winning_tile=t,
                        self_draw=False,
                        from_player=discarder,
                        after_kong=False,
                        rob_kong=False,
                        last_tile_draw=False,
                        last_tile_discard=disc.from_last_tile_draw,
                    )
                    fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
                    if (not self.rules.allow_zero_fan) and fan.fan_total <= 0:
                        continue
                    add_delta(
                        settle_hu(
                            winner=winner,
                            from_player=discarder,
                            self_draw=False,
                            fan_total=fan.fan_total,
                            alive=alive_before,
                            rules=self.rules,
                            dealer=state.dealer,
                        )
                    )
                    state.players[winner].won = True
                    events.append(
                        Event(
                            "hu",
                            actor=winner,
                            tile=t,
                            meta={
                                "self_draw": False,
                                "from": discarder,
                                "fan": fan.fan_total,
                                "patterns": sorted(fan.patterns),
                            },
                        )
                    )

                state.pending_discard = None

                if state.num_won() >= self.rules.max_round_wins:
                    state.phase = Phase.ROUND_END
                    events.append(Event("round_end", meta={"reason": "max_wins"}))
                else:
                    state.current_player = self._next_alive_after(state, discarder)
                    state.phase = Phase.TURN_DRAW
                    state.turn_has_drawn = False
                    add_delta(self._auto_advance(state, events, rng=rng))

                state.rng_state = rng.getstate()
                if self.enable_events:
                    state.events.extend(events)
                return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

            # No HU. Try exposed kong.
            g = self._pick_closest_to_player(discarder, gang)
            if g is not None:
                if state.players[g].hand[t] < 3:
                    raise ValueError(
                        f"illegal GANG_MING: player {g} has {state.players[g].hand[t]} of tile {t} (need 3)"
                    )
                self._consume_last_discard_for_claim(state, discarder=discarder, tile=t)
                counts_remove(state.players[g].hand, t, 3)
                state.players[g].melds.append(Meld(MeldKind.GANG_MING, t, from_player=discarder))
                alive = [i for i in range(4) if not state.players[i].won]
                add_delta(settle_gang(actor=g, gang_kind=MeldKind.GANG_MING, alive=alive, rules=self.rules, dealer=state.dealer))
                events.append(Event("gang", actor=g, tile=t, meta={"kind": "ming", "from": discarder}))

                state.pending_discard = None
                state.current_player = g
                state.phase = Phase.TURN_DRAW
                state.turn_has_drawn = False
                add_delta(
                    self._auto_advance(
                        state,
                        events,
                        rng=rng,
                        draw_from_dead_wall=True,
                        after_kong=True,
                    )
                )

                state.rng_state = rng.getstate()
                if self.enable_events:
                    state.events.extend(events)
                return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

            # Try peng.
            pclaimer = self._pick_closest_to_player(discarder, peng)
            if pclaimer is not None:
                if state.players[pclaimer].hand[t] < 2:
                    raise ValueError("illegal PENG")
                self._consume_last_discard_for_claim(state, discarder=discarder, tile=t)
                counts_remove(state.players[pclaimer].hand, t, 2)
                state.players[pclaimer].melds.append(Meld(MeldKind.PENG, t, from_player=discarder))
                events.append(Event("peng", actor=pclaimer, tile=t, meta={"from": discarder}))

                state.pending_discard = None
                state.current_player = pclaimer
                state.last_draw_after_kong = False
                state.last_draw_was_last_tile = False
                state.turn_has_drawn = False
                state.phase = Phase.TURN_ACTION

                state.rng_state = rng.getstate()
                if self.enable_events:
                    state.events.extend(events)
                return StepResult(score_delta=score_delta, events=events, done=False)

            # Everyone passes.
            state.pending_discard = None
            state.current_player = self._next_alive_after(state, discarder)
            state.phase = Phase.TURN_DRAW
            state.turn_has_drawn = False
            add_delta(self._auto_advance(state, events, rng=rng))

            state.rng_state = rng.getstate()
            if self.enable_events:
                state.events.extend(events)
            return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

        if state.phase == Phase.RESPONSE_QIANGGANG:
            assert state.pending_kong is not None
            pk = state.pending_kong
            actor = pk.actor
            t = pk.tile

            hu: list[int] = []
            for pid in required:
                aid = actions[pid]
                da = decode_action(aid)
                if da.kind == ActionKind.PASS:
                    continue
                if da.kind != ActionKind.HU:
                    raise ValueError("only HU/PASS allowed in RESPONSE_QIANGGANG")

                mask = self.legal_action_mask(state, pid)
                if mask[aid] != 1:
                    raise ValueError(
                        f"Player {pid} chose HU in RESPONSE_QIANGGANG but action_mask forbids it "
                        f"(actor={actor}, tile={t})"
                    )
                hu.append(pid)

            for pid in required:
                self._record_recent_action(state, pid=pid, action_id=actions[pid])

            alive_before = [i for i in range(4) if not state.players[i].won]

            if hu:
                if not self.rules.allow_yipao_duoxiang:
                    winner = self._pick_closest_to_player(actor, hu)
                    hu = [winner] if winner is not None else []

                # Remove the tile from actor once (the robbed tile is treated as a discard).
                if state.players[actor].hand[t] < 1:
                    raise ValueError("actor does not have the tile for bu-gang")
                counts_remove(state.players[actor].hand, t, 1)
                # Keep tile conservation consistent with normal discard-HU: the winning tile
                # is public, but we don't add it into the winner's concealed hand.
                state.players[actor].discards.append(t)

                for winner in hu:
                    p = state.players[winner]
                    tmp = p.hand.copy()
                    tmp[t] += 1
                    ctx = WinContext(
                        winner=winner,
                        winning_tile=t,
                        self_draw=False,
                        from_player=actor,
                        after_kong=False,
                        rob_kong=True,
                        last_tile_draw=False,
                        last_tile_discard=False,
                    )
                    fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
                    if (not self.rules.allow_zero_fan) and fan.fan_total <= 0:
                        continue
                    add_delta(
                        settle_hu(
                            winner=winner,
                            from_player=actor,
                            self_draw=False,
                            fan_total=fan.fan_total,
                            alive=alive_before,
                            rules=self.rules,
                            dealer=state.dealer,
                        )
                    )
                    state.players[winner].won = True
                    events.append(
                        Event(
                            "hu",
                            actor=winner,
                            tile=t,
                            meta={
                                "rob_kong": True,
                                "from": actor,
                                "fan": fan.fan_total,
                                "patterns": sorted(fan.patterns),
                            },
                        )
                    )

                state.pending_kong = None
                if state.num_won() >= self.rules.max_round_wins:
                    state.phase = Phase.ROUND_END
                    events.append(Event("round_end", meta={"reason": "max_wins"}))
                else:
                    state.current_player = self._next_alive_after(state, actor)
                    state.phase = Phase.TURN_DRAW
                    state.turn_has_drawn = False
                    add_delta(self._auto_advance(state, events, rng=rng))

                state.rng_state = rng.getstate()
                if self.enable_events:
                    state.events.extend(events)
                return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

            # No one robs -> execute bu-gang.
            if state.players[actor].hand[t] < 1:
                raise ValueError("illegal GANG_BU")
            counts_remove(state.players[actor].hand, t, 1)
            m = state.players[actor].melds[pk.meld_index]
            state.players[actor].melds[pk.meld_index] = Meld(MeldKind.GANG_BU, m.tile, from_player=m.from_player)
            alive = [i for i in range(4) if not state.players[i].won]
            add_delta(settle_gang(actor=actor, gang_kind=MeldKind.GANG_BU, alive=alive, rules=self.rules, dealer=state.dealer))
            events.append(Event("gang", actor=actor, tile=t, meta={"kind": "bu"}))

            state.pending_kong = None
            state.current_player = actor
            state.phase = Phase.TURN_DRAW
            state.turn_has_drawn = False
            add_delta(
                self._auto_advance(
                    state,
                    events,
                    rng=rng,
                    draw_from_dead_wall=True,
                    after_kong=True,
                )
            )

            state.rng_state = rng.getstate()
            if self.enable_events:
                state.events.extend(events)
            return StepResult(score_delta=score_delta, events=events, done=(state.phase == Phase.ROUND_END))

        raise ValueError(f"cannot step in phase: {state.phase}")

    def _resolve_swap(self, state: GameState, rng: RNG, events: list[Event]) -> list[int]:
        if not self.rules.swap_enabled:
            state.phase = Phase.DINGQUE
            return [0, 0, 0, 0]

        direction = self.rules.swap_direction
        if direction == "random":
            direction = rng.choice(["clockwise", "counterclockwise", "across"])

        events.append(Event("swap_resolve", meta={"direction": direction}))

        recv_from: list[int]
        if direction == "clockwise":
            recv_from = [(i - 1) % 4 for i in range(4)]
        elif direction == "counterclockwise":
            recv_from = [(i + 1) % 4 for i in range(4)]
        elif direction == "across":
            recv_from = [(i + 2) % 4 for i in range(4)]
        else:
            raise ValueError(f"invalid swap_direction: {direction}")

        # Validate picks up-front to avoid partially mutating hands on error.
        for pid in range(4):
            if len(state.swap_picks[pid]) != 3:
                raise ValueError(f"player {pid} swap_picks must have exactly 3 tiles, got {len(state.swap_picks[pid])}")

        for pid in range(4):
            src = recv_from[pid]
            tiles = state.swap_picks[src]
            for tid in tiles:
                counts_add(state.players[pid].hand, tid, 1)

        state.swap_picks = [[], [], [], []]

        state.phase = Phase.DINGQUE if self.rules.dingque_enabled else Phase.TURN_DRAW
        if state.phase == Phase.TURN_DRAW:
            state.turn_has_drawn = False
            return self._auto_advance(state, events, rng=rng)

        return [0, 0, 0, 0]

    def _auto_advance(
        self,
        state: GameState,
        events: list[Event],
        *,
        rng: RNG,
        draw_from_dead_wall: bool = False,
        after_kong: bool = False,
    ) -> list[int]:
        """Advance internal phases until a decision is needed or the round ends.

        Returns score delta generated during auto-advance (e.g. wall-empty settlement).
        """
        score_delta = [0, 0, 0, 0]

        # Defensive: callers should normally stop the round when max wins is reached,
        # but guard here to prevent infinite loops on corrupted/hand-crafted states.
        if state.num_won() >= self.rules.max_round_wins:
            if state.phase != Phase.ROUND_END:
                state.phase = Phase.ROUND_END
                events.append(Event("round_end", meta={"reason": "max_wins"}))
            return score_delta

        while True:
            if state.phase == Phase.TURN_DRAW:
                if state.wall_pos >= state.wall_end:
                    alive = [i for i in range(4) if not state.players[i].won]
                    d = settle_hua_zhu_and_cha_jiao(players=state.players, alive=alive, rules=self.rules)
                    for i in range(4):
                        score_delta[i] += d[i]
                    events.append(Event("round_end", meta={"reason": "wall_empty", "delta": d}))
                    state.phase = Phase.ROUND_END
                    return score_delta

                pid = state.current_player
                if state.players[pid].won:
                    nxt = self._next_alive_after(state, pid)
                    if nxt == pid:
                        state.phase = Phase.ROUND_END
                        events.append(Event("round_end", meta={"reason": "all_won"}))
                        return score_delta
                    state.current_player = nxt
                    continue

                if draw_from_dead_wall:
                    # Draw from the tail (a simplified dead wall model).
                    state.wall_end -= 1
                    tid = state.wall[state.wall_end]
                else:
                    tid = state.wall[state.wall_pos]
                    state.wall_pos += 1

                state.players[pid].hand[tid] += 1
                state.last_draw_after_kong = after_kong
                state.last_draw_was_last_tile = state.wall_pos >= state.wall_end
                state.turn_has_drawn = True

                events.append(
                    Event(
                        "draw",
                        actor=pid,
                        tile=tid,
                        meta={
                            "after_kong": after_kong,
                            "from_dead_wall": draw_from_dead_wall,
                            "last_tile": state.last_draw_was_last_tile,
                        },
                    )
                )

                state.phase = Phase.TURN_ACTION
                return score_delta

            # Stop on decision phases.
            if state.phase in (
                Phase.SWAP_PICK_1,
                Phase.SWAP_PICK_2,
                Phase.SWAP_PICK_3,
                Phase.DINGQUE,
                Phase.TURN_ACTION,
                Phase.RESPONSE,
                Phase.RESPONSE_QIANGGANG,
                Phase.ROUND_END,
            ):
                return score_delta

            return score_delta

    def _next_alive_after(self, state: GameState, pid: int) -> int:
        for i in range(1, 5):
            nid = (pid + i) % 4
            if not state.players[nid].won:
                return nid
        return pid
