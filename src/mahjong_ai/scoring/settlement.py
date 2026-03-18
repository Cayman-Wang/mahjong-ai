from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.core.state import MeldKind, PlayerState
from mahjong_ai.core.tiles import NUM_TILE_TYPES

from .fan_patterns import WinContext, detect_fan_patterns
from .ting import is_ting
from .win_check import detect_win
from mahjong_ai.rules.schema import RulesConfig


@dataclass(frozen=True, slots=True)
class FanResult:
    fan_total: int
    patterns: set[str]


def compute_fan(
    *,
    counts: list[int],
    player: PlayerState,
    ctx: WinContext,
    rules: RulesConfig,
) -> FanResult:
    win = detect_win(
        counts,
        meld_count=player.meld_count(),
        dingque_suit=player.dingque_suit,
        dingque_in_melds=player.has_dingque_tiles_in_melds(),
    )
    if not win.ok:
        return FanResult(0, set())

    patterns = detect_fan_patterns(counts=counts, melds=player.melds, win_kind=win.kind or "", ctx=ctx)
    fan = 0
    for p in patterns:
        fan += int(rules.fan_patterns.get(p, 0))
    fan = min(fan, rules.fan_cap)
    return FanResult(fan, patterns)


def win_points(fan_total: int, rules: RulesConfig) -> int:
    return int(rules.base_points * (2 ** int(fan_total)))


def _transfer_amount(*, base: int, payer: int, receiver: int, dealer: int, rules: RulesConfig) -> int:
    if not rules.enable_dealer_multiplier:
        return base
    if payer == dealer or receiver == dealer:
        return base * int(rules.dealer_multiplier)
    return base


def _stable_fan_total_for_cha_jiao(*, counts: list[int], player: PlayerState, winning_tile: int, rules: RulesConfig) -> int:
    win = detect_win(
        counts,
        meld_count=player.meld_count(),
        dingque_suit=player.dingque_suit,
        dingque_in_melds=player.has_dingque_tiles_in_melds(),
    )
    if not win.ok:
        return 0

    ctx = WinContext(
        winner=-1,
        winning_tile=winning_tile,
        self_draw=False,
        from_player=None,
        after_kong=False,
        rob_kong=False,
        last_tile_draw=False,
        last_tile_discard=False,
    )
    patterns = detect_fan_patterns(counts=counts, melds=player.melds, win_kind=win.kind or "", ctx=ctx)
    fan_total = sum(int(rules.fan_patterns.get(pattern, 0)) for pattern in patterns)
    return min(fan_total, rules.fan_cap)


def _is_ting_for_cha_jiao(*, player: PlayerState, rules: RulesConfig) -> bool:
    if player.has_dingque_tiles_anywhere():
        return False

    if rules.allow_zero_fan:
        return is_ting(player.hand, meld_count=player.meld_count(), dingque_suit=player.dingque_suit)

    for tid in range(NUM_TILE_TYPES):
        if player.hand[tid] >= 4:
            continue
        tmp = player.hand.copy()
        tmp[tid] += 1
        if _stable_fan_total_for_cha_jiao(counts=tmp, player=player, winning_tile=tid, rules=rules) > 0:
            return True
    return False


def settle_gang(
    *,
    actor: int,
    gang_kind: MeldKind,
    alive: list[int],
    rules: RulesConfig,
    dealer: int,
) -> list[int]:
    if gang_kind == MeldKind.GANG_AN:
        pay = rules.gang_an_pay
    elif gang_kind == MeldKind.GANG_BU:
        pay = rules.gang_bu_pay
    else:
        pay = rules.gang_ming_pay

    delta = [0, 0, 0, 0]
    for pid in alive:
        if pid == actor:
            continue
        base = rules.base_points * pay
        amt = _transfer_amount(base=base, payer=pid, receiver=actor, dealer=dealer, rules=rules)
        delta[pid] -= amt
        delta[actor] += amt
    return delta


def settle_hu(
    *,
    winner: int,
    from_player: int | None,
    self_draw: bool,
    fan_total: int,
    alive: list[int],
    rules: RulesConfig,
    dealer: int,
) -> list[int]:
    pts = win_points(fan_total, rules)
    delta = [0, 0, 0, 0]

    if self_draw:
        for pid in alive:
            if pid == winner:
                continue
            amt = _transfer_amount(base=pts, payer=pid, receiver=winner, dealer=dealer, rules=rules)
            delta[pid] -= amt
            delta[winner] += amt
        return delta

    assert from_player is not None
    if rules.dianpao_pays_all_alive:
        # To keep total transfer comparable to self-draw, the discarder pays as if every alive opponent paid.
        payers = [pid for pid in alive if pid != winner]
        total = 0
        for pid in payers:
            total += _transfer_amount(base=pts, payer=pid, receiver=winner, dealer=dealer, rules=rules)
        delta[from_player] -= total
        delta[winner] += total
        return delta

    # Common behavior: only the discarder pays the winner.
    amt = _transfer_amount(base=pts, payer=from_player, receiver=winner, dealer=dealer, rules=rules)
    delta[from_player] -= amt
    delta[winner] += amt
    return delta


def settle_hua_zhu_and_cha_jiao(
    *,
    players: list[PlayerState],
    alive: list[int],
    rules: RulesConfig,
) -> list[int]:
    delta = [0, 0, 0, 0]
    if not alive:
        return delta

    hua_zhu: set[int] = set()
    if rules.enable_hua_zhu:
        for pid in alive:
            if players[pid].has_dingque_tiles_anywhere():
                hua_zhu.add(pid)

        non_hua = [pid for pid in alive if pid not in hua_zhu]
        for payer in hua_zhu:
            for recv in non_hua:
                amt = rules.hua_zhu_penalty
                delta[payer] -= amt
                delta[recv] += amt

    if rules.enable_cha_jiao:
        ting_players: set[int] = set()
        for pid in alive:
            # Dingque violation players cannot be in ting (even if hua_zhu penalty is disabled).
            if players[pid].has_dingque_tiles_anywhere() or pid in hua_zhu:
                continue
            p = players[pid]
            if _is_ting_for_cha_jiao(player=p, rules=rules):
                ting_players.add(pid)

        not_ting = [pid for pid in alive if pid not in ting_players]
        for recv in ting_players:
            for payer in not_ting:
                if payer == recv:
                    continue
                amt = rules.cha_jiao_penalty
                delta[payer] -= amt
                delta[recv] += amt

    return delta
