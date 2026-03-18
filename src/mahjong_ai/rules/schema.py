from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    value = _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    value = _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(slots=True)
class RulesConfig:
    """Rule/config bundle for Sichuan Mahjong (Xuezhan Daodi).

    This project keeps the engine flexible by pushing scoring knobs to config.
    """

    # Core flow.
    swap_enabled: bool = True
    swap_direction: str = "random"  # random|clockwise|counterclockwise|across
    dingque_enabled: bool = True

    allow_chi: bool = False
    allow_yipao_duoxiang: bool = True

    # Match termination.
    max_round_wins: int = 3

    # Scoring.
    base_points: int = 1
    fan_cap: int = 8
    allow_zero_fan: bool = True

    # Dealer multiplier ("zhuang-xian" multiplier).
    # If enabled, any transfer involving the dealer (payer or receiver) is multiplied.
    enable_dealer_multiplier: bool = True
    dealer_multiplier: int = 2

    # Dianpao (discard win) settlement.
    # - False: only the discarder pays the winner (common platform behavior).
    # - True: discarder pays as if all alive opponents paid (keeps total comparable to self-draw).
    dianpao_pays_all_alive: bool = False

    gang_ming_pay: int = 1  # each alive opponent pays base*pay
    gang_an_pay: int = 2
    gang_bu_pay: int = 1

    enable_hua_zhu: bool = True
    hua_zhu_penalty: int = 16

    enable_cha_jiao: bool = True
    cha_jiao_penalty: int = 8

    # Fan patterns (pattern_name -> fan_value).
    fan_patterns: dict[str, int] = field(default_factory=lambda: {
        "qidui": 2,
        "qingyise": 2,
        "pengpenghu": 2,
        "gangshanghua": 1,
        "qiangganghu": 1,
        "haidilaoyue": 1,
        "haidipao": 1,
    })

    def validate(self) -> None:
        if self.swap_direction not in {"random", "clockwise", "counterclockwise", "across"}:
            raise ValueError(f"invalid swap_direction: {self.swap_direction}")
        _require_positive_int("fan_cap", self.fan_cap)
        _require_positive_int("base_points", self.base_points)
        _require_positive_int("dealer_multiplier", self.dealer_multiplier)
        _require_non_negative_int("gang_ming_pay", self.gang_ming_pay)
        _require_non_negative_int("gang_an_pay", self.gang_an_pay)
        _require_non_negative_int("gang_bu_pay", self.gang_bu_pay)
        _require_non_negative_int("hua_zhu_penalty", self.hua_zhu_penalty)
        _require_non_negative_int("cha_jiao_penalty", self.cha_jiao_penalty)
        _require_int("max_round_wins", self.max_round_wins)
        if self.max_round_wins not in {1, 2, 3}:
            raise ValueError("max_round_wins must be 1..3")
        if not isinstance(self.fan_patterns, dict):
            raise ValueError("fan_patterns must be a mapping")
        for name, value in self.fan_patterns.items():
            if not isinstance(name, str) or not name:
                raise ValueError("fan_patterns keys must be non-empty strings")
            _require_non_negative_int(f"fan_patterns[{name!r}]", value)
