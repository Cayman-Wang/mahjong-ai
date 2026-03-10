from __future__ import annotations

from dataclasses import dataclass, field


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
        if self.fan_cap <= 0:
            raise ValueError("fan_cap must be positive")
        if self.base_points <= 0:
            raise ValueError("base_points must be positive")
        if self.dealer_multiplier <= 0:
            raise ValueError("dealer_multiplier must be positive")
        if self.max_round_wins not in {1, 2, 3}:
            raise ValueError("max_round_wins must be 1..3")
