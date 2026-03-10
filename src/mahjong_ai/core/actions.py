from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .tiles import NUM_TILE_TYPES, NUM_SUITS


class ActionKind(str, Enum):
    DISCARD = "discard"
    SWAP_PICK = "swap_pick"
    PENG = "peng"
    GANG_MING = "gang_ming"
    GANG_AN = "gang_an"
    GANG_BU = "gang_bu"
    DINGQUE = "dingque"
    HU = "hu"
    PASS = "pass"


# Fixed discrete action space (size=167).
# Offsets chosen to keep each tile-indexed action in a contiguous block.
_DISCARD_OFF = 0
_SWAP_OFF = _DISCARD_OFF + NUM_TILE_TYPES
_PENG_OFF = _SWAP_OFF + NUM_TILE_TYPES
_GANG_MING_OFF = _PENG_OFF + NUM_TILE_TYPES
_GANG_AN_OFF = _GANG_MING_OFF + NUM_TILE_TYPES
_GANG_BU_OFF = _GANG_AN_OFF + NUM_TILE_TYPES
_DINGQUE_OFF = _GANG_BU_OFF + NUM_TILE_TYPES
_HU_ID = _DINGQUE_OFF + NUM_SUITS
_PASS_ID = _HU_ID + 1

N_ACTIONS = _PASS_ID + 1


@dataclass(frozen=True, slots=True)
class DecodedAction:
    kind: ActionKind
    arg: int | None = None


def encode_action(kind: ActionKind, arg: int | None = None) -> int:
    if kind == ActionKind.DISCARD:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("DISCARD requires tile arg")
        return _DISCARD_OFF + arg
    if kind == ActionKind.SWAP_PICK:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("SWAP_PICK requires tile arg")
        return _SWAP_OFF + arg
    if kind == ActionKind.PENG:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("PENG requires tile arg")
        return _PENG_OFF + arg
    if kind == ActionKind.GANG_MING:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("GANG_MING requires tile arg")
        return _GANG_MING_OFF + arg
    if kind == ActionKind.GANG_AN:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("GANG_AN requires tile arg")
        return _GANG_AN_OFF + arg
    if kind == ActionKind.GANG_BU:
        if arg is None or not (0 <= arg < NUM_TILE_TYPES):
            raise ValueError("GANG_BU requires tile arg")
        return _GANG_BU_OFF + arg
    if kind == ActionKind.DINGQUE:
        if arg is None or not (0 <= arg < NUM_SUITS):
            raise ValueError("DINGQUE requires suit arg")
        return _DINGQUE_OFF + arg
    if kind == ActionKind.HU:
        if arg is not None:
            raise ValueError("HU takes no arg")
        return _HU_ID
    if kind == ActionKind.PASS:
        if arg is not None:
            raise ValueError("PASS takes no arg")
        return _PASS_ID
    raise ValueError(f"unknown action kind: {kind}")


def decode_action(action_id: int) -> DecodedAction:
    if not (0 <= action_id < N_ACTIONS):
        raise ValueError(f"invalid action_id: {action_id}")

    if _DISCARD_OFF <= action_id < _SWAP_OFF:
        return DecodedAction(ActionKind.DISCARD, action_id - _DISCARD_OFF)
    if _SWAP_OFF <= action_id < _PENG_OFF:
        return DecodedAction(ActionKind.SWAP_PICK, action_id - _SWAP_OFF)
    if _PENG_OFF <= action_id < _GANG_MING_OFF:
        return DecodedAction(ActionKind.PENG, action_id - _PENG_OFF)
    if _GANG_MING_OFF <= action_id < _GANG_AN_OFF:
        return DecodedAction(ActionKind.GANG_MING, action_id - _GANG_MING_OFF)
    if _GANG_AN_OFF <= action_id < _GANG_BU_OFF:
        return DecodedAction(ActionKind.GANG_AN, action_id - _GANG_AN_OFF)
    if _GANG_BU_OFF <= action_id < _DINGQUE_OFF:
        return DecodedAction(ActionKind.GANG_BU, action_id - _GANG_BU_OFF)
    if _DINGQUE_OFF <= action_id < _HU_ID:
        return DecodedAction(ActionKind.DINGQUE, action_id - _DINGQUE_OFF)
    if action_id == _HU_ID:
        return DecodedAction(ActionKind.HU, None)
    if action_id == _PASS_ID:
        return DecodedAction(ActionKind.PASS, None)

    raise AssertionError("unreachable")
