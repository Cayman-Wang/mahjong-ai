from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Event:
    type: str
    actor: int | None = None
    tile: int | None = None
    meta: dict[str, object] = field(default_factory=dict)
