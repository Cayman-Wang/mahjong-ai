from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class RNG:
    """Thin wrapper to make RNG state explicit and serializable."""

    seed: int
    _rng: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> "RNG":
        return cls(seed=seed, _rng=random.Random(seed))

    def getstate(self) -> object:
        return self._rng.getstate()

    def setstate(self, state: object) -> None:
        self._rng.setstate(state)

    def shuffle(self, x: list[Any]) -> None:
        self._rng.shuffle(x)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: list[Any]) -> Any:
        return self._rng.choice(seq)

    def random(self) -> float:
        return self._rng.random()
