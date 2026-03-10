from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MatchStats:
    games: int
    total_scores: list[int]

    @property
    def avg_scores(self) -> list[float]:
        if self.games <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [s / self.games for s in self.total_scores]


def merge_stats(a: MatchStats, b: MatchStats) -> MatchStats:
    if a.games == 0:
        return b
    if b.games == 0:
        return a
    return MatchStats(games=a.games + b.games, total_scores=[a.total_scores[i] + b.total_scores[i] for i in range(4)])

