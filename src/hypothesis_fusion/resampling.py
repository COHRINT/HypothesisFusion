"""Hypothesis resampling.

Hypotheses are grown incrementally as sensor tracks arrive, so a partition can
degenerate: ambiguous sensor tracks may keep the eventually-correct hypothesis
from ever being initialized, or a sensor track that switches targets may drag
down every hypothesis built on it. Akin to resampling in a particle filter,
resampling re-hypothesizes over a partition's sensor tracks (plus orphaned
ones) all at once:

1. Build an adjacency graph whose edges join sensor tracks from *different*
   sensors whose pairwise residual is within ``max_pair_residual``.
2. Enumerate maximal cliques (Bron-Kerbosch with pivoting).
3. Every sub-clique of at least ``min_size`` tracks whose N-way residual is
   within ``max_clique_residual`` is a candidate hypothesis, ranked by
   ``-residual + size_bias * size``; the ``k`` best are returned.
4. :func:`reconcile_partition` merges the candidates into the partition.

The residual is application-specific (for bearings sensors, the RMS distance
of the lines of sight from their least-squares intersection); see
:meth:`hypothesis_fusion.initialization.Initializer.residual`. Since edges only
join different sensors, every candidate has at most one track per sensor.

Enumerating sub-cliques scales with the power set of the largest mutually
consistent cluster, which is fine for the small sensor track sets of a
partition but is the knob to revisit for dense scenes.
"""

import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence

from hypothesis_fusion.types import SensorTrack

Residual = Callable[[Sequence[SensorTrack]], float | None]


@dataclasses.dataclass(frozen=True)
class ResampleCandidate:
    """A candidate hypothesis: sensor tracks consistent with one common target."""

    sensor_track_ids: frozenset[int]
    residual: float
    score: float


def bron_kerbosch(adjacency: dict[int, set[int]]) -> list[set[int]]:
    """All maximal cliques of an undirected graph (Bron-Kerbosch with pivoting)."""
    cliques: list[set[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            cliques.append(r)
            return
        pivot = max(p | x, key=lambda u: len(adjacency[u] & p))
        for v in list(p - adjacency[pivot]):
            expand(r | {v}, p & adjacency[v], x & adjacency[v])
            p = p - {v}
            x = x | {v}

    expand(set(), set(adjacency), set())
    return cliques


class CliqueResampler:
    def __init__(
        self,
        k: int,
        max_pair_residual: float,
        max_clique_residual: float,
        min_size: int,
        size_bias: float,
    ):
        if min_size < 2:
            raise ValueError(f'min_size must be >= 2, got {min_size}')
        self.k = k
        self.max_pair_residual = max_pair_residual
        self.max_clique_residual = max_clique_residual
        self.min_size = min_size
        self.size_bias = size_bias

    def candidates(
        self, tracks: Sequence[SensorTrack], residual: Residual
    ) -> list[ResampleCandidate]:
        """Up to ``k`` candidate hypotheses over ``tracks``, best first."""
        if len(tracks) < self.min_size:
            return []

        adjacency: dict[int, set[int]] = defaultdict(set)
        for i, j in itertools.combinations(range(len(tracks)), 2):
            if tracks[i].sensor_id == tracks[j].sensor_id:
                continue
            pair = residual([tracks[i], tracks[j]])
            if pair is not None and pair <= self.max_pair_residual:
                adjacency[i].add(j)
                adjacency[j].add(i)

        seen: set[frozenset[int]] = set()
        candidates: list[ResampleCandidate] = []
        for clique in bron_kerbosch(dict(adjacency)):
            members = sorted(clique)
            for size in range(self.min_size, len(members) + 1):
                for subset in itertools.combinations(members, size):
                    key = frozenset(subset)
                    if key in seen:
                        continue
                    seen.add(key)
                    value = residual([tracks[i] for i in subset])
                    if value is None or value > self.max_clique_residual:
                        continue
                    candidates.append(
                        ResampleCandidate(
                            sensor_track_ids=frozenset(tracks[i].id for i in subset),
                            residual=value,
                            score=-value + self.size_bias * size,
                        )
                    )

        candidates.sort(key=lambda c: (-c.score, sorted(c.sensor_track_ids)))
        return candidates[: self.k]


@dataclasses.dataclass(frozen=True)
class PartitionHypothesis:
    """A hypothesis currently in the partition being resampled."""

    hypothesis_id: int
    sensor_track_ids: frozenset[int]
    is_best: bool


@dataclasses.dataclass(frozen=True)
class Reconciliation:
    """How to update a partition after resampling: keep ``kept_hypothesis_ids``
    (preserving their filter histories), create a hypothesis for each of
    ``new_sensor_track_sets``, and prune ``pruned_hypothesis_ids``."""

    kept_hypothesis_ids: set[int]
    new_sensor_track_sets: list[frozenset[int]]
    pruned_hypothesis_ids: set[int]


def reconcile_partition(
    current: Sequence[PartitionHypothesis],
    candidates: Sequence[ResampleCandidate],
) -> Reconciliation:
    """Keep the best hypotheses and any candidate that already exists; add the
    remaining candidates; prune everything else in the partition."""
    ids_by_set: dict[frozenset[int], list[int]] = defaultdict(list)
    for hypothesis in current:
        ids_by_set[hypothesis.sensor_track_ids].append(hypothesis.hypothesis_id)

    kept = {h.hypothesis_id for h in current if h.is_best}
    new_sets: list[frozenset[int]] = []
    for candidate in candidates:
        if existing := ids_by_set.get(candidate.sensor_track_ids):
            kept.update(existing)
        else:
            new_sets.append(candidate.sensor_track_ids)

    pruned = {h.hypothesis_id for h in current if h.hypothesis_id not in kept}
    return Reconciliation(kept, new_sets, pruned)
