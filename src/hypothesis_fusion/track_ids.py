"""Hypothesis-to-track assignment.

No single hypothesis is guaranteed to be "correct" over a target's lifetime, so
system track IDs are kept separate from hypothesis IDs. Every frame, the
current best hypotheses are matched to the previous frame's tracks through a
hierarchy:

1. the same hypothesis ID as the previous assignment;
2. active sensor track overlap (:func:`~hypothesis_fusion.overlap.identity_overlap`)
   with the previous assignment of at least ``min_overlap``;
3. positional proximity to the previous assignment within ``max_distance``.

Matched hypotheses keep their track ID; the rest get a fresh one.
"""

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
from numpy import typing as npt

from hypothesis_fusion.overlap import identity_overlap


@dataclasses.dataclass(frozen=True)
class TrackCandidate:
    """The view of a best hypothesis needed to assign it a track ID."""

    hypothesis_id: int
    sensor_track_ids: frozenset[int]
    active_sensor_track_ids: frozenset[int]
    position: npt.NDArray[np.float64]


class TrackIdAssigner:
    """Frame-to-frame matcher keeping track IDs stable as best hypotheses change.

    ``assign`` is stateful: call it exactly once per frame.
    """

    def __init__(
        self, min_overlap: float, max_distance: float, new_track_id: Callable[[], int]
    ):
        self._min_overlap = min_overlap
        self._max_distance = max_distance
        self._new_track_id = new_track_id
        # Previous frame: hypothesis ID -> (track ID, candidate).
        self._previous: dict[int, tuple[int, TrackCandidate]] = {}

    def assign(self, candidates: Sequence[TrackCandidate]) -> dict[int, int]:
        """Map each candidate's hypothesis ID to a track ID."""
        assignment: dict[int, int] = {}
        available = dict(self._previous)

        # Tier 1: the exact hypothesis was assigned last frame.
        unmatched: list[TrackCandidate] = []
        for cand in candidates:
            if (prev := available.pop(cand.hypothesis_id, None)) is not None:
                assignment[cand.hypothesis_id] = prev[0]
            else:
                unmatched.append(cand)

        # Tier 2: sensor track overlap, greedy by descending overlap.
        def by_overlap(cand: TrackCandidate, prev: TrackCandidate) -> float | None:
            score = identity_overlap(
                cand.active_sensor_track_ids,
                cand.sensor_track_ids,
                prev.active_sensor_track_ids,
                prev.sensor_track_ids,
            )
            return -score if score >= self._min_overlap else None

        unmatched = self._greedy(unmatched, available, assignment, by_overlap)

        # Tier 3: positional proximity, greedy by ascending distance.
        def by_distance(cand: TrackCandidate, prev: TrackCandidate) -> float | None:
            distance = float(np.linalg.norm(cand.position - prev.position))
            return distance if distance <= self._max_distance else None

        unmatched = self._greedy(unmatched, available, assignment, by_distance)

        for cand in unmatched:
            assignment[cand.hypothesis_id] = self._new_track_id()

        self._previous = {
            cand.hypothesis_id: (assignment[cand.hypothesis_id], cand)
            for cand in candidates
        }
        return assignment

    @staticmethod
    def _greedy(
        unmatched: list[TrackCandidate],
        available: dict[int, tuple[int, TrackCandidate]],
        assignment: dict[int, int],
        cost: Callable[[TrackCandidate, TrackCandidate], float | None],
    ) -> list[TrackCandidate]:
        """Greedily match by ascending cost (``None`` = not allowed)."""
        pairs = []
        for cand in unmatched:
            for prev_hyp_id, (track_id, prev) in available.items():
                if (c := cost(cand, prev)) is not None:
                    pairs.append((c, cand.hypothesis_id, track_id, cand, prev_hyp_id))
        pairs.sort(key=lambda p: p[:3])
        matched: set[int] = set()
        for _, hyp_id, track_id, _cand, prev_hyp_id in pairs:
            if hyp_id in matched or prev_hyp_id not in available:
                continue
            assignment[hyp_id] = track_id
            matched.add(hyp_id)
            del available[prev_hyp_id]
        return [c for c in unmatched if c.hypothesis_id not in matched]
