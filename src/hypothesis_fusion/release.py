"""Track release.

The best hypothesis for a track can flip between competing hypotheses, and
positional matching in track ID assignment gives no guarantee that consecutive
assignments share any sensor tracks. A track is therefore only *released*
(surfaced as unambiguous) once its assignment has held for ``maturity``
seconds, where an assignment holds if each frame's hypothesis is the same as
the last, or overlaps it (active sensor tracks) by at least ``min_overlap``.
Any other change restarts the track's timer.
"""

import dataclasses
from collections.abc import Sequence

from hypothesis_fusion.overlap import identity_overlap


@dataclasses.dataclass(frozen=True)
class ReleaseCandidate:
    """The view of a best hypothesis needed to gate its release."""

    hypothesis_id: int
    track_id: int
    sensor_track_ids: frozenset[int]
    active_sensor_track_ids: frozenset[int]


@dataclasses.dataclass(frozen=True)
class _Record:
    candidate: ReleaseCandidate
    since: float


class ReleaseGate:
    """Releases tracks whose hypothesis assignment has been stable long enough.

    ``released`` is stateful: call it exactly once per frame.
    """

    def __init__(self, maturity: float, min_overlap: float):
        self._maturity = maturity
        self._min_overlap = min_overlap
        self._records: dict[int, _Record] = {}

    def released(self, candidates: Sequence[ReleaseCandidate], time: float) -> set[int]:
        """Hypothesis IDs of the candidates whose tracks are released."""
        records: dict[int, _Record] = {}
        released: set[int] = set()
        for cand in candidates:
            prev = self._records.get(cand.track_id)
            if prev is not None and self._continues(prev.candidate, cand):
                since = prev.since
            else:
                since = time
            records[cand.track_id] = _Record(cand, since)
            # A brand new record is never released on its first frame, even
            # with zero maturity, so every release follows an observed frame.
            if prev is not None and time - since >= self._maturity:
                released.add(cand.hypothesis_id)
        # Tracks without a candidate this frame lose their record.
        self._records = records
        return released

    def _continues(self, prev: ReleaseCandidate, cand: ReleaseCandidate) -> bool:
        if prev.hypothesis_id == cand.hypothesis_id:
            return True
        return (
            identity_overlap(
                prev.active_sensor_track_ids,
                prev.sensor_track_ids,
                cand.active_sensor_track_ids,
                cand.sensor_track_ids,
            )
            >= self._min_overlap
        )
