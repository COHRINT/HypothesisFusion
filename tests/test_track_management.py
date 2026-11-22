import itertools

import numpy as np

from hypothesis_fusion.release import ReleaseCandidate, ReleaseGate
from hypothesis_fusion.track_ids import TrackCandidate, TrackIdAssigner


def _track(hyp_id, tracks, position=(0.0, 0.0), active=None):
    return TrackCandidate(
        hypothesis_id=hyp_id,
        sensor_track_ids=frozenset(tracks),
        active_sensor_track_ids=frozenset(tracks if active is None else active),
        position=np.asarray(position, dtype=float),
    )


def _assigner():
    return TrackIdAssigner(
        min_overlap=0.6, max_distance=10.0, new_track_id=itertools.count(100).__next__
    )


class TestTrackIdAssigner:
    def test_new_hypotheses_get_fresh_ids(self):
        assignment = _assigner().assign([_track(1, {1, 2}), _track(2, {3, 4})])
        assert sorted(assignment.values()) == [100, 101]

    def test_same_hypothesis_keeps_id(self):
        assigner = _assigner()
        first = assigner.assign([_track(1, {1, 2})])
        assert assigner.assign([_track(1, {1, 2})]) == first

    def test_overlapping_hypothesis_inherits_id(self):
        assigner = _assigner()
        first = assigner.assign([_track(1, {1, 2})])
        second = assigner.assign([_track(7, {1, 2, 3}, position=(1e6, 0))])
        assert second[7] == first[1]

    def test_nearby_hypothesis_inherits_id(self):
        assigner = _assigner()
        first = assigner.assign([_track(1, {1, 2})])
        second = assigner.assign([_track(7, {5, 6}, position=(3.0, 4.0))])
        assert second[7] == first[1]

    def test_distant_disjoint_hypothesis_gets_new_id(self):
        assigner = _assigner()
        first = assigner.assign([_track(1, {1, 2})])
        second = assigner.assign([_track(7, {5, 6}, position=(30.0, 0.0))])
        assert second[7] != first[1]

    def test_each_previous_track_matches_once(self):
        assigner = _assigner()
        first = assigner.assign([_track(1, {1, 2})])
        second = assigner.assign([_track(7, {1, 2}), _track(8, {1, 2, 3})])
        # Higher overlap wins; the other hypothesis is a new track.
        assert second[7] == first[1]
        assert second[8] != first[1]


def _release(hyp_id, track_id, tracks, active=None):
    return ReleaseCandidate(
        hypothesis_id=hyp_id,
        track_id=track_id,
        sensor_track_ids=frozenset(tracks),
        active_sensor_track_ids=frozenset(tracks if active is None else active),
    )


class TestReleaseGate:
    def test_released_only_after_maturity(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        cand = _release(1, 100, {1, 2})
        assert gate.released([cand], 0.0) == set()
        assert gate.released([cand], 2.0) == set()
        assert gate.released([cand], 3.0) == {1}

    def test_similar_hypothesis_preserves_timer(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        gate.released([_release(1, 100, {1, 2})], 0.0)
        assert gate.released([_release(2, 100, {1, 2, 3})], 3.0) == {2}

    def test_dissimilar_hypothesis_restarts_timer(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        gate.released([_release(1, 100, {1, 2})], 0.0)
        assert gate.released([_release(2, 100, {5, 6})], 3.0) == set()
        assert gate.released([_release(2, 100, {5, 6})], 6.0) == {2}

    def test_disappearance_restarts_timer(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        cand = _release(1, 100, {1, 2})
        gate.released([cand], 0.0)
        gate.released([], 1.0)
        gate.released([cand], 2.0)
        assert gate.released([cand], 4.0) == set()
        assert gate.released([cand], 5.0) == {1}

    def test_inactive_differences_do_not_restart_timer(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        gate.released([_release(1, 100, {1, 2, 8}, active={1, 2})], 0.0)
        assert gate.released([_release(2, 100, {1, 2, 9}, active={1, 2})], 3.0) == {2}

    def test_independent_timers_per_track(self):
        gate = ReleaseGate(maturity=3.0, min_overlap=0.6)
        a, b = _release(1, 100, {1, 2}), _release(2, 200, {3, 4})
        gate.released([a], 0.0)
        gate.released([a, b], 2.0)
        assert gate.released([a, b], 3.0) == {1}
