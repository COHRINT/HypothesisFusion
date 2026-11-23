import numpy as np
import pytest

from hypothesis_fusion.resampling import (
    CliqueResampler,
    PartitionHypothesis,
    ResampleCandidate,
    bron_kerbosch,
    reconcile_partition,
)
from hypothesis_fusion.types import SensorTrack


def _track(track_id: int, sensor_id: int, position) -> SensorTrack:
    """A sensor track whose 'local mean' is a direct position fix."""
    return SensorTrack(
        id=track_id,
        sensor_id=sensor_id,
        time=0.0,
        measurements=[],
        local_mean=np.asarray(position, dtype=float),
    )


def _spread(tracks) -> float:
    """RMS distance of the tracks' positions from their centroid."""
    points = np.array([t.local_mean for t in tracks])
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def _resampler(**overrides) -> CliqueResampler:
    params = dict(
        k=10, max_pair_residual=1.0, max_clique_residual=1.0, min_size=2, size_bias=1.0
    )
    params.update(overrides)
    return CliqueResampler(**params)


def _sets(candidates):
    return [set(c.sensor_track_ids) for c in candidates]


def test_bron_kerbosch_finds_maximal_cliques():
    adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 3}, 3: {2}}
    assert sorted(sorted(c) for c in bron_kerbosch(adjacency)) == [[0, 1, 2], [2, 3]]


def test_returns_full_clique_first_then_subsets():
    tracks = [_track(1, 1, [0, 0]), _track(2, 2, [0.1, 0]), _track(3, 3, [0, 0.1])]
    candidates = _resampler().candidates(tracks, _spread)
    assert _sets(candidates)[0] == {1, 2, 3}
    assert len(candidates) == 4  # the triple and its three pairs
    assert [c.score for c in candidates] == sorted(
        (c.score for c in candidates), reverse=True
    )


def test_two_separated_targets_never_mix():
    tracks = [
        _track(1, 1, [0, 0]),
        _track(2, 2, [0.1, 0]),
        _track(3, 1, [50, 0]),
        _track(4, 2, [50.1, 0]),
    ]
    assert sorted(map(sorted, _sets(_resampler().candidates(tracks, _spread)))) == [
        [1, 2],
        [3, 4],
    ]


def test_same_sensor_tracks_never_share_a_candidate():
    tracks = [_track(1, 1, [0, 0]), _track(2, 1, [0, 0]), _track(3, 2, [0, 0])]
    for candidate in _resampler().candidates(tracks, _spread):
        assert not {1, 2} <= candidate.sensor_track_ids


def test_infeasible_residual_excludes_pair():
    tracks = [_track(1, 1, [0, 0]), _track(2, 2, [0, 0])]
    assert _resampler().candidates(tracks, lambda ts: None) == []


def test_min_size_and_k_limit_results():
    tracks = [_track(i, i, [0.01 * i, 0]) for i in range(1, 5)]
    assert all(
        len(c.sensor_track_ids) >= 3
        for c in _resampler(min_size=3).candidates(tracks, _spread)
    )
    assert len(_resampler(k=2).candidates(tracks, _spread)) == 2


def test_min_size_must_allow_stereo():
    with pytest.raises(ValueError):
        _resampler(min_size=1)


def test_reconcile_keeps_best_and_existing_adds_new_prunes_rest():
    current = [
        PartitionHypothesis(1, frozenset({1, 2}), is_best=True),
        PartitionHypothesis(2, frozenset({1, 3}), is_best=False),
        PartitionHypothesis(3, frozenset({2, 3}), is_best=False),
    ]
    candidates = [
        ResampleCandidate(frozenset({1, 2, 3}), 0.1, 2.9),
        ResampleCandidate(frozenset({2, 3}), 0.1, 1.9),
    ]
    plan = reconcile_partition(current, candidates)
    assert plan.kept_hypothesis_ids == {1, 3}
    assert plan.new_sensor_track_sets == [frozenset({1, 2, 3})]
    assert plan.pruned_hypothesis_ids == {2}
