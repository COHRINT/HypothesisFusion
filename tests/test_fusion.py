import numpy as np
import pytest

from hypothesis_fusion.config import (
    FusionConfig,
    GrowthConfig,
    ResampleConfig,
    TrackIdConfig,
)
from hypothesis_fusion.fusion import HypothesisFusion
from tests.fakes import cv_truth
from tests.position_sensing import PositionInitializer, PositionSensing

DT = 0.5
TARGETS = [[0.0, 0.0, 1.0, 0.0], [50.0, 50.0, 0.0, -1.0]]


def _config(**overrides) -> FusionConfig:
    params = dict(
        growth=GrowthConfig(partition_gate=5.0, pair_gate=2.0),
        track_id=TrackIdConfig(max_distance=5.0),
    )
    params.update(overrides)
    return FusionConfig(**params)


def _node(feasibility=None, node_id=0, **overrides) -> HypothesisFusion:
    return HypothesisFusion(
        _config(**overrides),
        PositionInitializer(),
        position_of=lambda x: x[:2],
        feasibility=feasibility,
        node_id=node_id,
    )


def _run(node, sensing, steps, start=1, **kwargs):
    summaries = []
    for k in range(start, start + steps):
        summaries.append(node.update(k * DT, sensing.releases(k * DT, **kwargs)))
    return summaries


def _target_of(track_id: int) -> int:
    return track_id % 100


def test_no_hypotheses_until_sensor_tracks_mature():
    node = _node()
    sensing = PositionSensing(TARGETS, sensors=[1, 2, 3])
    _run(node, sensing, steps=2)
    assert not node.hypotheses
    _run(node, sensing, steps=1, start=3)
    assert node.hypotheses


def test_tracks_separated_targets_with_pure_hypotheses():
    node = _node()
    sensing = PositionSensing(TARGETS, sensors=[1, 2, 3])
    _run(node, sensing, steps=40)

    released = node.released_tracks
    assert len(released) == 2
    for hypothesis in released.values():
        targets = {_target_of(t) for t in hypothesis.sensor_track_ids}
        assert len(targets) == 1
        # Every sensor's track of that target was fused.
        assert len(hypothesis.sensor_track_ids) == 3
        truth = cv_truth(np.array(TARGETS[targets.pop()]), 40 * DT)
        assert np.linalg.norm(hypothesis.filter.latest.mean[:2] - truth[:2]) < 0.5


def test_partitions_separate_targets():
    node = _node()
    _run(node, PositionSensing(TARGETS, sensors=[1, 2, 3]), steps=20)
    summary = node.summary()
    assert len(summary.partitions) == 2
    for partition in summary.partitions:
        assert len({_target_of(t) for t in partition.sensor_track_ids}) == 1
        assert len(partition.best_tracks) == 1


def test_track_ids_and_histories_are_stable():
    node = _node()
    sensing = PositionSensing(TARGETS, sensors=[1, 2, 3])
    _run(node, sensing, steps=20)
    ids = set(node.released_tracks)
    _run(node, sensing, steps=20, start=21)
    assert set(node.released_tracks) == ids
    for track_id in ids:
        times = [e.time for e in node.track_histories[track_id].estimates]
        assert times == sorted(times)
        assert len(times) > 20


def test_release_waits_for_maturity():
    node = _node()
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2])
    _run(node, sensing, steps=4)
    assert node.best_hypotheses()
    assert not node.released_tracks
    _run(node, sensing, steps=10, start=5)
    assert node.released_tracks


def test_update_returns_summary_only_on_structural_change():
    node = _node()
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2])
    summaries = _run(node, sensing, steps=30)
    assert summaries[2] is not None  # hypotheses created
    assert summaries[-1] is None  # steady state


def test_terminated_sensor_track_goes_inactive_but_stays_in_pedigree():
    node = _node()
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2, 3])
    _run(node, sensing, steps=20)
    _run(node, sensing, steps=1, start=21, terminated={300})
    assert 300 in node.inactive_sensor_track_ids
    (best,) = node.best_hypotheses()
    assert 300 in best.sensor_track_ids
    assert node.active_sensor_track_ids(best) == {100, 200}


def test_silent_sensor_track_times_out():
    node = _node()
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2, 3])
    _run(node, sensing, steps=10)
    sensing.sensors = [1, 2]
    _run(node, sensing, steps=10, start=11)
    assert 300 in node.inactive_sensor_track_ids


def test_shrinking_drops_a_sensor_track_that_switches_targets(monkeypatch):
    shrunk = []
    original = HypothesisFusion._shrink_hypothesis

    def spy(self, time, parent, outlier):
        shrunk.append(outlier)
        return original(self, time, parent, outlier)

    monkeypatch.setattr(HypothesisFusion, '_shrink_hypothesis', spy)
    node = _node()
    # Sensor track 400 jumps to the other target at t = 10.
    sensing = PositionSensing(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 30.0, 1.0, 0.0]],
        sensors=[1, 2],
        switch=(400, 10.0, 1),
    )
    # Stagger sensors so each matures into an existing partition.
    _run(node, sensing, steps=4)
    sensing.sensors = [1, 2, 3]
    _run(node, sensing, steps=4, start=5)
    sensing.sensors = [1, 2, 3, 4]
    _run(node, sensing, steps=6, start=9)
    assert {100, 200, 300, 400} in [h.sensor_track_ids for h in node.best_hypotheses()]

    _run(node, sensing, steps=26, start=15)
    assert 400 in shrunk
    target0 = [
        h
        for h in node.released_tracks.values()
        if {100, 200, 300} <= h.sensor_track_ids
    ]
    assert len(target0) == 1
    assert 400 not in target0[0].sensor_track_ids


def test_housekeeping_prunes_infeasible_hypotheses():
    class FarOut:
        def is_feasible(self, estimate):
            return estimate.mean[0] < 10.0

    node = _node(feasibility=FarOut())
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2])
    _run(node, sensing, steps=10)
    assert node.housekeeping(5.0) is None
    _run(node, sensing, steps=20, start=11)  # the target passes x = 10
    assert node.housekeeping(15.0) is not None
    assert not node.hypotheses


def test_housekeeping_prunes_stale_hypotheses():
    node = _node()
    _run(node, PositionSensing(TARGETS[:1], sensors=[1, 2]), steps=10)
    assert node.hypotheses
    node.housekeeping(5.0 + node.config.prune.inactivity_window + 1.0)
    assert not node.hypotheses


def test_pruned_sensor_tracks_are_ignored():
    node = _node()
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2, 3])
    _run(node, sensing, steps=10)
    summary = node.prune_sensor_tracks(5.0, {300})
    assert 300 in summary.ignored_sensor_track_ids
    _run(node, sensing, steps=5, start=11)
    assert all(300 not in h.sensor_track_ids for h in node.hypotheses.values())
    assert 300 not in node.sensor_tracks


def test_merged_sensor_tracks_are_resampled_into_the_partition():
    resample = ResampleConfig(max_pair_residual=2.0, max_clique_residual=2.0)
    sensing = PositionSensing(TARGETS[:1], sensors=[1, 2, 3])
    donor, node = _node(resample=resample, node_id=1), _node(resample=resample)
    for k in range(1, 11):
        releases = sensing.releases(k * DT)
        node.update(k * DT, [r for r in releases if r.sensor_id != 3])
        donor.update(k * DT, releases)

    exported = donor.export_sensor_tracks({300}, since=0.0)
    assert [t.id for t in exported] == [300]
    summary = node.merge_sensor_tracks(5.0, exported, anchor_sensor_track_ids={100})
    assert {100, 200, 300} in [p.sensor_track_ids for p in summary.partitions]


def test_ids_are_namespaced_by_node():
    node = _node(node_id=3)
    _run(node, PositionSensing(TARGETS[:1], sensors=[1, 2]), steps=10)
    assert all(h // 1_000_000_000 == 3 for h in node.hypotheses)


@pytest.mark.parametrize('sensors', [[1], [1, 1]])
def test_single_sensor_never_forms_hypotheses(sensors):
    node = _node()
    _run(node, PositionSensing(TARGETS[:1], sensors=sensors), steps=10)
    assert not node.hypotheses


def test_triggered_resampling_keeps_hypotheses_pure(monkeypatch):
    resampled = []
    original = HypothesisFusion._resample

    def spy(self, time, partition):
        resampled.append(frozenset(partition.sensor_track_ids))
        return original(self, time, partition)

    monkeypatch.setattr(HypothesisFusion, '_resample', spy)
    node = _node(
        resample=ResampleConfig(max_pair_residual=2.0, max_clique_residual=2.0)
    )
    sensing = PositionSensing(TARGETS, sensors=[1, 2])
    _run(node, sensing, steps=4)
    sensing.sensors = [1, 2, 3]
    _run(node, sensing, steps=36, start=5)

    # Extending a partition leaves its best hypothesis under-utilizing it.
    assert resampled
    assert sorted(
        sorted(h.sensor_track_ids) for h in node.released_tracks.values()
    ) == [
        [100, 200, 300],
        [101, 201, 301],
    ]
