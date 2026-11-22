import math

import pytest

from hypothesis_fusion.partitions import build_partitions, redundancy, utilization


def _partition_sets(partitions):
    return sorted(sorted(p.hypothesis_ids) for p in partitions)


def test_chained_hypotheses_share_a_partition():
    # 1-2 share track 11, 2-3 share track 12; 4 is disjoint (MFI Fig. 2).
    tracks = {1: {10, 11}, 2: {11, 12}, 3: {12, 13}, 4: {20, 21}}
    partitions = build_partitions(tracks, tracks, best_hypothesis_ids=[1, 4])
    assert _partition_sets(partitions) == [[1, 2, 3], [4]]
    by_first = {min(p.hypothesis_ids): p for p in partitions}
    assert by_first[1].sensor_track_ids == {10, 11, 12, 13}
    assert by_first[1].best_hypothesis_ids == {1}
    assert by_first[4].best_hypothesis_ids == {4}


def test_inactive_tracks_do_not_connect_partitions():
    tracks = {1: {10, 11}, 2: {11, 12}}
    active = {1: {10}, 2: {12}}
    partitions = build_partitions(tracks, active, best_hypothesis_ids=[])
    assert _partition_sets(partitions) == [[1], [2]]
    assert all(p.inactive_sensor_track_ids == {11} for p in partitions)


def test_utilization_of_best_hypotheses():
    assert utilization([{1, 2}, {3}], {1, 2, 3}) == 1.0
    assert utilization([{1}], {1, 2, 3}) == pytest.approx(0.5)


def test_lone_full_hypothesis_has_no_redundancy():
    assert redundancy([{1, 2, 3}], {1, 2, 3}) == 0.0


def test_any_excluding_hypothesis_counts_as_a_fallback():
    assert redundancy([{1, 2, 3}, {1, 2}], {1, 2, 3}) == pytest.approx(1 / 3)
    assert redundancy([{1, 2}, {2, 3}, {1, 3}], {1, 2, 3}) == 1.0


def test_hypothesis_with_no_active_tracks_is_not_a_fallback():
    assert redundancy([{1, 2, 3}, set()], {1, 2, 3}) == 0.0


def test_empty_partition_redundancy_is_undefined():
    assert math.isnan(redundancy([], set()))
