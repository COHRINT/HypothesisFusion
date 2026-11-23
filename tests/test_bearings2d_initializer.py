"""Tests of the 2D bearings demo's stereo initializer."""

import numpy as np
import pytest

from examples.bearings2d.initializer import BearingsInitializer
from examples.bearings2d.models import ConstantVelocity
from hypothesis_fusion.types import Estimate, SensorTrack

ROBOTS = {1: np.array([0.0, 0.0]), 2: np.array([100.0, 0.0])}


def _initializer() -> BearingsInitializer:
    return BearingsInitializer(
        robot_position=lambda robot_id, t: ROBOTS[robot_id],
        dynamics=ConstantVelocity(),
        sensing_range=1000.0,
    )


def _track(track_id: int, robot_id: int, target) -> SensorTrack:
    offset = np.asarray(target, dtype=float) - ROBOTS[robot_id]
    return SensorTrack(
        id=track_id,
        sensor_id=robot_id,
        time=0.0,
        measurements=[],
        local_mean=np.array([np.arctan2(offset[1], offset[0]), 0.0]),
    )


class TestBearingsGeometry:
    def test_two_lines_of_sight_intersect_at_the_target(self):
        tracks = [_track(1, 1, (50, 50)), _track(2, 2, (50, 50))]
        point, residual = _initializer().intersect(tracks, 0.0)
        np.testing.assert_allclose(point, [50.0, 50.0], atol=1e-9)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_intersection_behind_a_robot_is_infeasible(self):
        # Robot 2 looks away from where robot 1's line crosses its own.
        tracks = [_track(1, 1, (50, 50)), _track(2, 2, (200, -100))]
        assert _initializer().residual(tracks, 0.0) is None

    def test_nearly_parallel_lines_are_infeasible(self):
        tracks = [_track(1, 1, (0, 1000)), _track(2, 2, (101, 1000))]
        assert _initializer().residual(tracks, 0.0) is None

    def test_track_residual_is_bearing_mahalanobis_distance(self):
        # 3 m off a line of sight at 100 m range, with a 1 m^2 position
        # variance across it: 0.03 rad over a 0.01 rad sigma.
        track = _track(1, 1, (100, 0))
        estimate = Estimate(0.0, np.array([100.0, 3.0, 0.0, 0.0]), np.eye(4))
        assert _initializer().track_residual(estimate, track) == pytest.approx(
            3.0, rel=1e-3
        )

    def test_track_residual_scales_with_uncertainty(self):
        # A near miss is far for a well-localized estimate, near for a vague one.
        track = _track(1, 1, (100, 0))
        mean = np.array([100.0, 8.0, 0.0, 0.0])
        tight = Estimate(0.0, mean, np.diag([4.0, 4.0, 1.0, 1.0]))
        loose = Estimate(0.0, mean, np.diag([400.0, 400.0, 1.0, 1.0]))
        initializer = _initializer()
        assert initializer.track_residual(tight, track) > 3.0
        assert initializer.track_residual(loose, track) < 1.0

    def test_track_residual_includes_track_uncertainty(self):
        track = _track(1, 1, (100, 0))
        track.local_covariance = np.diag([0.03**2, 0.0])
        estimate = Estimate(0.0, np.array([100.0, 3.0, 0.0, 0.0]), np.eye(4))
        assert _initializer().track_residual(estimate, track) < 1.0

    def test_track_residual_wraps_bearings(self):
        # Just either side of the -x axis: bearings near +pi and -pi.
        track = _track(1, 1, (-100, 0.5))
        estimate = Estimate(0.0, np.array([-100.0, -0.5, 0.0, 0.0]), np.eye(4))
        assert _initializer().track_residual(estimate, track) == pytest.approx(
            1.0, rel=1e-3
        )
