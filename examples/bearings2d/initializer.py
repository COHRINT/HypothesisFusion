"""Stereo initialization for 2D bearings.

A sensor track's local estimate ``[bearing, bearing rate]``, extrapolated to a
common time and anchored at its robot's (known, predetermined) position, gives
a line of sight. Sensor tracks are related through those lines:

- ``residual``: RMS distance of the lines from their least-squares intersection.
  Note that in 2D any two non-parallel lines intersect exactly, so a *pair*
  only gates on geometry (in front of both robots, within range, not
  near-parallel); it takes a third line, or time, to expose a ghost.
- ``initialize`` supplies the time: it seeds a filter at the intersection with a
  loose covariance and "zips" through the tracks' recent measurements in time
  order. A ghost intersection moves inconsistently with the robots' bearing
  histories, so the mean Mahalanobis innovation distance gates it out.
- ``track_residual``: Mahalanobis distance between the bearing a hypothesis
  predicts from the track's robot and the track's own bearing.
"""

from collections.abc import Callable, Sequence

import numpy as np

from examples.bearings2d.models import BearingModel, ConstantVelocity, wrap
from hypothesis_fusion.filtering import UnscentedFilter
from hypothesis_fusion.scoring import innovation_distance
from hypothesis_fusion.types import Estimate, SensorTrack

RobotPosition = Callable[[int, float], np.ndarray]


class BearingsInitializer:
    def __init__(
        self,
        robot_position: RobotPosition,
        dynamics: ConstantVelocity,
        sensing_range: float,
        history: float = 10.0,
        innovation_gate: float = 2.0,
        min_crossing_angle: float = np.radians(5.0),
        initial_position_sigma: float = 100.0,
        initial_velocity_sigma: float = 10.0,
        filter_history: int = 50,
    ):
        self.robot_position = robot_position
        self.dynamics = dynamics
        self.model = BearingModel()
        self.sensing_range = sensing_range
        self.history = history
        self.innovation_gate = innovation_gate
        self.min_crossing_angle = min_crossing_angle
        self.initial_sigmas = np.array(
            [initial_position_sigma] * 2 + [initial_velocity_sigma] * 2
        )
        self.filter_history = filter_history

    # ----------------------------------------------------------- geometry

    def line_of_sight(
        self, track: SensorTrack, time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The track's (origin, unit direction) at ``time``."""
        bearing, rate = track.local_mean
        bearing = bearing + rate * (time - track.time)
        origin = self.robot_position(track.sensor_id, time)
        return origin, np.array([np.cos(bearing), np.sin(bearing)])

    def intersect(
        self, tracks: Sequence[SensorTrack], time: float
    ) -> tuple[np.ndarray, float] | None:
        """Least-squares intersection of the tracks' lines of sight and its RMS
        residual, or ``None`` if the geometry is infeasible."""
        lines = [self.line_of_sight(t, time) for t in tracks]
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for origin, direction in lines:
            projector = np.eye(2) - np.outer(direction, direction)
            A += projector
            b += projector @ origin
        # For two lines crossing at angle a, A's eigenvalues are 1 +- cos(a), so
        # this rejects geometry no better than two lines at the minimum angle.
        if np.linalg.eigvalsh(A)[0] < 1.0 - np.cos(self.min_crossing_angle):
            return None
        point = np.linalg.solve(A, b)
        squared = []
        for origin, direction in lines:
            along = direction @ (point - origin)
            if along <= 0.0 or along > self.sensing_range:
                return None  # behind the robot or out of its range
            perpendicular = point - origin - along * direction
            squared.append(perpendicular @ perpendicular)
        return point, float(np.sqrt(np.mean(squared)))

    # ------------------------------------------------- Initializer protocol

    def residual(self, tracks: Sequence[SensorTrack], time: float) -> float | None:
        result = self.intersect(tracks, time)
        return None if result is None else result[1]

    def track_residual(self, estimate: Estimate, track: SensorTrack) -> float:
        """Mahalanobis distance between the bearing the estimate predicts from
        the track's robot and the track's own bearing, under both
        uncertainties. A raw miss distance would let a well-localized target's
        hypothesis claim any line of sight passing a few meters from it, e.g.
        a new target directly behind it."""
        origin = self.robot_position(track.sensor_id, estimate.time)
        offset = estimate.mean[:2] - origin
        squared_range = float(offset @ offset)
        if squared_range == 0.0:
            return np.inf
        predicted = np.arctan2(offset[1], offset[0])
        bearing, rate = track.local_mean
        dt = estimate.time - track.time
        miss = wrap(bearing + rate * dt - predicted)
        # Bearing Jacobian of the estimate's position, and the track's bearing
        # variance extrapolated to the estimate's time.
        jacobian = np.array([-offset[1], offset[0]]) / squared_range
        variance = jacobian @ estimate.covariance[:2, :2] @ jacobian
        if track.local_covariance is not None:
            transition = np.array([1.0, dt])
            variance += transition @ track.local_covariance @ transition
        return float(abs(miss) / np.sqrt(variance))

    def initialize(
        self, tracks: Sequence[SensorTrack], time: float
    ) -> UnscentedFilter | None:
        result = self.intersect(tracks, time)
        if result is None:
            return None
        point, _ = result
        measurements = sorted(
            (
                m
                for t in tracks
                for m in t.measurements
                if time - self.history <= m.time <= time
            ),
            key=lambda m: m.time,
        )
        if not measurements:
            return None
        # The intersection is placed at the first measurement time with a loose
        # covariance; the zip through the history pulls it into place.
        initial = Estimate(
            time=measurements[0].time,
            mean=np.r_[point, 0.0, 0.0],
            covariance=np.diag(self.initial_sigmas**2),
        )
        filter = UnscentedFilter(
            self.dynamics, self.model, initial, history_len=self.filter_history
        )
        filter.update(measurements)
        distances = [
            innovation_distance(ll)
            for t in tracks
            for ll in filter.log_likelihoods(t.id)
        ]
        if not distances or np.mean(distances) > self.innovation_gate:
            return None
        return filter


class BearingsFeasibility:
    """Prunes hypotheses with implausible speed or a position that bearings
    alone no longer constrain (e.g. after losing stereo geometry)."""

    def __init__(self, max_speed: float, max_position_sigma: float):
        self.max_speed = max_speed
        self.max_position_sigma = max_position_sigma

    def is_feasible(self, estimate: Estimate) -> bool:
        speed = float(np.linalg.norm(estimate.mean[2:]))
        position_sigma = float(
            np.sqrt(np.linalg.eigvalsh(estimate.covariance[:2, :2])[-1])
        )
        return speed <= self.max_speed and position_sigma <= self.max_position_sigma
