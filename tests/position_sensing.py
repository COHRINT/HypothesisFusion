"""Position-fix sensors for exercising the core node in tests.

Sensors here measure a target's position directly, which keeps expected
behavior easy to reason about and shows the core algorithm is not tied to
bearings (it still requires two sensor tracks per hypothesis).
"""

import numpy as np

from hypothesis_fusion.filtering import UnscentedFilter
from hypothesis_fusion.types import Estimate, SensorTrackRelease
from tests.fakes import (
    ConstantVelocity2D,
    PositionModel,
    cv_truth,
    position_measurement,
)


class PositionInitializer:
    """Initializer for sensors that measure position directly.

    A sensor track's local mean is its latest ``[x, y, vx, vy]`` fix; the
    residual of a set of tracks is the RMS distance of their fixes (propagated
    to a common time) from their centroid.
    """

    def __init__(self, history: float = 10.0, gate: float = 3.0):
        self.dynamics = ConstantVelocity2D()
        self.model = PositionModel()
        self.history = history
        self.gate = gate

    def _position(self, track, time):
        return track.local_mean[:2] + track.local_mean[2:] * (time - track.time)

    def residual(self, tracks, time):
        points = np.array([self._position(t, time) for t in tracks])
        return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))

    def track_residual(self, estimate, track):
        return float(
            np.linalg.norm(estimate.mean[:2] - self._position(track, estimate.time))
        )

    def initialize(self, tracks, time):
        measurements = sorted(
            (
                m
                for t in tracks
                for m in t.measurements
                if time - m.time <= self.history
            ),
            key=lambda m: m.time,
        )
        if not measurements:
            return None
        start = measurements[0].time
        center = np.mean([self._position(t, start) for t in tracks], axis=0)
        f = UnscentedFilter(
            self.dynamics,
            self.model,
            Estimate(
                start, np.r_[center, 0.0, 0.0], np.diag([100.0, 100.0, 10.0, 10.0])
            ),
        )
        f.update(measurements)
        distances = [np.sqrt(-2 * ll) for t in tracks for ll in f.log_likelihoods(t.id)]
        if np.mean(distances) > self.gate:
            return None
        return f


class PositionSensing:
    """Every sensor perfectly tracks every target, releasing each step.

    Sensor track ``sensor * 100 + target`` observes target ``target``; a
    ``switch`` of ``(track_id, time, new_target)`` makes that sensor track start
    following another target at ``time``.
    """

    def __init__(self, targets, sensors, sigma=0.1, seed=0, switch=None):
        self.targets = [np.asarray(t, dtype=float) for t in targets]
        self.sensors = list(sensors)
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.switch = switch

    def track_id(self, sensor, target):
        return sensor * 100 + target

    def releases(self, time, terminated=()):
        out = []
        for sensor in self.sensors:
            for target in range(len(self.targets)):
                track_id = self.track_id(sensor, target)
                followed = target
                if (
                    self.switch
                    and self.switch[0] == track_id
                    and time >= self.switch[1]
                ):
                    followed = self.switch[2]
                truth = cv_truth(self.targets[followed], time)
                meas = position_measurement(
                    truth, time, track_id, sensor, self.sigma, self.rng
                )
                out.append(
                    SensorTrackRelease(
                        sensor_track_id=track_id,
                        sensor_id=sensor,
                        time=time,
                        measurements=(meas,),
                        local_mean=np.r_[meas.z, truth[2:]],
                        terminated=track_id in terminated,
                    )
                )
        return out
