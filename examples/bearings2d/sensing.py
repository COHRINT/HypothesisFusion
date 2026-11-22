"""The sensing layer: each robot's sensor tracks.

Sensor tracks here are *perfect* in association -- each robot produces exactly
one sensor track per target in range, and every measurement in it comes from
that target -- which is the idealization of MFI Sec. II-B's assumptions. Each
sensor track runs a small constant-velocity Kalman filter on its (unwrapped)
bearing to provide the local estimate ``[bearing, bearing rate]`` that
:class:`~examples.bearings2d.initializer.BearingsInitializer` uses to relate
sensor tracks. A target leaving a robot's sensing range terminates the track;
re-entering starts a new one.
"""

import dataclasses
import itertools

import numpy as np

from examples.bearings2d.models import wrap
from examples.bearings2d.scenario import Scenario
from hypothesis_fusion.types import Measurement, SensorTrackRelease


@dataclasses.dataclass
class _LocalTrack:
    id: int
    mean: np.ndarray  # [bearing, bearing rate], bearing unwrapped
    covariance: np.ndarray
    time: float


class Sensing:
    """Generates the robots' sensor track releases, one sample at a time."""

    def __init__(self, scenario: Scenario, seed: int = 0, bearing_rate_q: float = 1e-5):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.q = bearing_rate_q
        self._measurement_ids = itertools.count(1)
        self._track_ids = itertools.count(1)
        self._tracks: dict[tuple[int, int], _LocalTrack] = {}
        # Ground truth labels, for evaluation only.
        self.target_of_sensor_track: dict[int, int] = {}

    def releases(self, robot_id: int, time: float) -> list[SensorTrackRelease]:
        """Releases of robot ``robot_id``'s sensor tracks from a sample at ``time``."""
        out: list[SensorTrackRelease] = []
        sigma = self.scenario.bearing_sigma
        robot = self.scenario.robot(robot_id)
        position = robot.position(time)
        for target in self.scenario.targets:
            key = (robot.id, target.id)
            offset = target.state(time)[:2] - position
            in_view = np.linalg.norm(offset) <= self.scenario.sensing_range
            if not (target.present(time) and in_view):
                if (ended := self._tracks.pop(key, None)) is not None:
                    out.append(self._terminate(ended, robot.id, time))
                continue

            bearing = float(wrap(np.arctan2(offset[1], offset[0])))
            z = float(wrap(bearing + self.rng.normal(0.0, sigma)))
            track = self._tracks.get(key)
            if track is None:
                track = _LocalTrack(
                    next(self._track_ids),
                    np.array([z, 0.0]),
                    np.diag([sigma**2, 0.1**2]),
                    time,
                )
                self._tracks[key] = track
                self.target_of_sensor_track[track.id] = target.id
            else:
                self._local_update(track, z, time, sigma)

            measurement = Measurement(
                id=next(self._measurement_ids),
                sensor_track_id=track.id,
                sensor_id=robot.id,
                time=time,
                z=np.array([z]),
                R=np.array([[sigma**2]]),
                sensor_state=position,
            )
            out.append(
                SensorTrackRelease(
                    sensor_track_id=track.id,
                    sensor_id=robot.id,
                    time=time,
                    measurements=(measurement,),
                    local_mean=np.array([wrap(track.mean[0]), track.mean[1]]),
                    local_covariance=track.covariance.copy(),
                )
            )
        return out

    def _local_update(self, track: _LocalTrack, z: float, time: float, sigma: float):
        dt = time - track.time
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        mean = F @ track.mean
        P = F @ track.covariance @ F.T + Q
        innovation = float(wrap(z - mean[0]))
        S = P[0, 0] + sigma**2
        K = P[:, 0] / S
        track.mean = mean + K * innovation
        track.covariance = P - np.outer(K, P[0, :])
        track.time = time

    @staticmethod
    def _terminate(
        track: _LocalTrack, robot_id: int, time: float
    ) -> SensorTrackRelease:
        return SensorTrackRelease(
            sensor_track_id=track.id,
            sensor_id=robot_id,
            time=track.time,
            measurements=(),
            local_mean=np.array([wrap(track.mean[0]), track.mean[1]]),
            local_covariance=track.covariance.copy(),
            terminated=True,
        )
