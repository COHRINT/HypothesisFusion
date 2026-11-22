"""Core data types shared across the package.

Times are floats in seconds. All identifiers are integers: sensor IDs, sensor
track IDs, and measurement IDs must be unique across the whole sensor network
(sensor track IDs are the "UUIDs" the distributed protocol reasons over).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy import typing as npt


@dataclasses.dataclass(frozen=True, eq=False)
class Measurement:
    """A single measurement released by a sensor track.

    ``sensor_state`` is opaque to the core algorithm; it is whatever the
    application's measurement model needs to predict ``z`` (e.g. the sensor's
    pose at capture time).
    """

    id: int
    sensor_track_id: int
    sensor_id: int
    time: float
    z: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    sensor_state: Any = None


@dataclasses.dataclass(frozen=True, eq=False)
class Estimate:
    """A Gaussian state estimate at a point in time."""

    time: float
    mean: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True, eq=False)
class SensorTrackRelease:
    """An update from a sensor track, as sent from a sensing node to fusion nodes.

    ``measurements`` holds the measurements new since the track's previous
    release. ``local_mean``/``local_covariance`` are the sensor track's own
    filtered estimate in the sensor's local frame (e.g. bearing and bearing
    rate) at ``time``; the core algorithm never interprets them, but the
    application's :class:`~hypothesis_fusion.initialization.Initializer` uses
    them to relate sensor tracks to each other. ``terminated`` marks the final
    release of a sensor track.
    """

    sensor_track_id: int
    sensor_id: int
    time: float
    measurements: tuple[Measurement, ...]
    local_mean: npt.NDArray[np.float64] | None = None
    local_covariance: npt.NDArray[np.float64] | None = None
    terminated: bool = False


@dataclasses.dataclass(eq=False)
class SensorTrack:
    """A fusion node's accumulated view of one sensor track."""

    id: int
    sensor_id: int
    time: float
    measurements: list[Measurement]
    local_mean: npt.NDArray[np.float64] | None = None
    local_covariance: npt.NDArray[np.float64] | None = None

    @classmethod
    def from_release(cls, release: SensorTrackRelease) -> SensorTrack:
        return cls(
            id=release.sensor_track_id,
            sensor_id=release.sensor_id,
            time=release.time,
            measurements=sorted(release.measurements, key=lambda m: m.time),
            local_mean=release.local_mean,
            local_covariance=release.local_covariance,
        )

    def absorb(self, release: SensorTrackRelease) -> None:
        """Fold a newer release of this sensor track into the accumulated view."""
        known = {m.id for m in self.measurements}
        self.measurements.extend(m for m in release.measurements if m.id not in known)
        self.measurements.sort(key=lambda m: m.time)
        if release.time >= self.time:
            self.time = release.time
            self.local_mean = release.local_mean
            self.local_covariance = release.local_covariance

    def trimmed(self, since: float) -> SensorTrack:
        """A copy keeping only measurements at or after ``since`` (at least one)."""
        recent = [m for m in self.measurements if m.time >= since]
        if not recent:
            recent = self.measurements[-1:]
        return dataclasses.replace(self, measurements=list(recent))
