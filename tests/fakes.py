"""Small application models used by the core tests.

These deliberately avoid bearings: sensors here measure a target's position
directly (with noise), which keeps the expected behavior easy to reason about
and shows the core algorithm is not tied to a particular sensor type.
"""

import itertools

import numpy as np

from hypothesis_fusion.types import Measurement

_ids = itertools.count(1)


class ConstantVelocity2D:
    """State ``[x, y, vx, vy]`` with white-noise acceleration."""

    def __init__(self, q: float = 0.01):
        self.q = q

    def transition_matrix(self, dt, x):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def process_noise(self, dt):
        q = self.q
        block = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        return Q


class PositionModel:
    """``z = [x, y] + v``."""

    def predict(self, x, measurement):
        return x[:2]

    def residual(self, z, z_pred):
        return z - z_pred


class AngleModel:
    """``z = atan2(y, x)`` from the origin, wrapped into ``[-pi, pi)``."""

    def predict(self, x, measurement):
        return np.array([np.arctan2(x[1], x[0])])

    def residual(self, z, z_pred):
        return (z - z_pred + np.pi) % (2 * np.pi) - np.pi


def position_measurement(
    truth: np.ndarray,
    time: float,
    sensor_track_id: int,
    sensor_id: int,
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Measurement:
    noise = rng.normal(0.0, sigma, 2) if rng is not None else np.zeros(2)
    return Measurement(
        id=next(_ids),
        sensor_track_id=sensor_track_id,
        sensor_id=sensor_id,
        time=time,
        z=np.asarray(truth[:2], dtype=float) + noise,
        R=np.eye(2) * sigma**2,
    )


def cv_truth(x0: np.ndarray, time: float) -> np.ndarray:
    """Truth of a constant-velocity target ``[x, y, vx, vy]`` at ``time``."""
    return np.array(
        [x0[0] + x0[2] * time, x0[1] + x0[3] * time, x0[2], x0[3]], dtype=float
    )
