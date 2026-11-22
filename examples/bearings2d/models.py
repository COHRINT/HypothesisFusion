"""Planar target dynamics and bearing measurements.

State: ``[x, y, vx, vy]`` in meters and meters/second in a shared world frame.
Measurement: the bearing (radians, counterclockwise from +x) from the robot at
``measurement.sensor_state`` (its ``[x, y]`` position at capture time).
"""

import numpy as np


def wrap(angle):
    """Wrap angles into ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class ConstantVelocity:
    """Nearly constant velocity with white-noise acceleration of spectral density
    ``q`` (m^2/s^3)."""

    def __init__(self, q: float = 0.5):
        self.q = q

    def transition_matrix(self, dt, x):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def process_noise(self, dt):
        block = self.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        return Q


class BearingModel:
    def predict(self, x, measurement):
        dx, dy = x[:2] - measurement.sensor_state
        return np.array([np.arctan2(dy, dx)])

    def residual(self, z, z_pred):
        return wrap(z - z_pred)
