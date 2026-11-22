"""Application models for :class:`~hypothesis_fusion.filtering.UnscentedFilter`.

The hypothesis filter tracks targets in a common state space ``x`` (for
example position and velocity in a global frame) from measurements expressed
in each sensor's local frame.
"""

from typing import Protocol

import numpy as np
from numpy import typing as npt

from hypothesis_fusion.types import Measurement


class DynamicsModel(Protocol):
    """Linear(ized) target dynamics ``x_{k+1} = F x_k + w_k, w_k ~ N(0, Q)``."""

    def transition_matrix(
        self, dt: float, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """State transition matrix over ``dt`` seconds (``dt`` may be negative)."""
        ...

    def process_noise(self, dt: float) -> npt.NDArray[np.float64]:
        """Process noise covariance accumulated over ``dt > 0`` seconds."""
        ...


class MeasurementModel(Protocol):
    """Nonlinear measurement model ``z = h(x) + v, v ~ N(0, R)``."""

    def predict(
        self, x: npt.NDArray[np.float64], measurement: Measurement
    ) -> npt.NDArray[np.float64]:
        """Predicted measurement of state ``x`` by the sensor that took
        ``measurement`` (use ``measurement.sensor_state`` for its pose)."""
        ...

    def residual(
        self, z: npt.NDArray[np.float64], z_pred: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``z - z_pred``, wrapped as needed (e.g. angles into ``[-pi, pi)``)."""
        ...
