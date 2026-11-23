"""Hypothesis filters.

Each hypothesis models every measurement from its sensor tracks as originating
from one common target and runs a filter over them. The core algorithm only
needs the :class:`HypothesisFilter` interface: the per-sensor-track history of
innovation log-likelihoods drives scoring, and rewinding drives hypothesis
shrinking. :class:`UnscentedFilter` is a ready-made implementation (linear
dynamics, unscented measurement update) for application models that satisfy
:mod:`hypothesis_fusion.models`.

Out-of-sequence measurements are handled by rewind-and-replay over a bounded
history: a late measurement rewinds the filter to just before its capture time,
is applied, and the retained later measurements are replayed. Measurements
older than the retained history are dropped (counted in ``dropped``).
"""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from typing import Protocol, Self

import numpy as np
from numpy import typing as npt

from hypothesis_fusion.models import DynamicsModel, MeasurementModel
from hypothesis_fusion.types import Estimate, Measurement


class HypothesisFilter(Protocol):
    """The filter interface a hypothesis needs."""

    @property
    def latest(self) -> Estimate:
        """The most recent posterior estimate."""
        ...

    @property
    def measurements(self) -> Sequence[Measurement]:
        """Retained processed measurements, oldest first."""
        ...

    def update(self, measurements: Iterable[Measurement]) -> None:
        """Process measurements (any order; already-processed IDs are ignored)."""
        ...

    def predict(self, time: float) -> Estimate:
        """The estimate predicted to ``time`` (may be in the past), without
        changing the filter."""
        ...

    def log_likelihoods(self, sensor_track_id: int) -> Sequence[float]:
        """Innovation log-likelihoods ``-0.5 nu^T W^-1 nu`` of the updates from
        one sensor track, oldest first."""
        ...

    def without_measurements(self, measurement_ids: set[int]) -> Self:
        """A copy with the given measurements' updates undone (only those within
        the retained history can be undone)."""
        ...

    def copy(self) -> Self:
        """An independent deep copy."""
        ...


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


def project_to_spd(
    covariance: npt.NDArray[np.float64], min_eigenvalue: float = 1e-9
) -> npt.NDArray[np.float64]:
    """Nearest symmetric matrix with eigenvalues clipped from below."""
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    projected = eigvecs @ np.diag(np.clip(eigvals, min_eigenvalue, None)) @ eigvecs.T
    return 0.5 * (projected + projected.T)


def stable_cholesky(covariance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lower Cholesky factor, adding jitter as needed."""
    covariance = project_to_spd(covariance)
    jitter = 1e-9
    identity = np.eye(covariance.shape[0])
    for _ in range(9):
        try:
            return np.linalg.cholesky(covariance + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(f'Cholesky failed on covariance:\n{covariance}')


def sigma_points(
    mean: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Scaled unscented sigma points with their mean and covariance weights."""
    n = mean.shape[0]
    lam = alpha**2 * (n + kappa) - n
    scale = n + lam
    if scale <= 0.0:
        raise ValueError(f'Invalid UKF scaling: n + lambda = {scale} <= 0')
    offsets = np.sqrt(scale) * stable_cholesky(covariance).T
    points = np.vstack([mean, mean + offsets, mean - offsets])
    w_m = np.full(2 * n + 1, 1.0 / (2.0 * scale))
    w_c = w_m.copy()
    w_m[0] = lam / scale
    w_c[0] = lam / scale + (1.0 - alpha**2 + beta)
    return points, w_m, w_c


# --------------------------------------------------------------------------
# Unscented filter
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Step:
    """One filter step: the posterior after applying ``measurement``."""

    estimate: Estimate
    measurement: Measurement | None  # None for the initial estimate
    log_likelihood: float | None


class UnscentedFilter:
    """UKF with linear dynamics and an unscented measurement update.

    ``history_len`` bounds the number of retained steps (and so how late an
    out-of-sequence measurement can be absorbed); ``None`` retains everything.
    Innovation log-likelihoods are retained separately, up to
    ``likelihood_len`` per sensor track, so scoring windows are unaffected by
    the step history bound.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        measurement_model: MeasurementModel,
        initial: Estimate,
        history_len: int | None = 50,
        likelihood_len: int = 32,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        if history_len is not None and history_len < 1:
            raise ValueError(f'history_len must be >= 1, got {history_len}')
        self._dynamics = dynamics
        self._measurement_model = measurement_model
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._steps: deque[_Step] = deque([_Step(initial, None, None)], history_len)
        self._processed: set[int] = set()
        self._likelihood_len = likelihood_len
        # Sensor track ID -> [(measurement ID, log-likelihood)], oldest first.
        self._likelihoods: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self._likelihood_len)
        )
        self.dropped = 0

    # ----------------------------------------------------------------- views

    @property
    def latest(self) -> Estimate:
        return self._steps[-1].estimate

    @property
    def estimates(self) -> list[Estimate]:
        """Retained posterior estimates, oldest first."""
        return [step.estimate for step in self._steps]

    @property
    def measurements(self) -> list[Measurement]:
        return [s.measurement for s in self._steps if s.measurement is not None]

    def log_likelihoods(self, sensor_track_id: int) -> list[float]:
        history = self._likelihoods.get(sensor_track_id)
        return [ll for _, ll in history] if history else []

    def copy(self) -> Self:
        return copy.deepcopy(self)

    # ------------------------------------------------------------ prediction

    def predict(self, time: float) -> Estimate:
        return self._predict(self.latest, time)

    def _predict(self, estimate: Estimate, time: float) -> Estimate:
        dt = time - estimate.time
        if dt == 0.0:
            return estimate
        F = self._dynamics.transition_matrix(dt, estimate.mean)
        P = F @ estimate.covariance @ F.T
        if dt > 0.0:
            # Process noise only accrues forward in time.
            P = P + self._dynamics.process_noise(dt)
        return Estimate(time=time, mean=F @ estimate.mean, covariance=P)

    # ---------------------------------------------------------------- update

    def update(self, measurements: Iterable[Measurement]) -> None:
        for measurement in sorted(measurements, key=lambda m: m.time):
            if measurement.id in self._processed:
                continue
            if measurement.time >= self.latest.time:
                self._apply(measurement)
            elif measurement.time <= self._steps[0].estimate.time:
                # Older than the retained history: cannot be applied in sequence.
                self.dropped += 1
            else:
                replay = self._rewind_after(measurement.time)
                self._apply(measurement)
                for later in replay:
                    self._apply(later)

    def _apply(self, measurement: Measurement) -> None:
        prior = self._predict(self.latest, measurement.time)
        posterior, log_likelihood = self._ukf_update(prior, measurement)
        self._steps.append(_Step(posterior, measurement, log_likelihood))
        self._processed.add(measurement.id)
        self._likelihoods[measurement.sensor_track_id].append(
            (measurement.id, log_likelihood)
        )

    def _rewind_after(self, time: float) -> list[Measurement]:
        """Undo every step strictly after ``time``; return their measurements."""
        index = 1
        while index < len(self._steps) and self._steps[index].estimate.time <= time:
            index += 1
        return self._pop_from(index)

    def _pop_from(self, index: int) -> list[Measurement]:
        """Undo the steps at ``index`` and later (``index >= 1``, so the oldest
        retained step always remains); return their measurements in order."""
        popped: list[Measurement] = []
        while len(self._steps) > max(index, 1):
            step = self._steps.pop()
            assert step.measurement is not None
            popped.append(step.measurement)
            self._processed.discard(step.measurement.id)
            history = self._likelihoods[step.measurement.sensor_track_id]
            if history and history[-1][0] == step.measurement.id:
                history.pop()
        popped.reverse()
        return popped

    def without_measurements(self, measurement_ids: set[int]) -> Self:
        new = self.copy()
        # The oldest retained step is folded into everything after it, so only
        # later steps can be undone.
        first = next(
            (
                i
                for i, step in enumerate(new._steps)
                if i > 0
                and step.measurement is not None
                and step.measurement.id in measurement_ids
            ),
            None,
        )
        if first is None:
            return new
        for measurement in new._pop_from(first):
            if measurement.id not in measurement_ids:
                new._apply(measurement)
        return new

    def _ukf_update(
        self, prior: Estimate, measurement: Measurement
    ) -> tuple[Estimate, float]:
        x, P = prior.mean, prior.covariance
        points, w_m, w_c = sigma_points(x, P, self._alpha, self._beta, self._kappa)
        model = self._measurement_model
        z_sigma = [model.predict(point, measurement) for point in points]
        # Average residuals about the central sigma point rather than raw
        # predictions, so wrapped quantities (angles) average correctly.
        z_ref = z_sigma[0]
        dz = np.array([model.residual(z, z_ref) for z in z_sigma])
        mean_offset = w_m @ dz
        dz = dz - mean_offset
        innovation = model.residual(measurement.z, z_ref) - mean_offset

        dx = points - x
        P_zz = (w_c[:, None] * dz).T @ dz + measurement.R
        P_zz = project_to_spd(P_zz)
        P_xz = (w_c[:, None] * dx).T @ dz
        K = np.linalg.solve(P_zz, P_xz.T).T

        mean = x + K @ innovation
        covariance = project_to_spd(P - K @ P_zz @ K.T)
        log_likelihood = float(-0.5 * innovation @ np.linalg.solve(P_zz, innovation))
        return Estimate(prior.time, mean, covariance), log_likelihood
