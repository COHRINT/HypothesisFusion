"""Hypothesis scoring.

For each active sensor track ``s`` of hypothesis ``h``, the hypothesis filter
records the innovation log-likelihood ``-0.5 nu^T W^-1 nu`` of every update
driven by ``s``. The sensor track's likelihood ``L(s)`` is the average over its
most recent ``window`` updates, and the hypothesis score is::

    score(h) = mean_{s in S_a(h)} L(s) - 0.5 log|P| + beta * |S(h)|

where ``P`` is the hypothesis' latest covariance, ``S_a(h)`` its active sensor
tracks and ``S(h)`` its full sensor track pedigree. Averaging over sensor tracks
debiases hypotheses with respect to their size; the linear size bias ``beta``
then deliberately favors one well-explained hypothesis over several hypotheses
that split the same sources.
"""

from collections.abc import Set

import numpy as np

from hypothesis_fusion.filtering import HypothesisFilter


def innovation_distance(log_likelihood: float) -> float:
    """Mahalanobis innovation distance ``sqrt(nu^T W^-1 nu)`` from an innovation
    log-likelihood. A windowed average maps to the RMS distance."""
    return float(np.sqrt(-2.0 * log_likelihood))


class WindowedScorer:
    """Scores hypotheses from windowed per-sensor-track innovation likelihoods."""

    def __init__(self, window: int, size_bias: float):
        if window < 1:
            raise ValueError(f'window must be >= 1, got {window}')
        self.window = window
        self.size_bias = size_bias

    def sensor_track_likelihoods(
        self, filter: HypothesisFilter, sensor_track_ids: Set[int]
    ) -> dict[int, float]:
        """``L(s)`` for each given sensor track with any innovation history."""
        likelihoods: dict[int, float] = {}
        for sensor_track_id in sensor_track_ids:
            history = filter.log_likelihoods(sensor_track_id)[-self.window :]
            if len(history):
                likelihoods[sensor_track_id] = float(np.mean(history))
        return likelihoods

    def score(
        self,
        filter: HypothesisFilter,
        sensor_track_ids: Set[int],
        active_sensor_track_ids: Set[int],
    ) -> float | None:
        """The hypothesis score, or ``None`` while it is undefined (fewer than two
        of its sensor tracks have updated the filter, i.e. no stereo view yet)."""
        if len(self.sensor_track_likelihoods(filter, sensor_track_ids)) < 2:
            return None

        covariance = filter.latest.covariance
        if not np.all(np.linalg.eigvalsh(covariance) > 0):
            return float('-inf')
        score = -0.5 * float(np.linalg.slogdet(covariance)[1])

        # Innovation term over active tracks only, so ended sensor tracks do not
        # weigh on the score with stale information. It may be empty when every
        # track has gone inactive; the hypothesis then rides on its size bias.
        active = list(
            self.sensor_track_likelihoods(filter, active_sensor_track_ids).values()
        )
        if active:
            score += float(np.mean(active))

        # The size bias counts the full pedigree, so hypotheses that explain
        # more of a target's history are favored.
        score += self.size_bias * len(sensor_track_ids)
        return float(score)
