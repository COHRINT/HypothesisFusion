import math

import numpy as np
import pytest

from hypothesis_fusion.scoring import WindowedScorer, innovation_distance
from hypothesis_fusion.types import Estimate


class StubFilter:
    def __init__(self, likelihoods: dict[int, list[float]], covariance=None):
        self._likelihoods = likelihoods
        cov = np.eye(2) if covariance is None else covariance
        self.latest = Estimate(0.0, np.zeros(2), cov)

    def log_likelihoods(self, sensor_track_id):
        return self._likelihoods.get(sensor_track_id, [])


def test_windowed_average_per_track_then_across_tracks():
    scorer = WindowedScorer(window=2, size_bias=0.0)
    f = StubFilter({1: [-100.0, -1.0, -3.0], 2: [-4.0]})
    # L(1) = mean(-1, -3) = -2, L(2) = -4; identity covariance has logdet 0.
    assert scorer.score(f, {1, 2}, {1, 2}) == pytest.approx(-3.0)


def test_undefined_until_two_tracks_have_innovations():
    scorer = WindowedScorer(window=3, size_bias=5.0)
    assert scorer.score(StubFilter({1: [-1.0]}), {1, 2}, {1, 2}) is None


def test_innovation_term_uses_active_tracks_only():
    scorer = WindowedScorer(window=3, size_bias=0.0)
    f = StubFilter({1: [-1.0], 2: [-9.0]})
    assert scorer.score(f, {1, 2}, {1}) == pytest.approx(-1.0)


def test_size_bias_counts_full_pedigree():
    scorer = WindowedScorer(window=3, size_bias=5.0)
    f = StubFilter({1: [-1.0], 2: [-1.0]})
    assert scorer.score(f, {1, 2, 3}, {1, 2}) == pytest.approx(-1.0 + 15.0)


def test_no_active_tracks_keeps_size_bias():
    scorer = WindowedScorer(window=3, size_bias=5.0)
    f = StubFilter({1: [-1.0], 2: [-1.0]})
    assert scorer.score(f, {1, 2}, set()) == pytest.approx(10.0)


def test_uncertainty_is_penalized():
    scorer = WindowedScorer(window=3, size_bias=0.0)
    f = StubFilter({1: [0.0], 2: [0.0]}, covariance=np.eye(2) * math.e**2)
    assert scorer.score(f, {1, 2}, {1, 2}) == pytest.approx(-2.0)


def test_non_spd_covariance_scores_negative_infinity():
    scorer = WindowedScorer(window=3, size_bias=0.0)
    f = StubFilter({1: [0.0], 2: [0.0]}, covariance=-np.eye(2))
    assert scorer.score(f, {1, 2}, {1, 2}) == float('-inf')


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        WindowedScorer(window=0, size_bias=0.0)


def test_innovation_distance():
    assert innovation_distance(-0.5 * 9.0) == pytest.approx(3.0)
