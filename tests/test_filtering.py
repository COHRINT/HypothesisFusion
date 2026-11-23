import numpy as np
import pytest

from hypothesis_fusion.filtering import UnscentedFilter, project_to_spd
from hypothesis_fusion.types import Estimate, Measurement
from tests.fakes import (
    AngleModel,
    ConstantVelocity2D,
    PositionModel,
    cv_truth,
    position_measurement,
)

X0 = np.array([0.0, 0.0, 1.0, 0.5])


def _filter(history_len: int | None = 50) -> UnscentedFilter:
    return UnscentedFilter(
        ConstantVelocity2D(),
        PositionModel(),
        Estimate(time=0.0, mean=np.zeros(4), covariance=np.eye(4) * 100.0),
        history_len=history_len,
    )


def _measurements(n: int, seed: int = 0, track_ids=(1, 2)) -> list[Measurement]:
    rng = np.random.default_rng(seed)
    return [
        position_measurement(
            cv_truth(X0, 0.5 * k),
            0.5 * k,
            sensor_track_id=track_ids[k % len(track_ids)],
            sensor_id=track_ids[k % len(track_ids)],
            rng=rng,
        )
        for k in range(1, n + 1)
    ]


def test_converges_on_constant_velocity_target():
    f = _filter()
    f.update(_measurements(40))
    truth = cv_truth(X0, 20.0)
    assert f.latest.time == 20.0
    assert np.linalg.norm(f.latest.mean[:2] - truth[:2]) < 0.2
    assert np.linalg.norm(f.latest.mean[2:] - truth[2:]) < 0.25


def test_out_of_sequence_matches_in_sequence():
    meas = _measurements(20)
    in_order = _filter()
    in_order.update(meas)

    shuffled = _filter()
    order = np.random.default_rng(1).permutation(len(meas))
    for i in order:
        shuffled.update([meas[i]])

    np.testing.assert_allclose(shuffled.latest.mean, in_order.latest.mean)
    np.testing.assert_allclose(shuffled.latest.covariance, in_order.latest.covariance)
    assert shuffled.log_likelihoods(1) == pytest.approx(in_order.log_likelihoods(1))
    assert [m.id for m in shuffled.measurements] == [
        m.id for m in in_order.measurements
    ]


def test_measurement_older_than_history_is_dropped():
    meas = _measurements(10)
    f = _filter(history_len=3)
    f.update(meas[1:])
    before = f.latest
    f.update([meas[0]])
    assert f.dropped == 1
    np.testing.assert_array_equal(f.latest.mean, before.mean)


def test_duplicate_measurements_are_ignored():
    meas = _measurements(5)
    f = _filter()
    f.update(meas)
    before = f.latest
    f.update(meas)
    assert f.latest is before
    assert len(f.log_likelihoods(1)) == 2


def test_without_measurements_matches_never_seeing_them():
    meas = _measurements(20)
    drop = {m.id for m in meas if m.sensor_track_id == 2 and m.time > 6.0}

    full = _filter()
    full.update(meas)
    undone = full.without_measurements(drop)

    never = _filter()
    never.update([m for m in meas if m.id not in drop])

    np.testing.assert_allclose(undone.latest.mean, never.latest.mean)
    assert undone.log_likelihoods(2) == pytest.approx(never.log_likelihoods(2))
    # The original filter is unchanged.
    assert len(full.measurements) == 20


def test_likelihoods_are_kept_per_sensor_track():
    f = _filter()
    f.update(_measurements(9, track_ids=(1, 2, 3)))
    assert [len(f.log_likelihoods(i)) for i in (1, 2, 3, 4)] == [3, 3, 3, 0]
    assert all(ll <= 0.0 for ll in f.log_likelihoods(1))


def test_predict_backwards_adds_no_process_noise():
    f = _filter()
    f.update(_measurements(10))
    back = f.predict(f.latest.time - 1.0)
    forward = f.predict(f.latest.time + 1.0)
    assert np.trace(back.covariance) < np.trace(forward.covariance)
    np.testing.assert_allclose(back.mean[:2], f.latest.mean[:2] - f.latest.mean[2:])


def test_angle_residual_wraps_across_pi():
    # A target just across the -x axis from the measured bearing.
    f = UnscentedFilter(
        ConstantVelocity2D(),
        AngleModel(),
        Estimate(0.0, np.array([-10.0, 0.01, 0.0, 0.0]), np.eye(4) * 0.01),
    )
    z = np.array([-np.pi + 0.0005])
    f.update([Measurement(1, 1, 1, 0.0, z, np.eye(1) * 1e-6)])
    # Wrapping keeps the innovation tiny instead of ~2 pi.
    assert f.log_likelihoods(1)[0] > -10.0


def test_project_to_spd_clips_negative_eigenvalues():
    P = project_to_spd(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert np.all(np.linalg.eigvalsh(P) > 0)
