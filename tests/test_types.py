import numpy as np

from hypothesis_fusion.types import Measurement, SensorTrack, SensorTrackRelease


def _meas(mid: int, t: float) -> Measurement:
    return Measurement(
        id=mid,
        sensor_track_id=7,
        sensor_id=1,
        time=t,
        z=np.zeros(1),
        R=np.eye(1),
    )


def _release(t: float, *meas: Measurement) -> SensorTrackRelease:
    return SensorTrackRelease(
        sensor_track_id=7, sensor_id=1, time=t, measurements=tuple(meas)
    )


def test_absorb_appends_new_measurements_in_time_order():
    track = SensorTrack.from_release(_release(1.0, _meas(1, 1.0)))
    track.absorb(_release(3.0, _meas(3, 3.0)))
    track.absorb(_release(2.0, _meas(2, 2.0)))
    assert [m.id for m in track.measurements] == [1, 2, 3]
    # An older release does not roll back the latest local state time.
    assert track.time == 3.0


def test_absorb_ignores_duplicate_measurements():
    track = SensorTrack.from_release(_release(1.0, _meas(1, 1.0)))
    track.absorb(_release(1.0, _meas(1, 1.0)))
    assert len(track.measurements) == 1


def test_trimmed_keeps_at_least_latest_measurement():
    track = SensorTrack.from_release(_release(2.0, _meas(1, 1.0), _meas(2, 2.0)))
    assert [m.id for m in track.trimmed(1.5).measurements] == [2]
    assert [m.id for m in track.trimmed(9.0).measurements] == [2]
    # The original is untouched.
    assert len(track.measurements) == 2
