from examples.bearings2d.scenario import default_scenario
from examples.bearings2d.sensing import Sensing


def test_robots_sample_once_per_period_with_staggered_phases():
    scenario = default_scenario()
    first = scenario.sample_times()[: len(scenario.robots)]
    times = [t for t, _ in first]
    assert [r for _, r in first] == [r.id for r in scenario.robots]
    assert times == sorted(set(times))
    assert times[-1] == scenario.dt
    assert len(scenario.sample_times()) == len(scenario.robots) * len(scenario.times)


def test_releases_are_from_one_robot_and_tracks_persist():
    scenario = default_scenario()
    sensing = Sensing(scenario)
    first = sensing.releases(1, 0.5)
    second = sensing.releases(1, 1.0)
    assert first and {r.sensor_id for r in first + second} == {1}
    assert [r.sensor_track_id for r in first] == [r.sensor_track_id for r in second]
