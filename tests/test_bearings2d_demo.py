"""End-to-end runs of the 2D bearings demo."""

import pytest

from examples.bearings2d import simulation
from examples.bearings2d.metrics import evaluate
from examples.bearings2d.scenario import default_scenario


def _assert_one_pure_track_per_target(result):
    labels = result.target_of_sensor_track
    final = [h for node in result.nodes for h in node.released_tracks.values()]
    targets = sorted(
        {labels[t] for t in h.sensor_track_ids}.pop()
        for h in final
        if len({labels[t] for t in h.sensor_track_ids}) == 1
    )
    assert len(final) == len(result.scenario.targets)
    assert targets == [t.id for t in result.scenario.targets]


def test_centralized_demo_tracks_every_target():
    result = simulation.run(default_scenario(), simulation.CENTRALIZED_ROUTES)
    summary = evaluate(result).summary()
    assert summary['rmse_m (settled)'] < 6.0
    assert summary['custody (settled)'] > 0.95
    assert summary['ambiguity (settled)'] == pytest.approx(1.0, abs=0.05)
    assert summary['purity'] > 0.9
    _assert_one_pure_track_per_target(result)


def test_decentralized_demo_tracks_every_target():
    result = simulation.run(default_scenario(), simulation.DECENTRALIZED_ROUTES)
    summary = evaluate(result).summary()
    assert summary['rmse_m (settled)'] < 6.0
    assert summary['custody (settled)'] > 0.9
    assert summary['ambiguity (settled)'] == pytest.approx(1.0, abs=0.05)
    assert summary['purity'] > 0.9


def test_decentralized_demo_deduplicates_across_nodes():
    result = simulation.run(default_scenario(), simulation.DECENTRALIZED_ROUTES)
    _assert_one_pure_track_per_target(result)
    # No sensor track is in use at two fusion nodes.
    in_use = [
        set().union(
            *(node.active_sensor_track_ids(h) for h in node.hypotheses.values())
        )
        for node in result.nodes
    ]
    for i, a in enumerate(in_use):
        for b in in_use[i + 1 :]:
            assert not a & b


def test_out_of_sequence_releases_are_absorbed():
    result = simulation.run(
        default_scenario(), simulation.CENTRALIZED_ROUTES, max_release_delay=0.25
    )
    summary = evaluate(result).summary()
    assert summary['rmse_m (settled)'] < 8.0
    _assert_one_pure_track_per_target(result)
