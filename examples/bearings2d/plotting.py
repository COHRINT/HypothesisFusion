"""Plots of a demo run (requires matplotlib: ``uv sync --extra demo``)."""

from collections import defaultdict

import numpy as np

from examples.bearings2d.metrics import Metrics
from examples.bearings2d.simulation import RunResult


def plot_run(result: RunResult, metrics: Metrics, title: str, path: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scenario = result.scenario
    times = scenario.times
    fig, (ax_map, ax_err) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [1.1, 1]}
    )

    for robot in scenario.robots:
        path_xy = np.array([robot.position(t) for t in times])
        ax_map.plot(*path_xy.T, ':', color='0.55', lw=1)
        ax_map.plot(*path_xy[-1], 's', color='0.35', ms=6)
        ax_map.annotate(
            f'R{robot.id}',
            path_xy[-1],
            xytext=(5, 5),
            textcoords='offset points',
            color='0.35',
        )
    for target in scenario.targets:
        alive = times[times >= target.appears]
        truth = np.array([target.state(t)[:2] for t in alive])
        ax_map.plot(*truth.T, color='0.2', lw=3, alpha=0.25)
        ax_map.annotate(
            f'T{target.id}',
            truth[0],
            xytext=(-12, -12),
            textcoords='offset points',
            color='0.3',
        )

    tracks: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for snapshot in result.snapshots:
        tracks[(snapshot.node_id, snapshot.track_id)].append(snapshot.mean[:2])
    colors = plt.get_cmap('tab10')
    for i, ((node_id, track_id), points) in enumerate(sorted(tracks.items())):
        xy = np.array(points)
        ax_map.plot(
            *xy.T,
            '.',
            ms=2.5,
            color=colors(i % 10),
            label=f'node {node_id} track {track_id % 1_000_000_000}',
        )
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('x (m)')
    ax_map.set_ylabel('y (m)')
    ax_map.set_title('Truth (gray), robot paths (dotted), released tracks')
    if len(tracks) <= 12:
        ax_map.legend(fontsize=7, loc='upper left', markerscale=4)

    ax_err.plot(times, metrics.rmse, label='position RMSE (m)')
    ax_err.set_xlabel('time (s)')
    ax_err.set_ylabel('RMSE (m)')
    ax_err.set_yscale('log')
    ax_cust = ax_err.twinx()
    ax_cust.step(
        times,
        metrics.custody,
        where='post',
        color='tab:green',
        alpha=0.7,
        label='custody',
    )
    ax_cust.set_ylim(-0.05, 1.05)
    ax_cust.set_ylabel('fraction of targets in custody')
    for target in scenario.targets:
        if target.appears > 0:
            ax_err.axvline(target.appears, color='0.7', lw=0.8, ls='--')
    ax_err.set_title('Accuracy and custody (dashed: targets appear)')
    lines = ax_err.get_lines()[:1] + ax_cust.get_lines()[:1]
    ax_err.legend(lines, [line.get_label() for line in lines], loc='upper right')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
