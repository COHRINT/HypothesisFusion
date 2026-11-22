"""Run the 2D bearings demo.

uv run python -m examples.bearings2d                  # both architectures
uv run python -m examples.bearings2d --mode decentralized --plot outputs
uv run python -m examples.bearings2d --release-delay 0.25   # out-of-sequence
"""

import argparse
import pathlib

from examples.bearings2d import simulation
from examples.bearings2d.metrics import evaluate
from examples.bearings2d.scenario import default_scenario

MODES = {
    'centralized': simulation.CENTRALIZED_ROUTES,
    'decentralized': simulation.DECENTRALIZED_ROUTES,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', choices=[*MODES, 'both'], default='both')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--release-delay',
        type=float,
        default=0.0,
        help='max random delay (s) of sensor track releases to fusion nodes',
    )
    parser.add_argument(
        '--plot', type=pathlib.Path, help='directory to save plots to (matplotlib)'
    )
    args = parser.parse_args()

    scenario = default_scenario()
    modes = list(MODES) if args.mode == 'both' else [args.mode]
    rows = {}
    for mode in modes:
        result = simulation.run(
            scenario,
            MODES[mode],
            seed=args.seed,
            max_release_delay=args.release_delay,
        )
        metrics = evaluate(result)
        rows[mode] = metrics.summary()
        print(f'\n{mode}: {len(result.nodes)} fusion node(s)')
        for node in result.nodes:
            tracks = {
                track_id % 1_000_000_000: sorted(h.sensor_track_ids)
                for track_id, h in node.released_tracks.items()
            }
            print(f'  node {node.node_id} final tracks (sensor tracks): {tracks}')
        if result.message_counts:
            print(f'  messages: {result.message_counts}')
        if args.plot:
            from examples.bearings2d.plotting import plot_run

            args.plot.mkdir(parents=True, exist_ok=True)
            path = args.plot / f'bearings2d_{mode}.png'
            plot_run(result, metrics, f'Hypothesis fusion, {mode}', str(path))
            print(f'  plot: {path}')

    print(f'\n{"metric":<24}' + ''.join(f'{m:>16}' for m in rows))
    for metric in next(iter(rows.values())):
        print(f'{metric:<24}' + ''.join(f'{rows[m][metric]:>16.3f}' for m in rows))


if __name__ == '__main__':
    main()
