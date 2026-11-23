"""Evaluation of a demo run.

Track-to-truth association is each released track's nearest target, up to
``cutoff`` meters. With that association, per step:

- **RMSE** of associated track positions;
- **custody**: fraction of targets with an associated track within ``custody``
  meters;
- **ambiguity**: average number of tracks per tracked target (1 is ideal);
- **spuriousness**: fraction of tracks associated to no target.

Over the whole run:

- **purity**: per track, the fraction of its sensor track memberships (summed
  over steps) belonging to its plurality target, averaged over tracks;
- **track ID changes** per target: distinct track IDs that held it, minus one.
"""

import dataclasses
from collections import Counter, defaultdict

import numpy as np

from examples.bearings2d.simulation import RunResult


@dataclasses.dataclass(frozen=True)
class Metrics:
    times: np.ndarray
    rmse: np.ndarray  # per step; NaN where nothing is associated
    custody: np.ndarray  # per step
    ambiguity: np.ndarray  # per step; NaN where nothing is tracked
    spuriousness: np.ndarray  # per step
    purity: float
    track_id_changes: dict[int, int]
    tracks: int

    def summary(self) -> dict[str, float]:
        settled = self.times >= self.times[0] + 0.25 * (self.times[-1] - self.times[0])
        return {
            'rmse_m (settled)': float(np.nanmean(self.rmse[settled])),
            'custody (settled)': float(np.mean(self.custody[settled])),
            'ambiguity (settled)': float(np.nanmean(self.ambiguity[settled])),
            'spuriousness (settled)': float(np.mean(self.spuriousness[settled])),
            'purity': self.purity,
            'track_id_changes': float(sum(self.track_id_changes.values())),
            'tracks': float(self.tracks),
        }


def evaluate(
    result: RunResult, cutoff: float = 100.0, custody: float = 20.0
) -> Metrics:
    scenario = result.scenario
    by_time: dict[float, list] = defaultdict(list)
    for snapshot in result.snapshots:
        by_time[snapshot.time].append(snapshot)

    times = scenario.times
    rmse, held, ambiguity, spurious = [], [], [], []
    ids_per_target: dict[int, set[int]] = defaultdict(set)
    for time in times:
        snapshots = by_time.get(float(time), [])
        errors: list[float] = []
        tracks_per_target: Counter[int] = Counter()
        in_custody: set[int] = set()
        unassociated = 0
        for snapshot in snapshots:
            distances = {
                target.id: float(
                    np.linalg.norm(
                        snapshot.mean[:2] - target.state(snapshot.estimate_time)[:2]
                    )
                )
                for target in scenario.targets
                if target.present(snapshot.estimate_time)
            }
            if not distances:
                unassociated += 1
                continue
            target_id = min(distances, key=distances.__getitem__)
            if distances[target_id] > cutoff:
                unassociated += 1
                continue
            errors.append(distances[target_id])
            tracks_per_target[target_id] += 1
            ids_per_target[target_id].add(snapshot.track_id)
            if distances[target_id] <= custody:
                in_custody.add(target_id)
        rmse.append(np.sqrt(np.mean(np.square(errors))) if errors else np.nan)
        present = sum(1 for target in scenario.targets if target.present(time))
        held.append(len(in_custody) / present if present else 1.0)
        ambiguity.append(
            np.mean(list(tracks_per_target.values())) if tracks_per_target else np.nan
        )
        spurious.append(unassociated / len(snapshots) if snapshots else 0.0)

    memberships: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for snapshot in result.snapshots:
        for sensor_track_id in snapshot.sensor_track_ids:
            target = result.target_of_sensor_track[sensor_track_id]
            memberships[(snapshot.node_id, snapshot.track_id)][target] += 1
    purities = [
        counts.most_common(1)[0][1] / sum(counts.values())
        for counts in memberships.values()
    ]

    return Metrics(
        times=times,
        rmse=np.array(rmse),
        custody=np.array(held),
        ambiguity=np.array(ambiguity),
        spuriousness=np.array(spurious),
        purity=float(np.mean(purities)) if purities else float('nan'),
        track_id_changes={t: len(ids) - 1 for t, ids in ids_per_target.items()},
        tracks=len(memberships),
    )
