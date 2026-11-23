"""Hypothesis partitioning (MFI Sec. III-B) and partition health.

Two hypotheses are equivalent if they are connected through a chain of
hypotheses in which consecutive hypotheses share an (active) sensor track. The
equivalence classes -- *partitions* -- are independent tracking subproblems
with disjoint sensor track sets, so they can be grown, pruned, resampled, and
deduplicated independently.
"""

import dataclasses
from collections.abc import Iterable, Mapping, Set

from hypothesis_fusion import overlap


@dataclasses.dataclass
class Partition:
    hypothesis_ids: set[int]
    best_hypothesis_ids: set[int]
    # Active sensor tracks: partition membership and deduplication use these.
    sensor_track_ids: set[int]
    # Ended sensor tracks still in the pedigree of the partition's hypotheses.
    inactive_sensor_track_ids: set[int]


def build_partitions(
    sensor_tracks: Mapping[int, Set[int]],
    active_sensor_tracks: Mapping[int, Set[int]],
    best_hypothesis_ids: Iterable[int],
) -> list[Partition]:
    """Partition hypotheses by shared active sensor tracks.

    ``sensor_tracks``/``active_sensor_tracks`` map each hypothesis ID to its full
    and active sensor track sets. Hypotheses only connect through *active*
    tracks, so an ended track shared in two pedigrees no longer glues them
    together.
    """
    parent = {hypothesis_id: hypothesis_id for hypothesis_id in sensor_tracks}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    owner_of_track: dict[int, int] = {}
    for hypothesis_id in sensor_tracks:
        for track_id in active_sensor_tracks[hypothesis_id]:
            if track_id in owner_of_track:
                parent[find(hypothesis_id)] = find(owner_of_track[track_id])
            else:
                owner_of_track[track_id] = hypothesis_id

    partitions: dict[int, Partition] = {}
    for hypothesis_id, tracks in sensor_tracks.items():
        partition = partitions.setdefault(
            find(hypothesis_id), Partition(set(), set(), set(), set())
        )
        active = active_sensor_tracks[hypothesis_id]
        partition.hypothesis_ids.add(hypothesis_id)
        partition.sensor_track_ids |= active
        partition.inactive_sensor_track_ids |= tracks - active
    for hypothesis_id in best_hypothesis_ids:
        partitions[find(hypothesis_id)].best_hypothesis_ids.add(hypothesis_id)
    return list(partitions.values())


def utilization(best_sensor_tracks: Iterable[Set[int]], partition: Set[int]) -> float:
    """Overlap between the best hypotheses' sensor tracks and the partition's.

    Low utilization means the best hypotheses leave much of the partition's
    information unexplained: the hypothesis that would explain it is missing.
    """
    used: set[int] = set()
    for tracks in best_sensor_tracks:
        used |= tracks
    return overlap.overlap(used, partition)


# A partition needs at least this many sensor tracks for redundancy to mean
# anything: with two, the only fallback for losing one is a single track, which
# cannot explain the target on its own (in a bearings sensor model).
MIN_REDUNDANCY_TRACKS = 3


def redundancy(
    hypothesis_sensor_tracks: Iterable[Set[int]], partition: Set[int]
) -> float:
    """Fraction of the partition's sensor tracks whose loss it could absorb:
    those for which some hypothesis in the partition excludes them.

    0.0 means one track runs through every hypothesis, so losing it (e.g. it
    switches targets) empties the partition -- the hypothesis-set analogue of
    particle depletion. NaN for an empty partition.
    """
    if not partition:
        return float('nan')
    # A hypothesis with no active tracks explains nothing, so it is nobody's
    # fallback even though it trivially excludes every track.
    sets = [tracks for tracks in hypothesis_sensor_tracks if tracks]
    absorbed = sum(1 for t in partition if any(t not in s for s in sets))
    return absorbed / len(partition)
