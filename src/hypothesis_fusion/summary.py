"""Partition summaries: a fusion node's hypothesis set, reduced to integer sets.

Because hypotheses are identified by the IDs of their sensor tracks, a node's
whole information pedigree can be summarized by its partitions' sensor track
sets. Nodes share these summaries to deduplicate (see
:mod:`hypothesis_fusion.distributed`).
"""

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class PartitionSummary:
    # Active sensor tracks in the partition.
    sensor_track_ids: frozenset[int]
    # Track ID -> sensor tracks of each best hypothesis in the partition.
    best_tracks: Mapping[int, frozenset[int]]
    # Derived health metric (see partitions.redundancy); informational only, so
    # excluded from equality.
    redundancy: float = dataclasses.field(default=float('nan'), compare=False)


@dataclasses.dataclass(frozen=True)
class NodeSummary:
    node_id: int
    partitions: tuple[PartitionSummary, ...]
    # Sensor tracks used by some hypothesis (the union of the partitions).
    active_sensor_track_ids: frozenset[int]
    # Live sensor tracks known to the node but in no hypothesis.
    unused_sensor_track_ids: frozenset[int] = frozenset()
    # Ended sensor tracks still in some hypothesis' pedigree.
    inactive_sensor_track_ids: frozenset[int] = frozenset()
    # Sensor tracks this node deduplicated away and now ignores.
    ignored_sensor_track_ids: frozenset[int] = frozenset()
