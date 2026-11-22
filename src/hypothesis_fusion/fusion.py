"""The hypothesis fusion node.

:class:`HypothesisFusion` is one fusion node's tracker. Each call to
:meth:`HypothesisFusion.update` ingests a batch of sensor track releases and
runs one frame of the algorithm:

1. **Growth** (MFI Sec. III-C): a sensor track with enough measurements is
   associated to at most one partition, whose hypotheses are extended with it
   (if they lack a track from the same sensor), and stereo root hypotheses are
   initialized against that partition's tracks and orphaned tracks. Orphaned
   tracks retry growth periodically.
2. **Filtering**: every hypothesis containing a released track processes its
   new measurements, then is rescored.
3. **Shrinking**: a hypothesis about to be pruned because of
   a single outlier track spawns a child without it.
4. **Pruning** (MFI Sec. III-E): by score, then per-partition caps.
5. **Track management**: greedy best hypothesis selection,
   hypothesis-to-track assignment, gated release into track histories.
6. **Resampling**: partitions touched this frame that have
   poor utilization or redundancy are re-hypothesized.

:meth:`HypothesisFusion.housekeeping` prunes stale and infeasible hypotheses on
the application's own clock. The remaining public methods serve the
distributed protocol (:mod:`hypothesis_fusion.distributed`).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy import typing as npt

from hypothesis_fusion import partitions as partitioning
from hypothesis_fusion.config import FusionConfig
from hypothesis_fusion.filtering import HypothesisFilter
from hypothesis_fusion.initialization import FeasibilityCheck, Initializer
from hypothesis_fusion.release import ReleaseCandidate, ReleaseGate
from hypothesis_fusion.resampling import (
    CliqueResampler,
    PartitionHypothesis,
    reconcile_partition,
)
from hypothesis_fusion.scoring import WindowedScorer, innovation_distance
from hypothesis_fusion.summary import NodeSummary, PartitionSummary
from hypothesis_fusion.track_ids import TrackCandidate, TrackIdAssigner
from hypothesis_fusion.types import Estimate, SensorTrack, SensorTrackRelease

# Hypothesis and track IDs are drawn from a per-node counter offset by
# ``node_id * ID_STRIDE`` so IDs from different nodes never collide.
ID_STRIDE = 1_000_000_000

PositionOf = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclasses.dataclass(eq=False)
class Hypothesis:
    """A hypothesis that a set of sensor tracks observe one common target."""

    id: int
    sensor_track_ids: frozenset[int]
    sensor_ids: frozenset[int]
    filter: HypothesisFilter
    parent_id: int | None
    created: float
    # None until defined (see WindowedScorer.score); such hypotheses are never
    # pruned by score nor selected as best.
    score: float | None = None


@dataclasses.dataclass(eq=False)
class SystemTrack:
    """A released track: its estimate history across hypothesis assignments."""

    track_id: int
    estimates: list[Estimate]
    hypothesis_id: int
    sensor_track_ids: frozenset[int]


class HypothesisFusion:
    def __init__(
        self,
        config: FusionConfig,
        initializer: Initializer,
        position_of: PositionOf,
        feasibility: FeasibilityCheck | None = None,
        node_id: int = 0,
    ):
        """
        Args:
            config: Algorithm parameters.
            initializer: Relates sensor tracks and initializes hypotheses.
            position_of: Extracts the position from a state vector, for the
                positional matching in track ID assignment and resampling.
            feasibility: Optional physical plausibility check for housekeeping.
            node_id: This node's ID; namespaces its hypothesis and track IDs.
        """
        self.node_id = node_id
        self.config = config
        self._initializer = initializer
        self._position_of = position_of
        self._feasibility = feasibility
        self._ids = itertools.count(node_id * ID_STRIDE + 1)

        self._scorer = WindowedScorer(config.score.window, config.score.size_bias)
        self._assigner = TrackIdAssigner(
            config.track_id.min_overlap,
            config.track_id.max_distance,
            new_track_id=self._new_id,
        )
        self._gate = ReleaseGate(config.release.maturity, config.release.min_overlap)
        self._resampler = (
            CliqueResampler(
                k=config.resample.k,
                max_pair_residual=config.resample.max_pair_residual,
                max_clique_residual=config.resample.max_clique_residual,
                min_size=config.resample.min_size,
                size_bias=config.resample.size_bias,
            )
            if config.resample is not None
            else None
        )

        self._hypotheses: dict[int, Hypothesis] = {}
        self._track_to_hypotheses: dict[int, set[int]] = defaultdict(set)
        # Every non-ignored sensor track this node knows of, and which of them
        # have enough measurements to be used.
        self._sensor_tracks: dict[int, SensorTrack] = {}
        self._mature: set[int] = set()
        # Sensor tracks deduplicated to another node; their releases are ignored.
        self._ignored: set[int] = set()
        # Sensor tracks that ended (terminated or timed out). They remain in
        # hypothesis pedigrees but are excluded from live computations.
        self._inactive: set[int] = set()
        self._last_release: dict[int, float] = {}
        self._last_growth: dict[int, float] = {}
        self._last_resample: dict[int, float] = {}

        # Track management state for the current frame.
        self._track_ids: dict[int, int] = {}  # best hypothesis ID -> track ID
        self._released: set[int] = set()  # released hypothesis IDs
        self._histories: dict[int, SystemTrack] = {}

        # Bumped on every structural change (hypotheses added/removed, sensor
        # tracks (in)activated) to invalidate the partition cache and to decide
        # whether a new summary should be shared.
        self._version = 0
        # Bumped on every rescore: partitions' best hypotheses depend on scores.
        self._score_version = 0
        self._partition_cache: tuple[tuple[int, int], list[partitioning.Partition]] = (
            (-1, -1),
            [],
        )

    # ------------------------------------------------------------------ views

    @property
    def hypotheses(self) -> Mapping[int, Hypothesis]:
        return self._hypotheses

    @property
    def sensor_tracks(self) -> Mapping[int, SensorTrack]:
        return self._sensor_tracks

    @property
    def inactive_sensor_track_ids(self) -> frozenset[int]:
        return frozenset(self._inactive)

    @property
    def ignored_sensor_track_ids(self) -> frozenset[int]:
        return frozenset(self._ignored)

    @property
    def released_tracks(self) -> dict[int, Hypothesis]:
        """Currently released best hypotheses, by track ID."""
        return {
            self._track_ids[h.id]: h
            for h in self.best_hypotheses()
            if h.id in self._released
        }

    @property
    def track_histories(self) -> Mapping[int, SystemTrack]:
        """Every track ever released by this node, with its estimate history."""
        return self._histories

    def active_sensor_track_ids(self, hypothesis: Hypothesis) -> frozenset[int]:
        return hypothesis.sensor_track_ids - self._inactive

    def partitions(self) -> list[partitioning.Partition]:
        key = (self._version, self._score_version)
        if self._partition_cache[0] != key:
            self._partition_cache = (
                key,
                partitioning.build_partitions(
                    {h.id: h.sensor_track_ids for h in self._hypotheses.values()},
                    {
                        h.id: self.active_sensor_track_ids(h)
                        for h in self._hypotheses.values()
                    },
                    (h.id for h in self.best_hypotheses()),
                ),
            )
        return self._partition_cache[1]

    def best_hypotheses(self) -> list[Hypothesis]:
        """Greedy best-guess selection (MFI Sec. III-F): the highest-scoring
        hypotheses whose (full) sensor track sets are still unclaimed."""
        scored = [h for h in self._hypotheses.values() if h.score is not None]
        scored.sort(key=lambda h: (-h.score, h.id))  # type: ignore[operator]
        selected: list[Hypothesis] = []
        claimed: set[int] = set()
        for hypothesis in scored:
            if hypothesis.sensor_track_ids & claimed:
                continue
            selected.append(hypothesis)
            claimed |= hypothesis.sensor_track_ids
        return selected

    def summary(self) -> NodeSummary:
        partitions: list[PartitionSummary] = []
        active: set[int] = set()
        inactive: set[int] = set()
        for partition in self.partitions():
            active |= partition.sensor_track_ids
            inactive |= partition.inactive_sensor_track_ids
            if not partition.sensor_track_ids:
                # Nothing active to coordinate on.
                continue
            partitions.append(
                PartitionSummary(
                    sensor_track_ids=frozenset(partition.sensor_track_ids),
                    best_tracks={
                        self._track_ids[h]: self._hypotheses[h].sensor_track_ids
                        for h in partition.best_hypothesis_ids
                        if h in self._track_ids
                    },
                    redundancy=partitioning.redundancy(
                        (
                            self.active_sensor_track_ids(self._hypotheses[h])
                            for h in partition.hypothesis_ids
                        ),
                        partition.sensor_track_ids,
                    ),
                )
            )
        unused = {
            track_id
            for track_id in self._sensor_tracks
            if track_id not in self._inactive
            and not self._track_to_hypotheses.get(track_id)
        }
        return NodeSummary(
            node_id=self.node_id,
            partitions=tuple(sorted(partitions, key=lambda p: min(p.sensor_track_ids))),
            active_sensor_track_ids=frozenset(active),
            unused_sensor_track_ids=frozenset(unused),
            inactive_sensor_track_ids=frozenset(inactive),
            ignored_sensor_track_ids=frozenset(self._ignored),
        )

    # ------------------------------------------------------------- the frame

    def update(
        self, time: float, releases: Iterable[SensorTrackRelease]
    ) -> NodeSummary | None:
        """Process one frame of sensor track releases arriving at ``time``.

        Returns the node's summary if the hypothesis set's structure changed
        (so it should be re-shared with peers), else ``None``.
        """
        version = self._version
        releases = [r for r in releases if r.sensor_track_id not in self._ignored]
        for release in releases:
            self._last_release[release.sensor_track_id] = time
            if release.sensor_track_id in self._inactive and not release.terminated:
                self._inactive.discard(release.sensor_track_id)
                self._version += 1

        new_roots: set[int] = set()
        for release in releases:
            new_roots |= self._ingest(time, release)

        updated: set[int] = set()
        for release in releases:
            if not release.measurements:
                continue
            for hypothesis_id in self._track_to_hypotheses.get(
                release.sensor_track_id, ()
            ):
                # Filters skip measurements they already processed, e.g. those
                # a new root was initialized from.
                self._hypotheses[hypothesis_id].filter.update(release.measurements)
                updated.add(hypothesis_id)
        # A new root's score stays undefined until its first update after
        # initialization (MFI Sec. III-E).
        updated -= new_roots
        for hypothesis_id in updated:
            self._rescore(self._hypotheses[hypothesis_id])

        for release in releases:
            if release.terminated:
                self.mark_inactive(release.sensor_track_id)
        self._sweep_inactive(time)

        self._shrink(time, updated)
        self._prune()
        self._commit(time)
        self._resample_triggered(time, {r.sensor_track_id for r in releases})

        return self.summary() if self._version != version else None

    def housekeeping(self, time: float) -> NodeSummary | None:
        """Prune stale and infeasible hypotheses.

        Returns the new summary if anything changed, else ``None``.
        """
        version = self._version
        self._sweep_inactive(time)
        window = self.config.prune.inactivity_window
        for hypothesis in list(self._hypotheses.values()):
            latest = hypothesis.filter.latest
            stale = time - latest.time > window
            infeasible = self._feasibility is not None and not (
                self._feasibility.is_feasible(latest)
            )
            if stale or infeasible:
                self._remove(hypothesis.id)
        if self._version == version:
            return None
        self._commit(time)
        return self.summary()

    def mark_inactive(self, sensor_track_id: int) -> None:
        if sensor_track_id not in self._inactive:
            self._inactive.add(sensor_track_id)
            self._version += 1

    # ------------------------------------------------------------ growth

    def _ingest(self, time: float, release: SensorTrackRelease) -> set[int]:
        """Fold a release into its sensor track; grow hypotheses if warranted.
        Returns the IDs of new root hypotheses."""
        track_id = release.sensor_track_id
        track = self._sensor_tracks.get(track_id)
        if track is None:
            track = SensorTrack.from_release(release)
            self._sensor_tracks[track_id] = track
        else:
            track.absorb(release)

        if track_id in self._mature:
            if self._should_retry_orphan(time, track_id):
                return self._grow(time, track)
            return set()
        if len(track.measurements) >= self.config.growth.min_track_measurements:
            self._mature.add(track_id)
            return self._grow(time, track)
        return set()

    def _should_retry_orphan(self, time: float, track_id: int) -> bool:
        interval = self.config.growth.orphan_retry_interval
        return (
            interval > 0
            and track_id not in self._inactive
            and not self._track_to_hypotheses.get(track_id)
            and time - self._last_growth.get(track_id, -np.inf) >= interval
        )

    def _grow(self, time: float, track: SensorTrack) -> set[int]:
        """Grow hypotheses with a newly mature (or orphaned) sensor track."""
        self._last_growth[track.id] = time
        growth = self.config.growth

        # Associate the track to at most one partition: the one whose best
        # hypothesis is closest to what the track observes.
        partition: partitioning.Partition | None = None
        closest = growth.partition_gate
        for candidate in self.partitions():
            for hypothesis_id in candidate.best_hypothesis_ids:
                estimate = self._hypotheses[hypothesis_id].filter.predict(track.time)
                residual = self._initializer.track_residual(estimate, track)
                if residual < closest:
                    closest, partition = residual, candidate

        # Extend the partition's hypotheses that lack a track from this sensor.
        existing = {h.sensor_track_ids for h in self._hypotheses.values()}
        if partition is not None:
            for hypothesis_id in sorted(partition.hypothesis_ids):
                parent = self._hypotheses[hypothesis_id]
                tracks = parent.sensor_track_ids | {track.id}
                if track.sensor_id in parent.sensor_ids or tracks in existing:
                    continue
                self._add(
                    Hypothesis(
                        id=self._new_id(),
                        sensor_track_ids=tracks,
                        sensor_ids=parent.sensor_ids | {track.sensor_id},
                        filter=parent.filter.copy(),
                        parent_id=parent.id,
                        created=time,
                    )
                )
                existing.add(tracks)

        # Initialize stereo roots against the partition's tracks and orphans.
        partition_tracks = partition.sensor_track_ids if partition else set()
        roots: set[int] = set()
        for other_id in sorted(self._mature):
            other = self._sensor_tracks[other_id]
            pair = frozenset({track.id, other_id})
            if (
                other.sensor_id == track.sensor_id
                or other_id in self._inactive
                or pair in existing
                or (
                    self._track_to_hypotheses.get(other_id)
                    and other_id not in partition_tracks
                )
            ):
                continue
            residual = self._initializer.residual([track, other], track.time)
            if residual is None or residual > growth.pair_gate:
                continue
            filter = self._initializer.initialize([track, other], track.time)
            if filter is None:
                continue
            root = Hypothesis(
                id=self._new_id(),
                sensor_track_ids=pair,
                sensor_ids=frozenset({track.sensor_id, other.sensor_id}),
                filter=filter,
                parent_id=None,
                created=time,
            )
            self._add(root)
            existing.add(pair)
            roots.add(root.id)
        return roots

    # ----------------------------------------------------------- scoring

    def _rescore(self, hypothesis: Hypothesis) -> None:
        self._score_version += 1
        hypothesis.score = self._scorer.score(
            hypothesis.filter,
            hypothesis.sensor_track_ids,
            self.active_sensor_track_ids(hypothesis),
        )

    # ---------------------------------------------------------- shrinking

    def _shrink(self, time: float, hypothesis_ids: Iterable[int]) -> None:
        """Split a single outlier sensor track off hypotheses it is about to get
        pruned, so the rest of their evidence survives."""
        threshold = self.config.growth.shrink_innovation_threshold
        if threshold is None:
            return
        existing = {h.sensor_track_ids for h in self._hypotheses.values()}
        for hypothesis_id in sorted(hypothesis_ids):
            parent = self._hypotheses.get(hypothesis_id)
            if (
                parent is None
                or parent.score is None
                or parent.score >= self.config.prune.min_score
            ):
                continue
            outlier = self._single_outlier(parent, threshold)
            if outlier is None or parent.sensor_track_ids - {outlier} in existing:
                continue
            child = self._shrink_hypothesis(time, parent, outlier)
            existing.add(child.sensor_track_ids)

    def _single_outlier(self, hypothesis: Hypothesis, threshold: float) -> int | None:
        """The one active sensor track whose RMS innovation distance exceeds
        ``threshold``, provided removing it leaves at least a stereo pair."""
        likelihoods = self._scorer.sensor_track_likelihoods(
            hypothesis.filter, self.active_sensor_track_ids(hypothesis)
        )
        outliers = [
            track_id
            for track_id, log_likelihood in likelihoods.items()
            if innovation_distance(log_likelihood) > threshold
        ]
        if len(outliers) != 1 or len(likelihoods) - 1 < 2:
            return None
        return outliers[0]

    def _shrink_hypothesis(
        self, time: float, parent: Hypothesis, outlier: int
    ) -> Hypothesis:
        """A child of ``parent`` without ``outlier``, with the outlier's updates
        within the scoring window undone."""
        undo = [
            m.id for m in parent.filter.measurements if m.sensor_track_id == outlier
        ][-self._scorer.window :]
        child = Hypothesis(
            id=self._new_id(),
            sensor_track_ids=parent.sensor_track_ids - {outlier},
            sensor_ids=parent.sensor_ids - {self._sensor_tracks[outlier].sensor_id},
            filter=parent.filter.without_measurements(set(undo)),
            parent_id=parent.id,
            created=time,
        )
        self._rescore(child)
        self._add(child)
        return child

    # ------------------------------------------------------------ pruning

    def _prune(self) -> None:
        """Prune by score, then cap each partition at ``base + b * N**2``
        hypotheses (N = its best hypotheses). Best hypotheses and hypotheses
        whose score is still undefined are always kept."""
        prune = self.config.prune
        for hypothesis in list(self._hypotheses.values()):
            if hypothesis.score is not None and hypothesis.score < prune.min_score:
                self._remove(hypothesis.id)

        for partition in self.partitions():
            cap = (
                prune.base
                + prune.per_best_squared * len(partition.best_hypothesis_ids) ** 2
            )
            scored = sorted(
                (
                    self._hypotheses[h]
                    for h in partition.hypothesis_ids
                    if self._hypotheses[h].score is not None
                ),
                key=lambda h: (-h.score, h.id),  # type: ignore[operator]
            )
            kept = 0
            for hypothesis in scored:
                if kept >= cap and hypothesis.id not in partition.best_hypothesis_ids:
                    self._remove(hypothesis.id)
                else:
                    kept += 1

    # --------------------------------------------------- track management

    def _commit(self, time: float) -> None:
        """Assign track IDs to the best hypotheses, gate their release, and
        append released estimates to the track histories. Once per frame."""
        best = self.best_hypotheses()
        self._track_ids = self._assigner.assign(
            [
                TrackCandidate(
                    hypothesis_id=h.id,
                    sensor_track_ids=h.sensor_track_ids,
                    active_sensor_track_ids=self.active_sensor_track_ids(h),
                    position=self._position_of(h.filter.latest.mean),
                )
                for h in best
            ]
        )
        self._released = self._gate.released(
            [
                ReleaseCandidate(
                    hypothesis_id=h.id,
                    track_id=self._track_ids[h.id],
                    sensor_track_ids=h.sensor_track_ids,
                    active_sensor_track_ids=self.active_sensor_track_ids(h),
                )
                for h in best
            ],
            time,
        )
        for track_id, hypothesis in self.released_tracks.items():
            latest = hypothesis.filter.latest
            track = self._histories.setdefault(
                track_id, SystemTrack(track_id, [], hypothesis.id, frozenset())
            )
            if not track.estimates or latest.time > track.estimates[-1].time:
                track.estimates.append(latest)
            track.hypothesis_id = hypothesis.id
            track.sensor_track_ids = hypothesis.sensor_track_ids

    # --------------------------------------------------------- resampling

    def _resample_triggered(self, time: float, touched: set[int]) -> None:
        """Resample partitions touched this frame whose best hypotheses
        under-utilize their sensor tracks or that lack redundancy."""
        if self._resampler is None:
            return
        due = [
            p
            for p in self.partitions()
            if p.sensor_track_ids & touched
            and self._resample_cooldown_elapsed(p, time)
            and self._needs_resample(p)
        ]
        for partition in due:
            self._resample(time, partition)
        if due:
            self._commit(time)

    def resample_all(self, time: float) -> NodeSummary | None:
        """Resample every partition now, regardless of triggers."""
        if self._resampler is None:
            return None
        version = self._version
        for partition in list(self.partitions()):
            self._resample(time, partition)
        self._commit(time)
        return self.summary() if self._version != version else None

    def _needs_resample(self, partition: partitioning.Partition) -> bool:
        cfg = self.config.resample
        assert cfg is not None
        tracks = partition.sensor_track_ids
        if len(tracks) < cfg.min_size:
            return False
        best = [
            self.active_sensor_track_ids(self._hypotheses[h])
            for h in partition.best_hypothesis_ids
        ]
        if partitioning.utilization(best, tracks) < cfg.utilization_threshold:
            return True
        if len(tracks) < partitioning.MIN_REDUNDANCY_TRACKS:
            return False
        redundancy = partitioning.redundancy(
            (
                self.active_sensor_track_ids(self._hypotheses[h])
                for h in partition.hypothesis_ids
            ),
            tracks,
        )
        return redundancy <= cfg.redundancy_threshold

    def _resample_cooldown_elapsed(
        self, partition: partitioning.Partition, time: float
    ) -> bool:
        assert self.config.resample is not None
        last = [
            self._last_resample[t]
            for t in partition.sensor_track_ids
            if t in self._last_resample
        ]
        return not last or time - max(last) >= self.config.resample.cooldown

    def _resample(self, time: float, partition: partitioning.Partition) -> None:
        """Clique resampling over the partition's active tracks plus orphans."""
        assert self._resampler is not None
        pool = {t for t in partition.sensor_track_ids if t not in self._inactive}
        pool |= {
            t
            for t in self._mature
            if t not in self._inactive and not self._track_to_hypotheses.get(t)
        }
        tracks = [self._sensor_tracks[t] for t in sorted(pool) if t in self._mature]
        if len(tracks) < 2:
            return
        for track in tracks:
            self._last_resample[track.id] = time

        candidates = self._resampler.candidates(
            tracks, lambda ts: self._initializer.residual(ts, time)
        )
        plan = reconcile_partition(
            [
                PartitionHypothesis(
                    hypothesis_id=h,
                    sensor_track_ids=self.active_sensor_track_ids(self._hypotheses[h]),
                    is_best=h in partition.best_hypothesis_ids,
                )
                for h in partition.hypothesis_ids
                if h in self._hypotheses
            ],
            candidates,
        )
        parents = [
            self._hypotheses[h]
            for h in partition.best_hypothesis_ids
            if h in self._hypotheses
        ]
        for sensor_track_ids in plan.new_sensor_track_sets:
            self._construct_resampled(time, sensor_track_ids, parents)
        for hypothesis_id in plan.pruned_hypothesis_ids:
            self._remove(hypothesis_id)

    def _construct_resampled(
        self,
        time: float,
        sensor_track_ids: frozenset[int],
        parents: Sequence[Hypothesis],
    ) -> None:
        """Create a resampled hypothesis, grown from the closest best hypothesis
        (inheriting its filter history and inactive pedigree).

        Only best hypotheses sharing a sensor track with the candidate can be
        its parent: a filter copied from an unrelated hypothesis would describe
        a different target. Without such a parent, the freshly initialized
        filter is used.
        """
        tracks = [self._sensor_tracks[t] for t in sorted(sensor_track_ids)]
        initialized = self._initializer.initialize(tracks, time)
        if initialized is None:
            return
        init = initialized.latest
        parent = min(
            (p for p in parents if p.sensor_track_ids & sensor_track_ids),
            key=lambda h: float(
                np.linalg.norm(
                    self._position_of(h.filter.predict(init.time).mean)
                    - self._position_of(init.mean)
                )
            ),
            default=None,
        )
        if parent is not None:
            filter = parent.filter.copy()
            inherited = parent.sensor_track_ids & self._inactive
        else:
            filter, inherited = initialized, frozenset()
        all_tracks = sensor_track_ids | inherited
        if any(h.sensor_track_ids == all_tracks for h in self._hypotheses.values()):
            return
        hypothesis = Hypothesis(
            id=self._new_id(),
            sensor_track_ids=all_tracks,
            sensor_ids=frozenset(self._sensor_tracks[t].sensor_id for t in all_tracks),
            filter=filter,
            parent_id=parent.id if parent is not None else None,
            created=time,
        )
        self._rescore(hypothesis)
        self._add(hypothesis)

    # ------------------------------------------- distributed fusion support

    def prune_sensor_tracks(
        self, time: float, sensor_track_ids: set[int]
    ) -> NodeSummary:
        """Deduplicate: drop every hypothesis using any of ``sensor_track_ids``
        and ignore those tracks from now on."""
        for hypothesis in list(self._hypotheses.values()):
            if hypothesis.sensor_track_ids & sensor_track_ids:
                self._remove(hypothesis.id)
        self._ignored |= sensor_track_ids
        for track_id in sensor_track_ids:
            self._sensor_tracks.pop(track_id, None)
            self._mature.discard(track_id)
            self._track_to_hypotheses.pop(track_id, None)
        self._version += 1
        self._prune()
        self._commit(time)
        return self.summary()

    def export_sensor_tracks(
        self, sensor_track_ids: Iterable[int], since: float
    ) -> list[SensorTrack]:
        """Copies of mature sensor tracks with measurements since ``since``, for
        handing them to a peer in a partition merge."""
        return [
            self._sensor_tracks[t].trimmed(since)
            for t in sorted(sensor_track_ids)
            if t in self._mature
        ]

    def merge_sensor_tracks(
        self,
        time: float,
        tracks: Sequence[SensorTrack],
        anchor_sensor_track_ids: set[int],
    ) -> NodeSummary:
        """Adopt sensor tracks handed over by a peer and resample the partition
        overlapping ``anchor_sensor_track_ids`` over the union."""
        for track in tracks:
            self._ignored.discard(track.id)
            self._inactive.discard(track.id)
            self._sensor_tracks[track.id] = dataclasses.replace(
                track, measurements=list(track.measurements)
            )
            self._mature.add(track.id)
            self._last_release[track.id] = time
            # The resample below is this track's first chance to join a
            # hypothesis; if it stays orphaned, growth retries it later.
            self._last_growth[track.id] = time
        self._version += 1
        if self._resampler is not None:
            for partition in list(self.partitions()):
                if partition.sensor_track_ids & anchor_sensor_track_ids:
                    self._resample(time, partition)
        self._commit(time)
        return self.summary()

    # ---------------------------------------------------------- internals

    def _new_id(self) -> int:
        return next(self._ids)

    def _add(self, hypothesis: Hypothesis) -> None:
        self._hypotheses[hypothesis.id] = hypothesis
        for track_id in hypothesis.sensor_track_ids:
            self._track_to_hypotheses[track_id].add(hypothesis.id)
        self._version += 1

    def _remove(self, hypothesis_id: int) -> None:
        hypothesis = self._hypotheses.pop(hypothesis_id, None)
        if hypothesis is None:
            return
        for track_id in hypothesis.sensor_track_ids:
            if (users := self._track_to_hypotheses.get(track_id)) is not None:
                users.discard(hypothesis_id)
        self._version += 1

    def _sweep_inactive(self, time: float) -> None:
        timeout = self.config.inactive_timeout
        if timeout <= 0:
            return
        for track_id, last in self._last_release.items():
            if track_id not in self._ignored and time - last > timeout:
                self.mark_inactive(track_id)
