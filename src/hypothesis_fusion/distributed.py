"""Decentralized hypothesis fusion (MFI Sec. III-H).

When sensor tracks are routed to several fusion nodes, the nodes can end up
with overlapping hypothesis sets. Because a node's hypotheses are summarized by
the sensor track IDs of its partitions
(:class:`~hypothesis_fusion.summary.NodeSummary`), nodes resolve the overlap
with integer set operations:

1. **Sharing.** After a structural change, a node shares its summary with its
   peers (which peers is the application's choice, e.g. k nearest).
2. **Deduplication rules.** On receiving a peer's summary, for each pair of an
   own partition ``A`` and a peer partition ``B``, the node asks the peer to
   prune when:

   - ``A == B`` and the node's active track set is smaller (tiebreak: lower
     node ID) -- the peer has more work to drop (``HAVE_SMALLER_SET``);
   - ``A`` is a strict subset of ``B`` and the node holds all of ``B``'s tracks
     across its partitions -- it has resolved the problem more finely
     (``HAVE_RESOLVED_PARTITION``);
   - ``A`` is a strict superset of ``B`` and the peer has none of ``A - B`` --
     the node has strictly more information (``HAVE_SUPER_PARTITION``);
   - ``A`` and ``B`` properly intersect, neither node has the full picture,
     and the partitions are merged at one node: the node with more tracks the
     peer is unaware of requests the peer's missing tracks (``MergeRequest``),
     then resamples the union.

3. **Subset re-sharing.** The superset rule is only checked by the node
   holding the superset, on receiving the subset holder's summary. If the
   superset forms after the subset holder last shared, and the subset holder's
   partitions then stop changing, the request would never be made. So when a
   received summary shows the sender would make that request, the receiver
   replies with its own summary. (This is an addition to the published method.)
4. **Request acceptance.** A node accepts a dedup request only if it does not
   contradict one of its own outstanding requests and its local state agrees
   with the stated reason; on accept, it prunes the sensor tracks and ignores
   them from then on.

:class:`FusionCoordinator` wraps a :class:`~hypothesis_fusion.fusion.HypothesisFusion`
node with this protocol. It is transport-agnostic: each handler returns the
messages to send back to the peer and, when the node changed, the summary to
re-share.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
from collections.abc import Iterable

from hypothesis_fusion.fusion import ID_STRIDE, HypothesisFusion
from hypothesis_fusion.summary import NodeSummary
from hypothesis_fusion.types import SensorTrack, SensorTrackRelease


class DedupReason(enum.Enum):
    # Equal partitions; the requester's overall active set is smaller.
    HAVE_SMALLER_SET = enum.auto()
    # The requester's partition is a strict subset of the peer's, and the
    # requester holds the rest of the peer's tracks in other partitions.
    HAVE_RESOLVED_PARTITION = enum.auto()
    # The requester's partition strictly contains the peer's, and the peer has
    # none of the extra tracks.
    HAVE_SUPER_PARTITION = enum.auto()


@dataclasses.dataclass(frozen=True)
class DedupRequest:
    """Asks the receiver to prune ``sensor_track_ids`` (the requester's
    partition) from its hypothesis set."""

    request_id: int
    reason: DedupReason
    sensor_track_ids: frozenset[int]


@dataclasses.dataclass(frozen=True)
class DedupResponse:
    request_id: int
    accept: bool


@dataclasses.dataclass(frozen=True)
class MergeRequest:
    """Asks the receiver to hand over the tracks of its partition overlapping
    ``sensor_track_ids`` (the requester's partition) that the requester lacks."""

    request_id: int
    sensor_track_ids: frozenset[int]


@dataclasses.dataclass(frozen=True)
class MergeResponse:
    request_id: int
    accept: bool
    sensor_tracks: tuple[SensorTrack, ...] = ()


Message = NodeSummary | DedupRequest | DedupResponse | MergeRequest | MergeResponse


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


def dedup_requests(
    own: NodeSummary, peer: NodeSummary
) -> list[tuple[DedupReason, frozenset[int]]]:
    """The dedup requests ``own`` should send ``peer``, as (reason, tracks)."""
    requests: list[tuple[DedupReason, frozenset[int]]] = []
    smaller = (len(own.active_sensor_track_ids), own.node_id) < (
        len(peer.active_sensor_track_ids),
        peer.node_id,
    )
    for theirs in peer.partitions:
        b = theirs.sensor_track_ids
        have_all_of_b = b <= own.active_sensor_track_ids
        for ours in own.partitions:
            a = ours.sensor_track_ids
            if a == b:
                reason = DedupReason.HAVE_SMALLER_SET if smaller else None
            elif a < b and have_all_of_b:
                reason = DedupReason.HAVE_RESOLVED_PARTITION
            elif a > b and not (a - b) & peer.active_sensor_track_ids:
                reason = DedupReason.HAVE_SUPER_PARTITION
            else:
                reason = None
            if reason is not None and (reason, a) not in requests:
                requests.append((reason, a))
    return requests


def merge_requests(own: NodeSummary, peer: NodeSummary) -> list[frozenset[int]]:
    """Own partitions to merge with a properly intersecting peer partition, for
    the pairs where ``own`` is the merging node (the "fusor")."""
    requests: list[frozenset[int]] = []
    for ours in own.partitions:
        a = ours.sensor_track_ids
        for theirs in peer.partitions:
            b = theirs.sensor_track_ids
            if not (a & b) or not (a - b) or not (b - a):
                continue
            if _initiates_merge(own, peer, a, b) and a not in requests:
                requests.append(a)
    return requests


def _initiates_merge(
    own: NodeSummary, peer: NodeSummary, a: frozenset[int], b: frozenset[int]
) -> bool:
    """The node with more tracks the other is unaware of merges, so the smaller
    unique set is transferred. Tiebreaks: smaller partition, lower node ID."""
    own_unique = len(a - peer.active_sensor_track_ids)
    peer_unique = len(b - own.active_sensor_track_ids)
    return (-own_unique, len(a), own.node_id) < (-peer_unique, len(b), peer.node_id)


# --------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    """What a handler wants transported: replies to the message's sender, and
    the node's new summary to share with peers if it changed."""

    replies: list[Message] = dataclasses.field(default_factory=list)
    summary: NodeSummary | None = None


class FusionCoordinator:
    """A fusion node participating in the decentralized protocol."""

    def __init__(self, node: HypothesisFusion, request_cooldown: float = 5.0):
        """
        Args:
            node: The local hypothesis fusion node.
            request_cooldown: Seconds before an equivalent request is re-sent to
                the same peer, damping request storms and retry loops. <= 0
                disables the cooldown.
        """
        self.node = node
        self._request_cooldown = request_cooldown
        self._request_ids = itertools.count(node.node_id * ID_STRIDE + 1)
        self._summary: NodeSummary = node.summary()
        self._outstanding_dedup: dict[int, DedupRequest] = {}
        self._outstanding_merge: dict[int, MergeRequest] = {}
        self._sent: dict[tuple[int, str, frozenset[int]], float] = {}

    @property
    def summary(self) -> NodeSummary:
        """The latest summary of the local node."""
        return self._summary

    def _changed(self, summary: NodeSummary | None) -> NodeSummary | None:
        if summary is None or summary == self._summary:
            return None
        self._summary = summary
        return summary

    # ------------------------------------------------------ local processing

    def update(
        self, time: float, releases: Iterable[SensorTrackRelease]
    ) -> NodeSummary | None:
        """Run a node update; returns the summary to share if it changed."""
        return self._changed(self.node.update(time, releases))

    def housekeeping(self, time: float) -> NodeSummary | None:
        return self._changed(self.node.housekeeping(time))

    # -------------------------------------------------------- peer messages

    def on_summary(self, time: float, peer: NodeSummary) -> Outcome:
        """React to a peer's summary with dedup and merge requests, or with our
        own summary if the peer holds a superset it would ask us to prune."""
        outcome = Outcome()
        own = self._summary
        superset = frozenset().union(
            *(
                tracks
                for reason, tracks in dedup_requests(peer, own)
                if reason is DedupReason.HAVE_SUPER_PARTITION
            )
        )
        if superset and self._claim(time, peer.node_id, 'SUMMARY', superset):
            outcome.replies.append(own)
        for reason, tracks in dedup_requests(own, peer):
            if self._claim(time, peer.node_id, reason.name, tracks):
                request = DedupRequest(next(self._request_ids), reason, tracks)
                self._outstanding_dedup[request.request_id] = request
                outcome.replies.append(request)
        if self.node.config.resample is not None:
            for tracks in merge_requests(own, peer):
                if self._claim(time, peer.node_id, 'MERGE', tracks):
                    request = MergeRequest(next(self._request_ids), tracks)
                    self._outstanding_merge[request.request_id] = request
                    outcome.replies.append(request)
        return outcome

    def on_dedup_request(self, time: float, request: DedupRequest) -> Outcome:
        accept = self._accepts(request)
        outcome = Outcome([DedupResponse(request.request_id, accept)])
        if accept:
            outcome.summary = self._changed(
                self.node.prune_sensor_tracks(time, set(request.sensor_track_ids))
            )
        return outcome

    def on_dedup_response(self, response: DedupResponse) -> None:
        self._outstanding_dedup.pop(response.request_id, None)

    def on_merge_request(self, time: float, request: MergeRequest) -> Outcome:
        """Hand over our tracks of an overlapping partition and relinquish it."""
        partition = self._merge_partition(request)
        if partition is None:
            return Outcome([MergeResponse(request.request_id, accept=False)])
        assert self.node.config.resample is not None
        tracks = self.node.export_sensor_tracks(
            partition - request.sensor_track_ids,
            since=time - self.node.config.resample.merge_history,
        )
        summary = self.node.prune_sensor_tracks(time, set(partition))
        return Outcome(
            [
                MergeResponse(
                    request.request_id, accept=True, sensor_tracks=tuple(tracks)
                )
            ],
            self._changed(summary),
        )

    def on_merge_response(self, time: float, response: MergeResponse) -> Outcome:
        """Adopt the handed-over tracks and resample our partition over the union."""
        request = self._outstanding_merge.pop(response.request_id, None)
        if request is None or not response.accept or not response.sensor_tracks:
            return Outcome()
        summary = self.node.merge_sensor_tracks(
            time, response.sensor_tracks, set(request.sensor_track_ids)
        )
        return Outcome(summary=self._changed(summary))

    # ------------------------------------------------------------ internals

    def _accepts(self, request: DedupRequest) -> bool:
        tracks = request.sensor_track_ids
        # Never accept a claim on tracks we are ourselves claiming.
        if any(
            tracks <= mine.sensor_track_ids for mine in self._outstanding_dedup.values()
        ):
            return False
        own = self._summary
        have_all = tracks <= own.active_sensor_track_ids
        partition_within = any(p.sensor_track_ids <= tracks for p in own.partitions)
        match request.reason:
            case DedupReason.HAVE_SMALLER_SET:
                return have_all and partition_within
            case DedupReason.HAVE_RESOLVED_PARTITION:
                return have_all
            case DedupReason.HAVE_SUPER_PARTITION:
                return partition_within and not have_all
        return False

    def _merge_partition(self, request: MergeRequest) -> frozenset[int] | None:
        """Our partition that overlaps the request and has tracks the requester
        lacks; ``None`` (reject) if it conflicts with our own merge requests."""
        if self.node.config.resample is None:
            return None
        for mine in self._outstanding_merge.values():
            if mine.sensor_track_ids & request.sensor_track_ids:
                return None
        for partition in self._summary.partitions:
            tracks = partition.sensor_track_ids
            if tracks & request.sensor_track_ids and tracks - request.sensor_track_ids:
                return tracks
        return None

    def _claim(
        self, time: float, peer_id: int, kind: str, tracks: frozenset[int]
    ) -> bool:
        """Whether an equivalent request may be sent to ``peer_id`` now; if so,
        records the send."""
        if self._request_cooldown <= 0:
            return True
        self._sent = {
            key: sent
            for key, sent in self._sent.items()
            if time - sent < self._request_cooldown
        }
        key = (peer_id, kind, tracks)
        if key in self._sent:
            return False
        self._sent[key] = time
        return True
