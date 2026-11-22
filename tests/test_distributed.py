from collections import deque

from hypothesis_fusion.config import (
    FusionConfig,
    GrowthConfig,
    ResampleConfig,
    TrackIdConfig,
)
from hypothesis_fusion.distributed import (
    DedupReason,
    DedupRequest,
    DedupResponse,
    FusionCoordinator,
    MergeRequest,
    MergeResponse,
    dedup_requests,
    merge_requests,
)
from hypothesis_fusion.fusion import HypothesisFusion
from hypothesis_fusion.summary import NodeSummary, PartitionSummary
from tests.position_sensing import PositionInitializer, PositionSensing


def _summary(node_id, *partitions, extra=()):
    parts = tuple(PartitionSummary(frozenset(p), {}) for p in partitions)
    active = frozenset().union(*map(frozenset, partitions), extra)
    return NodeSummary(
        node_id=node_id, partitions=parts, active_sensor_track_ids=active
    )


class TestDedupRules:
    def test_equal_partitions_smaller_set_requests(self):
        small = _summary(1, {1, 2})
        large = _summary(2, {1, 2}, {7, 8})
        assert dedup_requests(small, large) == [
            (DedupReason.HAVE_SMALLER_SET, frozenset({1, 2}))
        ]
        assert dedup_requests(large, small) == []

    def test_equal_partitions_tiebreak_on_node_id(self):
        a, b = _summary(1, {1, 2}), _summary(2, {1, 2})
        assert dedup_requests(a, b) and not dedup_requests(b, a)

    def test_resolved_subset_requests(self):
        # Node 1 split {1,2,3} into {1,2} and {3}; node 2 still has it whole.
        resolved = _summary(1, {1, 2}, {3})
        whole = _summary(2, {1, 2, 3})
        reasons = {r for r, _ in dedup_requests(resolved, whole)}
        assert reasons == {DedupReason.HAVE_RESOLVED_PARTITION}

    def test_subset_without_the_rest_does_not_request(self):
        assert dedup_requests(_summary(1, {1, 2}), _summary(2, {1, 2, 3})) == []

    def test_unique_superset_requests(self):
        assert dedup_requests(_summary(1, {1, 2, 3}), _summary(2, {1, 2})) == [
            (DedupReason.HAVE_SUPER_PARTITION, frozenset({1, 2, 3}))
        ]

    def test_superset_known_to_peer_does_not_request(self):
        # The peer holds track 3 elsewhere, so neither side has the full story.
        assert dedup_requests(_summary(1, {1, 2, 3}), _summary(2, {1, 2}, {3})) == []


class TestMergeRules:
    def test_proper_intersection_merges_at_one_node(self):
        a, b = _summary(1, {1, 2, 3}), _summary(2, {3, 4})
        # Node 1 has two tracks node 2 lacks; node 2 has one node 1 lacks.
        assert merge_requests(a, b) == [frozenset({1, 2, 3})]
        assert merge_requests(b, a) == []

    def test_containment_is_not_a_merge(self):
        assert merge_requests(_summary(1, {1, 2, 3}), _summary(2, {1, 2})) == []
        assert merge_requests(_summary(1, {1, 2}), _summary(2, {1, 2})) == []


def _coordinator(node_id: int) -> FusionCoordinator:
    config = FusionConfig(
        growth=GrowthConfig(partition_gate=5.0, pair_gate=2.0),
        track_id=TrackIdConfig(max_distance=5.0),
        resample=ResampleConfig(max_pair_residual=2.0, max_clique_residual=2.0),
    )
    node = HypothesisFusion(
        config, PositionInitializer(), position_of=lambda x: x[:2], node_id=node_id
    )
    return FusionCoordinator(node)


class TestCoordinator:
    def test_accepts_matching_dedup_request_and_prunes(self):
        coordinator = _coordinator(1)
        sensing = PositionSensing([[0, 0, 1, 0]], sensors=[1, 2])
        for k in range(1, 11):
            coordinator.update(0.5 * k, sensing.releases(0.5 * k))
        tracks = coordinator.summary.partitions[0].sensor_track_ids
        request = DedupRequest(7, DedupReason.HAVE_SMALLER_SET, tracks)
        outcome = coordinator.on_dedup_request(5.0, request)
        assert outcome.replies == [DedupResponse(7, accept=True)]
        assert outcome.summary is not None and not outcome.summary.partitions
        assert tracks <= coordinator.node.ignored_sensor_track_ids

    def test_rejects_claim_on_tracks_it_lacks(self):
        coordinator = _coordinator(1)
        request = DedupRequest(7, DedupReason.HAVE_RESOLVED_PARTITION, frozenset({9}))
        assert coordinator.on_dedup_request(0.0, request).replies == [
            DedupResponse(7, accept=False)
        ]

    def test_rejects_merge_without_overlap(self):
        coordinator = _coordinator(1)
        outcome = coordinator.on_merge_request(0.0, MergeRequest(3, frozenset({1})))
        assert outcome.replies == [MergeResponse(3, accept=False)]

    def test_requests_are_rate_limited_per_peer(self):
        coordinator = _coordinator(1)
        sensing = PositionSensing([[0, 0, 1, 0]], sensors=[1, 2])
        for k in range(1, 11):
            coordinator.update(0.5 * k, sensing.releases(0.5 * k))
        tracks = coordinator.summary.partitions[0].sensor_track_ids
        peer = NodeSummary(
            node_id=2,
            partitions=(PartitionSummary(tracks, {}),),
            active_sensor_track_ids=tracks | {99},
        )
        assert coordinator.on_summary(5.0, peer).replies
        assert not coordinator.on_summary(6.0, peer).replies
        assert coordinator.on_summary(11.0, peer).replies


class Bus:
    """Delivers each node's messages to its peers in FIFO order."""

    def __init__(self, coordinators):
        self.nodes = {c.node.node_id: c for c in coordinators}
        self.queue: deque = deque()

    def share(self, sender, summary):
        if summary is not None:
            for peer in self.nodes:
                if peer != sender:
                    self.queue.append((sender, peer, summary))

    def drain(self, time):
        while self.queue:
            sender, receiver, message = self.queue.popleft()
            node = self.nodes[receiver]
            if isinstance(message, NodeSummary):
                outcome = node.on_summary(time, message)
            elif isinstance(message, DedupRequest):
                outcome = node.on_dedup_request(time, message)
            elif isinstance(message, DedupResponse):
                node.on_dedup_response(message)
                continue
            elif isinstance(message, MergeRequest):
                outcome = node.on_merge_request(time, message)
            else:
                outcome = node.on_merge_response(time, message)
            for reply in outcome.replies:
                self.queue.append((receiver, sender, reply))
            self.share(receiver, outcome.summary)


def test_two_nodes_with_overlapping_tracks_deduplicate():
    targets = [[0.0, 0.0, 1.0, 0.0], [50.0, 50.0, 0.0, -1.0]]
    sensing = PositionSensing(targets, sensors=[1, 2])
    a, b = _coordinator(1), _coordinator(2)
    bus = Bus([a, b])
    for k in range(1, 61):
        time = 0.5 * k
        if k == 5:
            sensing.sensors = [1, 2, 3]
        releases = sensing.releases(time)
        # Both nodes get sensors 1 and 2; only node 2 gets sensor 3.
        bus.share(1, a.update(time, [r for r in releases if r.sensor_id != 3]))
        bus.share(2, b.update(time, releases))
        bus.drain(time)

    # The nodes split the work: no sensor track is used at both, and each
    # target is tracked exactly once across the network.
    assert not (a.summary.active_sensor_track_ids & b.summary.active_sensor_track_ids)
    released = [
        h.sensor_track_ids
        for node in (a, b)
        for h in node.node.released_tracks.values()
    ]
    assert sorted({t % 100 for t in tracks} for tracks in released) == [{0}, {1}]


def test_properly_intersecting_partitions_merge_at_one_node():
    sensing = PositionSensing([[0.0, 0.0, 1.0, 0.0]], sensors=[1, 2, 3])
    a, b = _coordinator(1), _coordinator(2)
    bus = Bus([a, b])
    default_routes = {1: {1, 2}, 2: {2, 3}}
    for k in range(1, 41):
        time = 0.5 * k
        releases = sensing.releases(time)
        for coordinator in (a, b):
            node = coordinator.node
            # Sensors route to their default nodes, and to any node that took
            # over one of their tracks in a merge.
            routed = [
                r
                for r in releases
                if r.sensor_id in default_routes[node.node_id]
                or r.sensor_track_id in node.sensor_tracks
            ]
            bus.share(node.node_id, coordinator.update(time, routed))
        bus.drain(time)

    # Node 1 (the fusor) resampled the union; node 2 relinquished its partition.
    assert [p.sensor_track_ids for p in a.summary.partitions] == [{100, 200, 300}]
    assert not b.summary.partitions
    (track,) = a.node.released_tracks.values()
    assert track.sensor_track_ids == {100, 200, 300}


def test_subset_holder_reshares_so_the_superset_rule_fires():
    a, b = _coordinator(1), _coordinator(2)
    sensing = PositionSensing([[0.0, 0.0, 1.0, 0.0]], sensors=[1, 2])
    # The nodes build their partitions without exchanging anything: A sees
    # sensors 1 and 2, B also sees sensor 3 (staggered so it joins).
    for k in range(1, 31):
        time = 0.5 * k
        if k == 9:
            sensing.sensors = [1, 2, 3]
        releases = sensing.releases(time)
        a.update(time, [r for r in releases if r.sensor_id != 3])
        b.update(time, releases)
    assert [p.sensor_track_ids for p in a.summary.partitions] == [{100, 200}]
    assert [p.sensor_track_ids for p in b.summary.partitions] == [{100, 200, 300}]

    # B shares its superset. A has no rule to act on it, so it re-shares ...
    (reply,) = a.on_summary(15.0, b.summary).replies
    assert reply == a.summary
    # (rate limited like requests)
    assert not a.on_summary(15.5, b.summary).replies
    # ... which lets B's superset rule fire, and A accepts.
    (request,) = b.on_summary(15.0, reply).replies
    assert request.reason is DedupReason.HAVE_SUPER_PARTITION
    outcome = a.on_dedup_request(15.0, request)
    assert outcome.replies == [DedupResponse(request.request_id, accept=True)]
    assert not a.summary.partitions


def test_no_reshare_without_a_peer_superset():
    coordinator = _coordinator(1)
    sensing = PositionSensing([[0, 0, 1, 0]], sensors=[1, 2])
    for k in range(1, 11):
        coordinator.update(0.5 * k, sensing.releases(0.5 * k))
    tracks = coordinator.summary.partitions[0].sensor_track_ids
    # An equal partition is settled by the equal-partition rule instead.
    replies = coordinator.on_summary(5.0, _summary(2, tracks)).replies
    assert replies and not any(isinstance(r, NodeSummary) for r in replies)
