"""Runs the demo: sensing, routing, fusion nodes, and the message bus.

Everything is a discrete event. Robots sample asynchronously (once per ``dt``,
with staggered phases); each sample's sensor track releases are routed to
fusion nodes, optionally with random delays that produce out-of-sequence
arrivals. On each sample, every node processes whatever has arrived as its own
frame and shares its summary if it changed; once per period, nodes run
housekeeping and their released tracks are recorded for evaluation. Protocol
messages between nodes are delivered after a fixed latency.

Routing models the sensing layer: each robot sends to a fixed set of fusion
nodes, plus any node currently holding one of its sensor tracks (e.g. one that
took the track over in a partition merge).
"""

import dataclasses
import heapq
import itertools
from collections import defaultdict

import numpy as np

from examples.bearings2d.initializer import BearingsFeasibility, BearingsInitializer
from examples.bearings2d.models import ConstantVelocity
from examples.bearings2d.scenario import Scenario
from examples.bearings2d.sensing import Sensing
from hypothesis_fusion.config import (
    FusionConfig,
    GrowthConfig,
    PruneConfig,
    ResampleConfig,
    TrackIdConfig,
)
from hypothesis_fusion.distributed import (
    DedupRequest,
    DedupResponse,
    FusionCoordinator,
    MergeRequest,
    Outcome,
)
from hypothesis_fusion.fusion import HypothesisFusion
from hypothesis_fusion.summary import NodeSummary
from hypothesis_fusion.types import SensorTrackRelease

CENTRALIZED_ROUTES = {1: {0}, 2: {0}, 3: {0}, 4: {0}}
# Every robot reports to two of three fusion nodes, so every pair of nodes
# shares some sensors and must deduplicate.
DECENTRALIZED_ROUTES = {1: {0, 1}, 2: {1, 2}, 3: {2, 0}, 4: {0, 2}}


def default_config() -> FusionConfig:
    return FusionConfig(
        growth=GrowthConfig(partition_gate=3.0, pair_gate=30.0),
        track_id=TrackIdConfig(max_distance=30.0),
        prune=PruneConfig(inactivity_window=10.0),
        # Triangulation residuals are in meters: the size bias must outweigh a
        # true clique's residual (a few meters) but not a ghost's.
        resample=ResampleConfig(
            max_pair_residual=30.0, max_clique_residual=15.0, size_bias=10.0
        ),
    )


def make_node(
    scenario: Scenario, node_id: int, config: FusionConfig
) -> HypothesisFusion:
    initializer = BearingsInitializer(
        robot_position=lambda robot_id, t: scenario.robot(robot_id).position(t),
        dynamics=ConstantVelocity(q=0.5),
        sensing_range=scenario.sensing_range,
    )
    return HypothesisFusion(
        config,
        initializer,
        position_of=lambda x: x[:2],
        feasibility=BearingsFeasibility(max_speed=20.0, max_position_sigma=150.0),
        node_id=node_id,
    )


@dataclasses.dataclass(frozen=True)
class TrackSnapshot:
    """A released track at one step, as reported by one node."""

    time: float  # simulation step
    estimate_time: float
    node_id: int
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    sensor_track_ids: frozenset[int]


@dataclasses.dataclass
class RunResult:
    scenario: Scenario
    snapshots: list[TrackSnapshot]
    target_of_sensor_track: dict[int, int]
    nodes: list[HypothesisFusion]
    message_counts: dict[str, int]


def run(
    scenario: Scenario,
    routes: dict[int, set[int]],
    config: FusionConfig | None = None,
    seed: int = 0,
    latency: float = 0.1,
    max_release_delay: float = 0.0,
) -> RunResult:
    """Simulate the scenario with fusion nodes fed according to ``routes``
    (robot ID -> fusion node IDs)."""
    config = config or default_config()
    node_ids = sorted(set().union(*routes.values()))
    coordinators = {
        n: FusionCoordinator(make_node(scenario, n, config)) for n in node_ids
    }
    sensing = Sensing(scenario, seed=seed)
    rng = np.random.default_rng(seed + 1)

    # (time, sequence, receiver, sender, message)
    events: list[tuple[float, int, int, int, object]] = []
    sequence = itertools.count()
    inbox: dict[int, list[tuple[float, SensorTrackRelease]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    snapshots: list[TrackSnapshot] = []

    def send(time: float, sender: int, receiver: int, message: object) -> None:
        counts[type(message).__name__] += 1
        heapq.heappush(
            events, (time + latency, next(sequence), receiver, sender, message)
        )

    def share(time: float, sender: int, summary: NodeSummary | None) -> None:
        if summary is not None:
            for peer in node_ids:
                if peer != sender:
                    send(time, sender, peer, summary)

    def deliver(time: float, receiver: int, sender: int, message: object) -> None:
        coordinator = coordinators[receiver]
        if isinstance(message, NodeSummary):
            outcome = coordinator.on_summary(time, message)
        elif isinstance(message, DedupRequest):
            outcome = coordinator.on_dedup_request(time, message)
        elif isinstance(message, DedupResponse):
            coordinator.on_dedup_response(message)
            outcome = Outcome()
        elif isinstance(message, MergeRequest):
            outcome = coordinator.on_merge_request(time, message)
        else:
            outcome = coordinator.on_merge_response(time, message)
        for reply in outcome.replies:
            send(time, receiver, sender, reply)
        share(time, receiver, outcome.summary)

    last_robot = scenario.robots[-1].id
    for time, robot_id in scenario.sample_times():
        while events and events[0][0] <= time:
            event_time, _, receiver, sender, message = heapq.heappop(events)
            deliver(event_time, receiver, sender, message)

        for release in sensing.releases(robot_id, time):
            destinations = set(routes[release.sensor_id]) | {
                n
                for n, c in coordinators.items()
                if release.sensor_track_id in c.node.sensor_tracks
            }
            for n in destinations:
                delay = (
                    rng.uniform(0.0, max_release_delay) if max_release_delay else 0.0
                )
                inbox[n].append((time + delay, release))

        # Each node processes whatever has arrived as its own frame.
        for n, coordinator in coordinators.items():
            arrived = [r for t, r in inbox[n] if t <= time]
            inbox[n] = [(t, r) for t, r in inbox[n] if t > time]
            if arrived:
                share(time, n, coordinator.update(time, arrived))

        # Housekeeping and evaluation once per sampling period.
        if robot_id != last_robot:
            continue
        for n, coordinator in coordinators.items():
            share(time, n, coordinator.housekeeping(time))
            for track_id, hypothesis in coordinator.node.released_tracks.items():
                latest = hypothesis.filter.latest
                snapshots.append(
                    TrackSnapshot(
                        time=time,
                        estimate_time=latest.time,
                        node_id=n,
                        track_id=track_id,
                        mean=latest.mean,
                        covariance=latest.covariance,
                        sensor_track_ids=hypothesis.sensor_track_ids,
                    )
                )

    return RunResult(
        scenario=scenario,
        snapshots=snapshots,
        target_of_sensor_track=dict(sensing.target_of_sensor_track),
        nodes=[c.node for c in coordinators.values()],
        message_counts=dict(counts),
    )
