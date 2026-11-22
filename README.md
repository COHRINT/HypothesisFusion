# Hypothesis Fusion

A reference implementation of **Hypothesis Fusion**: multiple hypothesis
tracking over *sensor tracks*, designed for asynchronous, decentralized
multi-sensor networks, particularly in which no single sensor observes
a target's full state (e.g. bearings-only sensors).

- R. Draves, N. Ahmed, E. Borcherding, and J. Chan, “Asynchronous
Distributed Multi-Sensor Fusion Over Sensor Track Hypotheses,”
in IEEE International Conference on Multisensor Fusion and
Integration (MFI), 2026, to appear. arXiv:2610.20290. [Online].
Available: https://arxiv.org/abs/2610.20290

The implementation is application-agnostic and a demo provides an example
with robots using 1-D bearings sensors to estimate the positions and
velocities of several targets.

## The idea

Each sensor runs its own simple tracker and produces **sensor tracks** in its
local frame (for bearings sensors: bearing and bearing rate), releasing new
measurements tagged with the sensor track's ID. A fusion node then only has to
decide *which sensor tracks observe the same target* rather than associate every
measurement. Each **hypothesis** is a set of sensor tracks (at most one per
sensor, at least two for observability) with a filter over all of their
measurements, scored by how well those measurements agree.

```
sensor tracks ──► growth ──► filtering ──► shrinking ──► pruning ──► best hypotheses ──► track IDs ──► release
 (releases)         ▲                                                                                     │
                    └──────────────────────── resampling (unhealthy partitions) ◄─────────────────────────┘
```

Hypotheses that share sensor tracks form **partitions**: independent
subproblems that are grown, pruned, and resampled separately. Because a
partition is summarized by the integer IDs of its sensor tracks, fusion nodes
deduplicate their work with set operations instead of exchanging estimates.

## Installation

```bash
uv sync --all-extras
```

```bash
uv run pytest
```

Requires Python 3.11+, NumPy, and SciPy; the demo's plots need matplotlib
(`--extra demo`).

## Package layout

| Module | Contents | Paper |
| --- | --- | --- |
| `types` | `Measurement`, `SensorTrackRelease`, `SensorTrack`, `Estimate` | MFI II-B |
| `fusion` | `HypothesisFusion`: one fusion node's tracker | MFI III |
| `distributed` | Dedup/merge rules and `FusionCoordinator` | MFI III-H |
| `partitions` | Partitioning, utilization, redundancy | MFI III-B |
| `scoring` | Windowed innovation-likelihood scoring | — |
| `resampling` | Bron–Kerbosch clique resampling and reconciliation | — |
| `track_ids` | Hypothesis-to-track assignment | — |
| `release` | Gated track release | — |
| `filtering` | `HypothesisFilter` protocol, `UnscentedFilter` | MFI III-G |
| `initialization` | `Initializer`, `FeasibilityCheck` protocols | MFI III-C |
| `models` | `DynamicsModel`, `MeasurementModel` protocols | MFI Eqs. 1–2 |
| `config` | Parameters, with experiment-tested defaults where unit-free | — |

## Using it in an application

An application provides:

1. **Sensor track releases** (`SensorTrackRelease`): each sensor track's new
   measurements plus its local estimate. Sensor track IDs and measurement IDs
   must be unique network-wide.
2. **An `Initializer`** — the stereo initialization mechanism:
   - `residual(tracks, time)`: how inconsistent 2+ sensor tracks are with one
     common target (`None` if geometrically impossible). For bearings: the RMS
     miss distance of the lines of sight from their intersection.
   - `track_residual(estimate, track)`: distance from a hypothesis estimate to
     what a sensor track observes (bearings: Mahalanobis distance between the
     predicted and observed bearing).
   - `initialize(tracks, time)`: a `HypothesisFilter` started from those tracks
     and run over their recent measurements, or `None` if they do not fit.
3. **`position_of(state)`**, extracting position from a state vector (used for
   positional matching).
4. Optionally a **`FeasibilityCheck`** (e.g. speed and position-uncertainty
   limits) for housekeeping.

For filters, either use `UnscentedFilter` with your `DynamicsModel` and
`MeasurementModel` (the measurement model's `residual` handles angle wrapping),
or implement `HypothesisFilter` yourself. `UnscentedFilter` absorbs
out-of-sequence measurements by rewinding and replaying a bounded history.

```python
from hypothesis_fusion.config import (
    FusionConfig,
    GrowthConfig,
    ResampleConfig,
    TrackIdConfig,
)
from hypothesis_fusion.fusion import HypothesisFusion

config = FusionConfig(
    # Gates are in the units of the Initializer's residuals (here, sigmas
    # for track_residual and meters for residual) ...
    growth=GrowthConfig(partition_gate=3.0, pair_gate=30.0),
    resample=ResampleConfig(
        max_pair_residual=30.0, max_clique_residual=15.0, size_bias=10.0
    ),
    # ... and distances in the units of position_of.
    track_id=TrackIdConfig(max_distance=30.0),
)
node = HypothesisFusion(
    config, my_initializer, position_of=lambda x: x[:2], feasibility=my_feasibility
)

for time, releases in my_sensor_stream:
    # One frame of the algorithm, then prune stale / infeasible hypotheses.
    node.update(time, releases)
    node.housekeeping(time)
    for track_id, hypothesis in node.released_tracks.items():
        estimate = hypothesis.filter.latest

# The estimate history of every track ever released.
histories = node.track_histories
```

### Decentralized fusion

Wrap each node in a `FusionCoordinator`. It is transport-agnostic: share
`coordinator.summary` (returned by `update`/`housekeeping` whenever it changes)
with peers, and route incoming messages to the matching handler:

| Message | Handler | Effect |
| --- | --- | --- |
| `NodeSummary` | `on_summary` | Returns dedup / merge requests to send back, or its own summary if the peer holds a superset it would ask to prune |
| `DedupRequest` | `on_dedup_request` | Accepts (prunes and ignores the tracks) or rejects |
| `DedupResponse` | `on_dedup_response` | Clears the outstanding request |
| `MergeRequest` | `on_merge_request` | Hands over its tracks and relinquishes the partition |
| `MergeResponse` | `on_merge_response` | Adopts the tracks and resamples the union |

Each handler returns an `Outcome` of replies (to the sender) and, if the node
changed, a summary to re-share. The application decides which peers to share
with and must route a sensor track's future releases to whichever node now
holds it.

## Demo: 2D bearings-only robots

`examples/bearings2d` is a small world in which four robots drive
predetermined arcs around an area and take noisy bearings (σ = 0.17°) of
five constant-velocity targets that appear in three waves, including a
closely spaced pair. Robots see targets within 1 km and sample
asynchronously: every 0.5 s, with their phases staggered across the period,
and each sample is processed by the fusion nodes as its own frame. Sensor
tracks are perfect in association (one per robot per target in view), each
with a small bearing/bearing-rate Kalman filter. The
initializer triangulates the lines of sight, then filters through 10 s of the
tracks' measurements and gates on the mean innovation distance.

```bash
uv run python -m examples.bearings2d --plot outputs
```

```bash
uv run python -m examples.bearings2d --mode centralized --release-delay 0.25
```

The decentralized run uses three fusion nodes, with each robot reporting to two
of them, so every pair of nodes must deduplicate. Typical results (seeds 0–3,
after the first 30 s):

| Metric | Centralized | Decentralized |
| --- | --- | --- |
| Position RMSE | 2.9 m | 2.7 m |
| Targets in custody (20 m) | 98% | 96% |
| Tracks per tracked target | 1.00 | 1.01 |
| Track purity | 0.96 | 1.00 |
| Track ID changes (whole run) | 0.8 | 1.5 |

The decentralized solution closely follows the
centralized one, with more track churn while nodes resolve their overlap.
Both settle on one pure track per target, nearly always fusing all four
robots.

Three behaviors are visible in the demo. When a wave of targets appears, every
robot's sensor track on a target matures within one sampling period; a track
that matures before its partition's newest pair hypothesis has been scored
cannot join that partition and may briefly form a duplicate pair, which
resolves within seconds. Any two non-parallel lines intersect in 2D, so a
*pair* of sensor tracks can only be rejected over time (the initializer's
innovation gate) or by a third sensor. And when two targets line up along a
robot's line of sight, swapped pairings fit nearly as well as the true ones;
the release gate then withholds the affected tracks until the best hypotheses
stabilize.

The initializer's `track_residual` is a Mahalanobis distance for a reason: when
the last target appears, it sits directly behind another target from one
robot's view, a few meters off that target's line of sight. A raw distance gate
lets the nearby target's partition claim the new sensor track, and resampling
then mixes the two targets' tracks for half a minute. Measured against both
uncertainties, the near miss falls outside the 3-sigma gate and the track starts its own
partition.

A known limitation: if the cold-start window leaves one target split across
two *disjoint* partitions (e.g. robots 1+2 and robots 3+4), nothing joins them
-- resampling works within a partition, and both partitions look healthy -- so
the target keeps two tracks. In the demo this happens when release delays
(`--release-delay`) reach about half a sampling period, which widens the
window.

## Departures from the MFI paper

- **Filtering.** `UnscentedFilter` handles out-of-sequence measurements by
  rewind-and-replay over a bounded history rather than CISI-FPS smoothing.
- **Subset re-sharing.** Only the node holding a superset partition can ask
  the other to prune, and it checks when the subset holder's summary arrives.
  If the superset forms after the subset holder last shared, and the subset
  holder's partitions then stop changing, the request would never be made
  and both nodes would keep a track. So a node that receives a summary showing
  the sender would make that request replies with its own summary
  (rate-limited like requests).
