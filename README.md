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

Requires Python 3.11+, NumPy, and SciPy.

## Package layout

| Module | Contents | Paper |
| --- | --- | --- |
| `types` | `Measurement`, `SensorTrackRelease`, `SensorTrack`, `Estimate` | MFI II-B |
| `fusion` | `HypothesisFusion`: one fusion node's tracker | MFI III |
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
