"""Configuration for a hypothesis fusion node.

Distances and residuals (``*_gate``, ``*_residual``, ``max_distance``) are in
whatever units the application's
:class:`~hypothesis_fusion.initialization.Initializer` and ``position_of``
produce, so they have no defaults. Defaults elsewhere are unit-free values
that worked well in the authors' experiments.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class GrowthConfig:
    """Hypothesis generation as sensor tracks arrive (MFI Sec. III-C)."""

    # Max ``Initializer.track_residual`` between a partition's best hypothesis
    # and a new sensor track for the track to join (extend) that partition.
    partition_gate: float
    # Max pairwise ``Initializer.residual`` to initialize a stereo root hypothesis.
    pair_gate: float
    # Measurements a sensor track needs before it is used for growth.
    min_track_measurements: int = 3
    # Seconds between growth retries on an orphaned sensor track (one with no
    # hypotheses) that keeps releasing measurements. <= 0 disables retries.
    orphan_retry_interval: float = 5.0
    # Shrinking: when a hypothesis is about to be pruned and exactly one of its
    # active sensor tracks has an RMS innovation distance above this, spawn a
    # child without that track and with its recent updates undone. None
    # disables shrinking.
    shrink_innovation_threshold: float | None = 3.0


@dataclasses.dataclass(frozen=True)
class ScoreConfig:
    """Hypothesis scoring."""

    # Recent updates per sensor track averaged into its likelihood, |M(s_i)|.
    window: int = 3
    # Linear bias per sensor track, beta.
    size_bias: float = 5.0


@dataclasses.dataclass(frozen=True)
class PruneConfig:
    """Hypothesis pruning (MFI Sec. III-E)."""

    # Hypotheses scoring below this are pruned.
    min_score: float = -200.0
    # Per partition, keep ``base + per_best_squared * N**2`` hypotheses, where N
    # is the number of best hypotheses in the partition. The quadratic term
    # lets unresolved clusters explore more combinations.
    base: int = 3
    per_best_squared: float = 5.0
    # Housekeeping: prune hypotheses whose latest estimate is older than this
    # many seconds (no recent measurement updates).
    inactivity_window: float = 15.0


@dataclasses.dataclass(frozen=True)
class TrackIdConfig:
    """Hypothesis-to-track assignment."""

    # Max distance between ``position_of`` the hypothesis and the previous
    # assignment for the positional fallback match.
    max_distance: float
    # Min active sensor track overlap to keep a track ID.
    min_overlap: float = 0.6


@dataclasses.dataclass(frozen=True)
class ReleaseConfig:
    """Track release."""

    # Seconds an assignment must hold before the track is released.
    maturity: float = 3.0
    # Min active sensor track overlap for a changed assignment to still hold.
    min_overlap: float = 0.6


@dataclasses.dataclass(frozen=True)
class ResampleConfig:
    """Hypothesis resampling."""

    # Max pairwise ``Initializer.residual`` for an adjacency graph edge.
    max_pair_residual: float
    # Max N-way ``Initializer.residual`` for a candidate hypothesis.
    max_clique_residual: float
    # Candidate hypotheses kept per resample.
    k: int = 10
    # Smallest candidate hypothesis (2 = stereo).
    min_size: int = 2
    # Per-track bonus when ranking candidates, so larger cliques survive the
    # k-best cutoff despite fitting more residual.
    size_bias: float = 2.0
    # Resample a partition whose best hypotheses' utilization (overlap with the
    # partition's sensor tracks) is below this, c. 0 disables the trigger.
    utilization_threshold: float = 0.89
    # Resample a partition (of 3+ tracks) whose redundancy is at or below this,
    # rho. The default fires only when no track has a fallback hypothesis.
    redundancy_threshold: float = 0.0
    # Min seconds between automatic resamples of the same sensor tracks.
    cooldown: float = 3.0
    # Seconds of measurement history sent with sensor tracks in a partition merge.
    merge_history: float = 10.0


@dataclasses.dataclass(frozen=True)
class FusionConfig:
    growth: GrowthConfig
    track_id: TrackIdConfig
    score: ScoreConfig = ScoreConfig()
    prune: PruneConfig = PruneConfig()
    release: ReleaseConfig = ReleaseConfig()
    # None disables resampling (and so partition merging in the distributed
    # protocol).
    resample: ResampleConfig | None = None
    # Mark a sensor track inactive after this many seconds without a release.
    # <= 0 disables the timeout (an explicit terminated release still applies).
    inactive_timeout: float = 3.0
