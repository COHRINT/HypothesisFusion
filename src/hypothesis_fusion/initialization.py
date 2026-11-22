"""Application interfaces for relating sensor tracks and initializing hypotheses.

Hypothesis fusion assumes a single sensor track does not observe the full
target state -- e.g. bearings-only sensors observe no range -- so a hypothesis
needs at least two sensor tracks (a "stereo" view) before it can be initialized.
How sensor tracks are related to each other and to a state estimate, and how a
filter is initialized from them, depends on the sensors and the state space,
so the application provides it through :class:`Initializer`.

For bearings sensors, for example, a sensor track defines a line of sight from
the sensor; the residual of a set of tracks is how far their lines of sight
miss a common intersection, and initialization triangulates that intersection
and filters over the tracks' recent measurements.
"""

from collections.abc import Sequence
from typing import Protocol

from hypothesis_fusion.filtering import HypothesisFilter
from hypothesis_fusion.types import Estimate, SensorTrack


class Initializer(Protocol):
    def residual(self, tracks: Sequence[SensorTrack], time: float) -> float | None:
        """How inconsistent ``tracks`` (from distinct sensors) are with observing
        one common target at ``time``; lower is more consistent.

        Return ``None`` when the tracks cannot observe a common target at all
        (e.g. their lines of sight intersect behind a sensor or are parallel).
        Used to gate stereo pairs during growth and to build candidate
        hypotheses during resampling.
        """
        ...

    def track_residual(self, estimate: Estimate, track: SensorTrack) -> float:
        """How far a hypothesis estimate (already predicted to ``track.time``) is
        from what ``track`` observes, gated by ``GrowthConfig.partition_gate``.
        Used to associate a new sensor track to an existing partition, so it
        should account for both uncertainties (e.g. a Mahalanobis distance): a
        raw distance lets a well-localized target claim any sensor track
        passing near it, such as one on a new target behind it."""
        ...

    def initialize(
        self, tracks: Sequence[SensorTrack], time: float
    ) -> HypothesisFilter | None:
        """A filter initialized from (and having processed the recent
        measurements of) ``tracks``, or ``None`` if they do not fit one target."""
        ...


class FeasibilityCheck(Protocol):
    """Optional application check pruning physically implausible hypotheses,
    e.g. infeasible velocities, or range uncertainty that
    has grown from single-sensor information."""

    def is_feasible(self, estimate: Estimate) -> bool: ...
