"""The demo world: robots on predetermined paths observing moving targets.

Robots circle the area of interest at different radii and phases, so the
viewing geometry between any two of them changes over time. Each robot has a
limited sensing range, so targets enter and leave each robot's view and the
robots' sensor tracks start and end. Robots sample asynchronously: at the same
rate, but with staggered phases. Targets appear in waves; two targets of
the first wave start close together and separate slowly, which makes
sensor-track pairings ambiguous early on.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Robot:
    """A robot driving a circle: ``center + radius * [cos, sin](phase + rate t)``."""

    id: int
    center: tuple[float, float]
    radius: float
    phase: float
    rate: float  # rad/s

    def position(self, time: float) -> np.ndarray:
        angle = self.phase + self.rate * time
        return np.asarray(self.center) + self.radius * np.array(
            [np.cos(angle), np.sin(angle)]
        )


@dataclasses.dataclass(frozen=True)
class Target:
    """A constant-velocity target (``[x, y, vx, vy]`` at ``t = 0``) that can be
    sensed from ``appears`` seconds on."""

    id: int
    initial: tuple[float, float, float, float]
    appears: float = 0.0

    def present(self, time: float) -> bool:
        return time >= self.appears

    def state(self, time: float) -> np.ndarray:
        x, y, vx, vy = self.initial
        return np.array([x + vx * time, y + vy * time, vx, vy])


@dataclasses.dataclass(frozen=True)
class Scenario:
    robots: tuple[Robot, ...]
    targets: tuple[Target, ...]
    duration: float = 120.0
    dt: float = 0.5
    bearing_sigma: float = 0.003  # rad
    sensing_range: float = 1000.0  # m

    def robot(self, robot_id: int) -> Robot:
        return next(r for r in self.robots if r.id == robot_id)

    @property
    def times(self) -> np.ndarray:
        """The end of each sampling period of length ``dt``."""
        steps = int(round(self.duration / self.dt))
        return self.dt * np.arange(1, steps + 1)

    def sample_times(self) -> list[tuple[float, int]]:
        """Every ``(time, robot_id)`` sample, in time order.

        Each robot samples once per period, with the robots' phases evenly
        staggered across the period (the last robot samples at its end).
        """
        n = len(self.robots)
        return [
            (float(end - self.dt + (i + 1) * self.dt / n), robot.id)
            for end in self.times
            for i, robot in enumerate(self.robots)
        ]


def default_scenario() -> Scenario:
    center = (500.0, 500.0)
    robots = (
        Robot(1, center, 650.0, phase=0.0, rate=0.010),
        Robot(2, center, 700.0, phase=np.pi / 2, rate=-0.008),
        Robot(3, center, 600.0, phase=np.pi, rate=0.012),
        Robot(4, center, 750.0, phase=3 * np.pi / 2, rate=-0.006),
    )
    targets = (
        # First wave: a clustered pair that slowly separates.
        Target(0, (250.0, 350.0, 4.0, 2.0)),
        Target(1, (270.0, 380.0, 4.0, 2.6)),
        # Second wave.
        Target(2, (850.0, 150.0, -3.0, 4.0), appears=20.0),
        Target(3, (450.0, 1000.0, 1.0, -5.0), appears=25.0),
        # Third wave.
        Target(4, (900.0, 750.0, -4.0, -1.0), appears=50.0),
    )
    return Scenario(robots=robots, targets=targets)
