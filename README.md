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

## Installation

```bash
uv sync --all-extras
```

```bash
uv run pytest
```

Requires Python 3.11+, NumPy, and SciPy.
