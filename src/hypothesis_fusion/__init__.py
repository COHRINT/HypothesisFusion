"""Hypothesis Fusion: multiple hypothesis tracking over sensor tracks.

A fusion node receives *sensor tracks* -- sensor-level tracks that associate
measurements over time in each sensor's local frame -- and forms hypotheses over
which sensor tracks observe the same target. Each hypothesis runs a filter over
the measurements of its sensor tracks and is scored by how well those
measurements agree. Hypotheses are grouped into independent *partitions*, from
which best-guess system tracks are selected, given stable track IDs, and
released once unambiguous. Because hypotheses are identified by the integer IDs
of their sensor tracks, fusion nodes can deduplicate their work with simple set
operations (see :mod:`hypothesis_fusion.distributed`).

References:
    R. Draves, N. Ahmed, E. Borcherding, J. Chan, "Asynchronous Distributed
    Multi-Sensor Fusion Over Sensor Track Hypotheses," MFI 2026.
"""

__version__ = '0.1.0'
