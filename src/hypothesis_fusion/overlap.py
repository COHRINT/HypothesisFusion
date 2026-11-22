"""The overlap measure for comparing sets of sensor track IDs.

Track-ID assignment, the release gate, and the resampling trigger all threshold
an overlap between sensor track sets. They share this module so every such
threshold means the same thing.

The measure is the Dice-Sorensen coefficient,
``2|A n B| / (|A| + |B|)``: the harmonic mean of the two containment fractions,
so it is symmetric and gives partial credit when the sets differ in size. At a
0.6 threshold a set can grow up to double (``{1,2} -> {1,2,3,4}`` scores 2/3)
and still count as the same.
"""

from collections.abc import Set


def overlap(a: Set[int], b: Set[int]) -> float:
    """Dice-Sorensen coefficient of ``a`` and ``b``; 0 if either is empty."""
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def identity_overlap(
    a_active: Set[int],
    a_all: Set[int],
    b_active: Set[int],
    b_all: Set[int],
) -> float:
    """How much two hypotheses look like the same track from frame to frame.

    Compared over the *active* sensor tracks: inactive tracks carry no new
    evidence but inflate both sets and their intersection. Falls back to the
    full pedigree when either side has no active tracks, since that is then the
    only identity left.
    """
    if not a_active or not b_active:
        return overlap(a_all, b_all)
    return overlap(a_active, b_active)
