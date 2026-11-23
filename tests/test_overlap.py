import pytest

from hypothesis_fusion.overlap import identity_overlap, overlap


def test_identical_sets_overlap_fully():
    assert overlap({1, 2, 3}, {1, 2, 3}) == 1.0


def test_disjoint_sets_do_not_overlap():
    assert overlap({1, 2}, {3, 4}) == 0.0


def test_empty_set_does_not_overlap():
    assert overlap(set(), {1}) == 0.0
    assert overlap(set(), set()) == 0.0


def test_dice_coefficient():
    assert overlap({1, 2}, {1, 2, 3, 4}) == pytest.approx(2 / 3)
    assert overlap({1}, {1, 2, 3}) == pytest.approx(0.5)


def test_overlap_is_symmetric():
    assert overlap({1, 2, 5}, {2, 5, 7, 9}) == overlap({2, 5, 7, 9}, {1, 2, 5})


def test_identity_overlap_uses_active_tracks():
    # The pedigrees differ only in inactive tracks, which are ignored.
    assert identity_overlap({1, 2}, {1, 2, 8}, {1, 2}, {1, 2, 9}) == 1.0


def test_identity_overlap_falls_back_to_pedigree():
    assert identity_overlap(set(), {1, 2}, {1}, {1, 2}) == 1.0
