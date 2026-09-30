"""Deterministic tie-breaking between adjacent mirrored motion ranges."""

from __future__ import annotations

import numpy as np

from motion_matching.core.database import (
    _mirror_frame_index,
    motion_matching_search_fixed,
    motion_matching_search_w_custom_ranges,
)


def _search(features: np.ndarray, **kwargs) -> tuple[int, float]:
    return motion_matching_search_fixed(
        best_index=-1,
        best_cost=np.inf,
        range_starts=np.array([0, 4]),
        range_stops=np.array([4, 8]),
        features=features,
        features_offset=np.zeros(1),
        features_scale=np.ones(1),
        query_normalized=np.zeros(1),
        ignore_range_end=0,
        ignore_surrounding=0,
        **kwargs,
    )


def test_mirror_frame_uses_same_offset_in_adjacent_range() -> None:
    starts = np.array([0, 4, 8, 12])
    stops = np.array([4, 8, 12, 16])
    assert _mirror_frame_index(2, starts, stops) == 6
    assert _mirror_frame_index(6, starts, stops) == 2
    assert _mirror_frame_index(9, starts, stops) == 13


def test_near_tie_prefers_earlier_mirror() -> None:
    features = np.full((8, 1), 10.0)
    features[2, 0] = np.sqrt(1.00005)  # earlier mirror, slightly higher cost
    features[6, 0] = 1.0              # true minimum
    best, cost = _search(features)
    assert best == 2
    assert cost == features[2, 0] ** 2


def test_meaningful_cost_difference_keeps_true_minimum() -> None:
    features = np.full((8, 1), 10.0)
    features[2, 0] = np.sqrt(1.001)
    features[6, 0] = 1.0
    best, cost = _search(features)
    assert best == 6
    assert cost == 1.0


def test_unequal_adjacent_ranges_are_not_treated_as_mirrors() -> None:
    assert _mirror_frame_index(
        1,
        np.array([0, 3]),
        np.array([3, 8]),
    ) is None


def test_custom_search_mask_blocks_disallowed_mirror() -> None:
    features = np.full((8, 1), 10.0)
    features[2, 0] = np.sqrt(1.00005)
    features[6, 0] = 1.0
    best, cost, matched_range, ending_frame = motion_matching_search_w_custom_ranges(
        range_starts=np.array([0, 4]),
        range_stops=np.array([4, 8]),
        features=features,
        features_offset=np.zeros(1),
        features_scale=np.ones(1),
        query_normalized=np.zeros(1),
        ignore_range_end=0,
        custom_search_ranges=[(4, 8, 8)],
    )
    assert best == 6
    assert cost == 1.0
    assert matched_range == 0
    assert ending_frame == 8


def test_tie_break_can_be_disabled() -> None:
    features = np.full((8, 1), 10.0)
    features[2, 0] = np.sqrt(1.00005)
    features[6, 0] = 1.0
    best, cost = _search(features, mirror_tie_epsilon=None)
    assert best == 6
    assert cost == 1.0
