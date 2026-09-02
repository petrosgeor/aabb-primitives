import numpy as np
import pytest

from aabb_primitives import AABBAxis, AABBFace, PairwiseAABBRelations


def test_rank_four_worlds_keep_environment_and_world_dimensions() -> None:
    """Keep environment and world axes outside each pairwise matrix."""
    world_aabbs = np.array(
        [
            [
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0, 2.0, 1.0]],
            ]
        ],
        dtype=np.float64,
    )

    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=world_aabbs, query_aabbs=world_aabbs)

    signed_distances = relations.signed_distances_all_faces()
    assert signed_distances.shape == (1, 2, 2, 2, 6)
    np.testing.assert_array_equal(signed_distances[0, 0, 0, 1], np.array([-2.0, 0.0, -1.0, -1.0, -1.0, -1.0]))
    np.testing.assert_array_equal(signed_distances[0, 1, 0, 1], np.array([-1.0, -1.0, -2.0, 0.0, -1.0, -1.0]))


def test_rank_four_methods_match_independent_rank_two_worlds() -> None:
    """Keep every geometry calculation local to one environment and world."""
    first_environment = np.array(
        [
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0, 1.0, 1.0], [0.25, 0.25, 1.0, 0.75, 0.75, 2.0]],
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0, 2.0, 1.0], [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]],
        ],
        dtype=np.float64,
    )
    x_shift = np.array([5.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    reference_aabbs = np.stack((first_environment, first_environment + x_shift), axis=0)
    query_aabbs = reference_aabbs[..., :2, :]

    batched = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)
    method_calls = {
        "signed_distances": lambda relations: relations.signed_distances(AABBFace.X_MAX),
        "axis_overlap": lambda relations: relations.axis_overlap(AABBAxis.X),
        "axis_overlap_all_axes": lambda relations: relations.axis_overlap_all_axes(),
        "contained_by_mask": lambda relations: relations.contained_by_mask(),
        "intersection_bounds": lambda relations: relations.intersection_bounds(AABBAxis.Y),
        "intersection_bounds_all_axes": lambda relations: relations.intersection_bounds_all_axes(),
        "overlap_lengths": lambda relations: relations.overlap_lengths(AABBAxis.Z),
        "overlap_lengths_all_axes": lambda relations: relations.overlap_lengths_all_axes(),
        "tangential_overlap_lengths": lambda relations: relations.tangential_overlap_lengths(AABBFace.Z_MIN),
        "projected_overlap_mask": lambda relations: relations.projected_overlap_mask(AABBFace.X_MIN),
        "inward_projected_overlap_mask": lambda relations: relations.inward_projected_overlap_mask(
            AABBFace.Y_MAX, inset=0.01
        ),
        "inward_axis_overlap_all_axes": lambda relations: relations.inward_axis_overlap_all_axes(inset=0.01),
        "projected_overlap_areas": lambda relations: relations.projected_overlap_areas(AABBFace.Z_MIN),
        "query_face_areas": lambda relations: relations.query_face_areas(AABBFace.Z_MIN),
        "projected_intersection_bounds": lambda relations: relations.projected_intersection_bounds(AABBFace.Z_MIN),
        "within_distance": lambda relations: relations.within_distance(
            AABBFace.X_MAX, minimum_distance=-0.01, maximum_distance=0.01
        ),
        "contact_mask": lambda relations: relations.contact_mask(AABBFace.Z_MIN, distance_tolerance=0.01),
    }

    for method_name, call_method in method_calls.items():
        expected = np.stack(
            [
                np.stack(
                    [
                        call_method(
                            PairwiseAABBRelations.from_aabbs(
                                reference_aabbs=reference_aabbs[environment_index, world_index],
                                query_aabbs=query_aabbs[environment_index, world_index],
                            )
                        )
                        for world_index in range(reference_aabbs.shape[1])
                    ],
                    axis=0,
                )
                for environment_index in range(reference_aabbs.shape[0])
            ],
            axis=0,
        )
        np.testing.assert_array_equal(call_method(batched), expected, err_msg=method_name)


def test_rank_four_query_thresholds_align_with_environment_world_and_query() -> None:
    """Apply each batched threshold only to its aligned query row."""
    reference_aabbs = np.array([[[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]], dtype=np.float64)
    query_aabbs = np.array(
        [
            [
                [[1.0, 0.0, 0.0, 2.0, 1.0, 1.0], [1.2, 0.0, 0.0, 2.2, 1.0, 1.0]],
                [[1.2, 0.0, 0.0, 2.2, 1.0, 1.0], [1.4, 0.0, 0.0, 2.4, 1.0, 1.0]],
            ]
        ],
        dtype=np.float64,
    )
    maximum_distances = np.array([[[0.0, 0.1], [0.25, 0.5]]], dtype=np.float64)
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)
    expected = np.array([[[[True], [False]], [[True], [True]]]], dtype=bool)

    np.testing.assert_array_equal(
        relations.within_distance(
            AABBFace.X_MIN, minimum_distance=np.zeros((1, 2, 2), dtype=np.float64), maximum_distance=maximum_distances
        ),
        expected,
    )
    np.testing.assert_array_equal(
        relations.contact_mask(AABBFace.X_MIN, distance_tolerance=maximum_distances), expected
    )
    with pytest.raises(ValueError, match=r"shape \(1, 2, 2\)"):
        relations.contact_mask(AABBFace.X_MIN, distance_tolerance=np.zeros((2,), dtype=np.float64))


def test_rank_four_inputs_reject_implicit_environment_or_world_broadcasting() -> None:
    """Require query and reference tensors to name the same explicit worlds."""
    query_aabbs = np.zeros((2, 3, 1, 6), dtype=np.float64)

    with pytest.raises(ValueError, match="matching environment and world dimensions"):
        PairwiseAABBRelations.from_aabbs(
            reference_aabbs=np.zeros((2, 1, 1, 6), dtype=np.float64), query_aabbs=query_aabbs
        )

    with pytest.raises(ValueError, match="same rank"):
        PairwiseAABBRelations.from_aabbs(reference_aabbs=np.zeros((1, 6), dtype=np.float64), query_aabbs=query_aabbs)


def test_projected_overlap_mask_checks_only_axes_in_selected_face() -> None:
    """Show that a Z-face projection checks X and Y but ignores Z distance."""
    reference_aabbs = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.0],  # Overlaps query in X and Y.
            [3.0, 1.0, 0.0, 4.0, 2.0, 1.0],  # Only touches at query x=3.
            [1.0, 4.0, 0.0, 2.0, 5.0, 1.0],  # Separated along Y.
        ],
        dtype=np.float64,
    )
    query_aabbs = np.array([[1.0, 1.0, 10.0, 3.0, 3.0, 11.0]], dtype=np.float64)

    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    z_min_overlap_mask = relations.projected_overlap_mask(AABBFace.Z_MIN, minimum_face_crossing=0.0)
    z_max_overlap_mask = relations.projected_overlap_mask(AABBFace.Z_MAX, minimum_face_crossing=0.0)

    expected_overlap_mask = np.array([[True, False, False]])

    np.testing.assert_array_equal(z_min_overlap_mask, expected_overlap_mask)
    np.testing.assert_array_equal(z_max_overlap_mask, expected_overlap_mask)


def test_axis_overlap_compares_actual_length_with_inclusive_threshold() -> None:
    """Require positive actual overlap and accept equality with the threshold."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 2.0, 2.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [4.0, 0.0, 0.0, 5.0, 2.0, 2.0],  # Contained with X overlap 1.
            [8.0, 0.0, 0.0, 12.0, 2.0, 2.0],  # Partial X overlap 2.
            [5.0, 0.0, 0.0, 5.0, 2.0, 2.0],  # Contained with zero X length.
            [10.0, 0.0, 0.0, 11.0, 2.0, 2.0],  # Touches at query x=10.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    default_overlap = relations.axis_overlap(AABBAxis.X)
    threshold_overlap = relations.axis_overlap(AABBAxis.X, minimum_face_crossing=2.0)

    np.testing.assert_array_equal(default_overlap, np.array([[True, True, False, False]]))
    np.testing.assert_array_equal(threshold_overlap, np.array([[False, True, False, False]]))


def test_axis_overlap_all_axes_compares_each_actual_length() -> None:
    """Apply the inclusive overlap threshold independently on X, Y, and Z."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [4.0, 3.0, 3.0, 5.0, 7.0, 7.0],  # Overlap lengths 1, 4, 4.
            [8.0, 8.0, 8.0, 12.0, 12.0, 12.0],  # Overlap lengths 2, 2, 2.
            [5.0, 3.0, 3.0, 5.0, 7.0, 7.0],  # Overlap lengths 0, 4, 4.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    actual_overlap = relations.axis_overlap_all_axes(minimum_face_crossing=2.0)

    expected_overlap = np.array([[[False, True, True], [True, True, True], [False, True, True]]])
    np.testing.assert_array_equal(actual_overlap, expected_overlap)


def test_projected_overlap_mask_compares_both_actual_lengths() -> None:
    """Require positive actual overlap on both axes in the selected face."""
    query_aabbs = np.array([[0.0, 0.0, 10.0, 10.0, 10.0, 11.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [4.0, 3.0, 0.0, 5.0, 7.0, 1.0],  # Tangential lengths 1, 4.
            [8.0, 8.0, 0.0, 12.0, 12.0, 1.0],  # Tangential lengths 2, 2.
            [5.0, 3.0, 0.0, 5.0, 7.0, 1.0],  # Tangential lengths 0, 4.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    default_overlap = relations.projected_overlap_mask(AABBFace.Z_MIN)
    threshold_overlap = relations.projected_overlap_mask(AABBFace.Z_MIN, minimum_face_crossing=2.0)

    np.testing.assert_array_equal(default_overlap, np.array([[True, True, False]]))
    np.testing.assert_array_equal(threshold_overlap, np.array([[False, True, False]]))


def test_inward_projected_overlap_mask_uses_selected_face_axes() -> None:
    """Use the same tangential axes for both opposite faces."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [9.0, 2.0, 2.0, 10.0, 3.0, 3.0],  # Crosses only the YZ inward projection.
            [2.0, 9.0, 2.0, 3.0, 10.0, 3.0],  # Crosses only the XZ inward projection.
            [2.0, 2.0, 9.0, 3.0, 3.0, 10.0],  # Crosses only the XY inward projection.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)
    expected_by_face = {
        AABBFace.X_MIN: np.array([[True, False, False]]),
        AABBFace.X_MAX: np.array([[True, False, False]]),
        AABBFace.Y_MIN: np.array([[False, True, False]]),
        AABBFace.Y_MAX: np.array([[False, True, False]]),
        AABBFace.Z_MIN: np.array([[False, False, True]]),
        AABBFace.Z_MAX: np.array([[False, False, True]]),
    }

    for face, expected in expected_by_face.items():
        np.testing.assert_array_equal(relations.inward_projected_overlap_mask(face, inset=1.0), expected)


def test_inward_axis_overlap_all_axes_returns_xyz_masks() -> None:
    """Return strict inward-boundary crossings for X, Y, and Z together."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [0.0, 2.0, 2.0, 1.0, 3.0, 3.0],  # Equal to the inward X-min boundary.
            [2.0, 0.0, 2.0, 3.0, 1.0, 3.0],  # Equal to the inward Y-min boundary.
            [2.0, 2.0, 0.0, 3.0, 3.0, 1.0],  # Equal to the inward Z-min boundary.
            [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],  # Strictly crosses every inward interval.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    np.testing.assert_array_equal(
        relations.inward_axis_overlap_all_axes(inset=1.0),
        np.array([[[False, True, True], [True, False, True], [True, True, False], [True, True, True]]]),
    )


def test_inward_projected_overlap_mask_rejects_boundary_equality() -> None:
    """Require strict crossing of both inward query boundaries."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [0.0, 2.0, 0.0, 1.0, 3.0, 1.0],  # x_max equals query x_min + inset.
            [9.0, 2.0, 0.0, 10.0, 3.0, 1.0],  # x_min equals query x_max - inset.
            [0.0, 2.0, 0.0, 1.0001, 3.0, 1.0],  # Crosses the lower inward boundary.
            [8.9999, 2.0, 0.0, 10.0, 3.0, 1.0],  # Crosses the upper inward boundary.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    actual = relations.inward_projected_overlap_mask(AABBFace.Z_MIN, inset=1.0)

    np.testing.assert_array_equal(actual, np.array([[False, False, True, True]]))


def test_inward_projected_overlap_mask_accepts_narrow_contained_reference() -> None:
    """Test inward boundaries rather than a minimum overlap length."""
    query_aabbs = np.array([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float64)
    reference_aabbs = np.array([[4.0, 4.0, 20.0, 5.0, 5.0, 21.0]], dtype=np.float64)
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    inward_overlap = relations.inward_projected_overlap_mask(AABBFace.Z_MIN, inset=2.0)
    minimum_length_overlap = relations.projected_overlap_mask(AABBFace.Z_MIN, minimum_face_crossing=2.0)

    np.testing.assert_array_equal(inward_overlap, np.array([[True]]))
    np.testing.assert_array_equal(minimum_length_overlap, np.array([[False]]))


def test_inward_projected_overlap_mask_preserves_empty_shapes() -> None:
    """Return natural query-first shapes for empty batches."""
    one_aabb = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    empty_aabbs = np.empty((0, 6), dtype=np.float64)

    empty_query_relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=one_aabb, query_aabbs=empty_aabbs)
    empty_reference_relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=empty_aabbs, query_aabbs=one_aabb)

    np.testing.assert_array_equal(
        empty_query_relations.inward_projected_overlap_mask(AABBFace.X_MIN, inset=0.1), np.empty((0, 1), dtype=bool)
    )
    np.testing.assert_array_equal(
        empty_reference_relations.inward_projected_overlap_mask(AABBFace.X_MIN, inset=0.1), np.empty((1, 0), dtype=bool)
    )
    np.testing.assert_array_equal(
        empty_query_relations.inward_axis_overlap_all_axes(inset=0.1), np.empty((0, 1, 3), dtype=bool)
    )
    np.testing.assert_array_equal(
        empty_reference_relations.inward_axis_overlap_all_axes(inset=0.1), np.empty((1, 0, 3), dtype=bool)
    )


@pytest.mark.parametrize("inset", [-1.0, np.nan, np.inf])
def test_inward_projected_overlap_mask_rejects_invalid_inset(inset: float) -> None:
    """Reject negative and non-finite inward offsets."""
    one_aabb = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=one_aabb, query_aabbs=one_aabb)

    with pytest.raises(ValueError):
        relations.inward_projected_overlap_mask(AABBFace.X_MIN, inset=inset)
    with pytest.raises(ValueError):
        relations.inward_axis_overlap_all_axes(inset=inset)


def test_signed_distances_show_gap_alignment_and_crossing_for_min_face() -> None:
    """Show that signed distance is positive, zero, or negative across a face."""
    query_aabbs = np.array([[0.0, 0.0, 2.0, 2.0, 2.0, 4.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.0],  # Gap below the query.
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],  # Top aligns with query bottom.
            [0.0, 0.0, 0.0, 2.0, 2.0, 3.0],  # Crosses the query bottom.
        ],
        dtype=np.float64,
    )

    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    actual_distances = relations.signed_distances(AABBFace.Z_MIN)
    expected_distances = np.array([[1.0, 0.0, -1.0]])

    np.testing.assert_array_equal(actual_distances, expected_distances)


def test_within_distance_selects_references_near_the_chosen_x_face() -> None:
    """Show that opposite X faces select references on opposite sides."""
    query_aabbs = np.array([[2.0, 0.0, 0.0, 4.0, 2.0, 2.0]], dtype=np.float64)
    reference_aabbs = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],  # One unit from X_MIN.
            [5.0, 0.0, 0.0, 6.0, 2.0, 2.0],  # One unit from X_MAX.
        ],
        dtype=np.float64,
    )

    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    x_min_mask = relations.within_distance(AABBFace.X_MIN, minimum_distance=0.0, maximum_distance=1.0)
    x_max_mask = relations.within_distance(AABBFace.X_MAX, minimum_distance=0.0, maximum_distance=1.0)

    expected_x_min_mask = np.array([[True, False]])
    expected_x_max_mask = np.array([[False, True]])

    np.testing.assert_array_equal(x_min_mask, expected_x_min_mask)
    np.testing.assert_array_equal(x_max_mask, expected_x_max_mask)


def test_contained_by_mask_is_query_first_inclusive_and_supports_empty_batches() -> None:
    """Check inclusive containment orientation, self-comparisons, and empty shapes."""
    reference_aabbs = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],
            [2.0, 3.0, 4.0, 5.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0, 9.0, 10.0, 11.0],
        ],
        dtype=np.float64,
    )
    query_aabbs = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],  # Equal to references 0 and 2.
            [2.0, 3.0, 4.0, 5.0, 7.0, 8.0],  # Equal to reference 1 and inside 0 and 2.
            [-1.0, 3.0, 4.0, 5.0, 7.0, 8.0],  # Extends outside every reference.
        ],
        dtype=np.float64,
    )
    relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=reference_aabbs, query_aabbs=query_aabbs)

    expected_containment = np.array(
        [[True, False, True, False], [True, True, True, True], [False, False, False, False]]
    )
    np.testing.assert_array_equal(relations.contained_by_mask(), expected_containment)

    self_relations = PairwiseAABBRelations.from_aabbs(reference_aabbs=query_aabbs, query_aabbs=query_aabbs)
    np.testing.assert_array_equal(
        np.diag(self_relations.contained_by_mask()), np.ones(query_aabbs.shape[0], dtype=bool)
    )

    empty_query_relations = PairwiseAABBRelations.from_aabbs(
        reference_aabbs=reference_aabbs, query_aabbs=np.empty((0, 6), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        empty_query_relations.contained_by_mask(), np.empty((0, reference_aabbs.shape[0]), dtype=bool)
    )

    empty_reference_relations = PairwiseAABBRelations.from_aabbs(
        reference_aabbs=np.empty((0, 6), dtype=np.float64), query_aabbs=query_aabbs
    )
    np.testing.assert_array_equal(
        empty_reference_relations.contained_by_mask(), np.empty((query_aabbs.shape[0], 0), dtype=bool)
    )

    both_empty_relations = PairwiseAABBRelations.from_aabbs(
        reference_aabbs=np.empty((0, 6), dtype=np.float64), query_aabbs=np.empty((0, 6), dtype=np.float64)
    )
    np.testing.assert_array_equal(both_empty_relations.contained_by_mask(), np.empty((0, 0), dtype=bool))
