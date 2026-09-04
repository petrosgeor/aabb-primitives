import pytest
import torch

import aabb_primitives as aabb
from aabb_primitives import AABBAxis, AABBFace


def test_signed_distances_preserve_face_order_and_orientation() -> None:
    """Measure gaps, alignment, and crossing for all six oriented faces."""
    query_aabbs = torch.tensor([[0.0, 0.0, 2.0, 2.0, 2.0, 4.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 2.0, 1.0], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 2.0, 2.0, 3.0]],
        dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [-2.0, -2.0, -2.0, -2.0, 1.0, -4.0],
            [-2.0, -2.0, -2.0, -2.0, 0.0, -4.0],
            [-2.0, -2.0, -2.0, -2.0, -1.0, -4.0],
        ],
        dtype=torch.float64,
    )
    all_faces = aabb.signed_distances_all_faces(query_aabbs, reference_aabbs)
    torch.testing.assert_close(all_faces[0], expected)

    for face_index, face in enumerate(AABBFace):
        torch.testing.assert_close(
            aabb.signed_distances(query_aabbs, reference_aabbs, face), expected[:, face_index].unsqueeze(0)
        )


def test_intersections_overlaps_and_containment_preserve_raw_bounds() -> None:
    """Keep inverted intersection intervals while clamping overlap lengths."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[4.0, 3.0, 3.0, 5.0, 7.0, 7.0], [8.0, 8.0, 8.0, 12.0, 12.0, 12.0], [5.0, 3.0, 3.0, 5.0, 7.0, 7.0]],
        dtype=torch.float64,
    )

    expected_bounds = torch.tensor(
        [
            [
                [[4.0, 5.0], [3.0, 7.0], [3.0, 7.0]],
                [[8.0, 10.0], [8.0, 10.0], [8.0, 10.0]],
                [[5.0, 5.0], [3.0, 7.0], [3.0, 7.0]],
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(aabb.intersection_bounds_all_axes(query_aabbs, reference_aabbs), expected_bounds)
    torch.testing.assert_close(
        aabb.intersection_bounds(query_aabbs, reference_aabbs, AABBAxis.X), expected_bounds[..., 0, :]
    )
    torch.testing.assert_close(
        aabb.overlap_lengths_all_axes(query_aabbs, reference_aabbs),
        torch.tensor([[[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [0.0, 4.0, 4.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        aabb.overlap_lengths(query_aabbs, reference_aabbs, AABBAxis.X),
        torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        aabb.tangential_overlap_lengths(query_aabbs, reference_aabbs, AABBFace.Z_MIN),
        torch.tensor([[[1.0, 4.0], [2.0, 2.0], [0.0, 4.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        aabb.projected_intersection_bounds(query_aabbs, reference_aabbs, AABBFace.Z_MIN),
        torch.tensor([[[4.0, 3.0, 5.0, 7.0], [8.0, 8.0, 10.0, 10.0], [5.0, 3.0, 5.0, 7.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        aabb.projected_overlap_areas(query_aabbs, reference_aabbs, AABBFace.Z_MIN),
        torch.tensor([[4.0, 4.0, 0.0]], dtype=torch.float64),
    )
    assert torch.equal(aabb.axis_overlap(query_aabbs, reference_aabbs, AABBAxis.X), torch.tensor([[True, True, False]]))
    assert torch.equal(
        aabb.axis_overlap_all_axes(query_aabbs, reference_aabbs),
        torch.tensor([[[True, True, True], [True, True, True], [False, True, True]]]),
    )
    assert torch.equal(aabb.contained_by_mask(query_aabbs, reference_aabbs), torch.tensor([[False, False, False]]))
    torch.testing.assert_close(
        aabb.query_face_areas(query_aabbs, AABBFace.Z_MIN), torch.tensor([100.0], dtype=torch.float64)
    )


def test_projected_inward_and_contact_masks_keep_strict_and_inclusive_rules() -> None:
    """Separate positive overlap, inward crossing, distance, and patch rules."""
    query_aabbs = torch.tensor([[0.0, 0.0, 10.0, 10.0, 10.0, 11.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[4.0, 3.0, 0.0, 5.0, 7.0, 1.0], [8.0, 8.0, 0.0, 12.0, 12.0, 1.0], [5.0, 3.0, 0.0, 5.0, 7.0, 1.0]],
        dtype=torch.float64,
    )
    assert torch.equal(
        aabb.projected_overlap_mask(query_aabbs, reference_aabbs, AABBFace.Z_MIN), torch.tensor([[True, True, False]])
    )
    assert torch.equal(
        aabb.projected_overlap_mask(query_aabbs, reference_aabbs, AABBFace.Z_MIN, minimum_face_crossing=2.0),
        torch.tensor([[False, True, False]]),
    )

    inward_query = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=torch.float64)
    inward_reference = torch.tensor(
        [
            [0.0, 2.0, 2.0, 1.0, 3.0, 3.0],
            [2.0, 0.0, 2.0, 3.0, 1.0, 3.0],
            [2.0, 2.0, 0.0, 3.0, 3.0, 1.0],
            [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        aabb.inward_axis_overlap_all_axes(inward_query, inward_reference, inset=1.0),
        torch.tensor([[[False, True, True], [True, False, True], [True, True, False], [True, True, True]]]),
    )
    assert torch.equal(
        aabb.inward_projected_overlap_mask(inward_query, inward_reference, AABBFace.Z_MIN, inset=1.0),
        torch.tensor([[False, False, True, True]]),
    )

    contact_query = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
    contact_reference = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.1, 2.0, 2.0, 2.0], [0.0, 1.5, 1.0, 2.0, 2.5, 2.0]],
        dtype=torch.float64,
    )
    assert torch.equal(
        aabb.within_distance(
            contact_query, contact_reference, AABBFace.Z_MAX, minimum_distance=0.0, maximum_distance=0.1
        ),
        torch.tensor([[True, False, True]]),
    )
    assert torch.equal(
        aabb.contact_mask(
            contact_query, contact_reference, AABBFace.Z_MAX, distance_tolerance=0.1, minimum_patch_lengths=(2.0, 1.0)
        ),
        torch.tensor([[True, False, False]]),
    )


def test_query_face_contact_patches_cover_all_oriented_faces() -> None:
    """Represent exact contact on each oriented query face as a degenerate AABB."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [
            [-1.0, 0.5, 1.0, 0.0, 2.5, 3.0],
            [2.0, 0.5, 1.0, 3.0, 2.5, 3.0],
            [0.5, -1.0, 1.0, 1.5, 0.0, 3.0],
            [0.5, 3.0, 1.0, 1.5, 4.0, 3.0],
            [0.5, 0.5, -1.0, 1.5, 2.5, 0.0],
            [0.5, 0.5, 4.0, 1.5, 2.5, 5.0],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [0.0, 0.5, 1.0, 0.0, 2.5, 3.0],
            [2.0, 0.5, 1.0, 2.0, 2.5, 3.0],
            [0.5, 0.0, 1.0, 1.5, 0.0, 3.0],
            [0.5, 3.0, 1.0, 1.5, 3.0, 3.0],
            [0.5, 0.5, 0.0, 1.5, 2.5, 0.0],
            [0.5, 0.5, 4.0, 1.5, 2.5, 4.0],
        ],
        dtype=torch.float64,
    )

    patches, face_contact_mask = aabb.query_face_contact_patches(query_aabbs, reference_aabbs, distance_tolerance=0.0)
    assert patches.shape == (1, 6, 6, 6)
    assert torch.equal(face_contact_mask, torch.eye(6, dtype=torch.bool).unsqueeze(0))
    for face_index, expected_patch in enumerate(expected):
        expected_faces = torch.zeros((6, 6), dtype=torch.float64)
        expected_faces[face_index] = expected_patch
        torch.testing.assert_close(patches[0, face_index], expected_faces)


@pytest.mark.parametrize("reference_maximum, expected_valid", [(-0.2, True), (0.0, True), (0.2, True), (-0.21, False)])
def test_query_face_contact_patches_accept_symmetric_normal_tolerance(
    reference_maximum: float, expected_valid: bool
) -> None:
    """Accept equal, gapped, and crossed faces within absolute distance tolerance."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[0.0, 0.0, reference_maximum - 1.0, 2.0, 2.0, reference_maximum]], dtype=torch.float64
    )

    patches, face_contact_mask = aabb.query_face_contact_patches(query_aabbs, reference_aabbs, distance_tolerance=0.2)
    expected = torch.zeros((1, 1, 6, 6), dtype=torch.float64)
    expected_mask = torch.zeros((1, 1, 6), dtype=torch.bool)
    if expected_valid:
        expected[0, 0, 4] = torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 0.0], dtype=torch.float64)
        expected_mask[0, 0, 4] = True
    torch.testing.assert_close(patches, expected)
    assert torch.equal(face_contact_mask, expected_mask)


def test_query_face_contact_patches_require_normal_and_tangential_thresholds() -> None:
    """Reject distance misses, edge contacts, and either insufficient tangential overlap."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [
            [0.0, 0.0, -1.21, 2.0, 2.0, -0.21],
            [2.0, 0.0, -1.0, 3.0, 2.0, 0.0],
            [0.0, 2.0, -1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, -1.0, 0.5, 2.0, 0.0],
            [0.0, 0.0, -1.0, 2.0, 0.5, 0.0],
            [2.0, 2.0, -1.0, 3.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )
    patches, face_contact_mask = aabb.query_face_contact_patches(
        query_aabbs, reference_aabbs, distance_tolerance=0.2, minimum_face_crossing=1.0
    )
    torch.testing.assert_close(patches, torch.zeros((1, 6, 6, 6), dtype=torch.float64))
    assert not face_contact_mask.any()

    threshold_reference = torch.tensor([[0.0, 0.0, -1.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    threshold_patch, threshold_mask = aabb.query_face_contact_patches(
        query_aabbs, threshold_reference, distance_tolerance=0.0, minimum_face_crossing=1.0
    )
    expected_threshold_patch = torch.zeros((1, 1, 6, 6), dtype=torch.float64)
    expected_threshold_patch[0, 0, 4] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(threshold_patch, expected_threshold_patch)
    assert torch.equal(threshold_mask, torch.tensor([[[False, False, False, False, True, False]]]))


def test_query_face_contact_patches_preserve_every_qualifying_face() -> None:
    """Keep simultaneous contacts instead of selecting one face per pair."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 10.0, 10.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[0.4, 0.0, 0.0, 1.5, 10.0, 10.0], [0.5, 0.0, 0.0, 1.5, 10.0, 10.0]], dtype=torch.float64
    )
    patches, face_contact_mask = aabb.query_face_contact_patches(query_aabbs, reference_aabbs, distance_tolerance=2.0)
    expected = torch.zeros((1, 2, 6, 6), dtype=torch.float64)
    expected[:, :, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 10.0], dtype=torch.float64)
    expected[:, :, 1] = torch.tensor([2.0, 0.0, 0.0, 2.0, 10.0, 10.0], dtype=torch.float64)
    torch.testing.assert_close(patches, expected)
    assert torch.equal(face_contact_mask, torch.tensor([[[True, True, False, False, False, False]] * 2]))


def test_query_face_contact_patches_keep_degenerate_queries_and_self_pairs() -> None:
    """Keep query-face coordinates for degenerate queries without self exclusion."""
    degenerate_query = torch.tensor([[0.0, 0.0, 1.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
    patch, face_contact_mask = aabb.query_face_contact_patches(
        degenerate_query, reference_aabbs, distance_tolerance=0.0
    )
    expected_patch = torch.zeros((1, 1, 6, 6), dtype=torch.float64)
    expected_patch[0, 0, 4] = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(patch, expected_patch)
    assert torch.equal(face_contact_mask, torch.tensor([[[False, False, False, False, True, False]]]))

    self_aabb = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    self_patch, self_mask = aabb.query_face_contact_patches(self_aabb, self_aabb, distance_tolerance=2.0)
    torch.testing.assert_close(
        self_patch,
        torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                        [2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                        [0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
                        [0.0, 2.0, 0.0, 2.0, 2.0, 2.0],
                        [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                    ]
                ]
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(self_mask, torch.ones((1, 1, 6), dtype=torch.bool))


def test_query_face_contact_patches_broadcast_thresholds_and_empty_dimensions() -> None:
    """Broadcast scalar and per-query thresholds while retaining natural empty shapes."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    reference_aabbs = torch.tensor(
        [[0.0, 0.0, -1.1, 2.0, 2.0, -0.1], [0.0, 0.0, -1.3, 2.0, 2.0, -0.3]], dtype=torch.float64
    )
    per_query_patches, per_query_mask = aabb.query_face_contact_patches(
        query_aabbs,
        reference_aabbs,
        distance_tolerance=torch.tensor([0.1, 0.2], dtype=torch.float64),
        minimum_face_crossing=torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    expected_per_query = torch.zeros((2, 2, 6, 6), dtype=torch.float64)
    expected_per_query[0, 0, 4] = torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(per_query_patches, expected_per_query)
    expected_per_query_mask = torch.zeros((2, 2, 6), dtype=torch.bool)
    expected_per_query_mask[0, 0, 4] = True
    assert torch.equal(per_query_mask, expected_per_query_mask)

    scalar_patches, scalar_mask = aabb.query_face_contact_patches(query_aabbs, reference_aabbs, distance_tolerance=0.1)
    expected_scalar = torch.zeros((2, 2, 6, 6), dtype=torch.float64)
    expected_scalar[:, 0, 4] = torch.tensor([0.0, 0.0, 0.0, 2.0, 2.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(scalar_patches, expected_scalar)
    expected_scalar_mask = torch.zeros((2, 2, 6), dtype=torch.bool)
    expected_scalar_mask[:, 0, 4] = True
    assert torch.equal(scalar_mask, expected_scalar_mask)

    empty_query = torch.empty((0, 6), dtype=torch.float64)
    empty_reference = torch.empty((0, 6), dtype=torch.float64)
    empty_query_patches, empty_query_mask = aabb.query_face_contact_patches(
        empty_query, reference_aabbs[:1], distance_tolerance=0.0
    )
    assert empty_query_patches.shape == (0, 1, 6, 6)
    assert empty_query_mask.shape == (0, 1, 6)
    empty_reference_patches, empty_reference_mask = aabb.query_face_contact_patches(
        query_aabbs[:1], empty_reference, distance_tolerance=0.0
    )
    assert empty_reference_patches.shape == (1, 0, 6, 6)
    assert empty_reference_mask.shape == (1, 0, 6)


def test_contact_tuple_patch_threshold_keeps_float64_boundary_precision() -> None:
    """Do not round Python patch thresholds to float32 at the comparison boundary."""
    query_aabbs = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    reference_aabbs = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.10000000149011612, 2.0]], dtype=torch.float32)

    assert not aabb.contact_mask(
        query_aabbs, reference_aabbs, AABBFace.Z_MAX, distance_tolerance=0.0, minimum_patch_lengths=(1.0, 0.100000002)
    ).item()


def test_broadcasted_batches_match_independent_world_calls() -> None:
    """Broadcast independent batch axes without materializing input copies."""
    query_aabbs = torch.tensor(
        [
            [[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]]],
            [[[5.0, 0.0, 0.0, 7.0, 2.0, 2.0], [6.0, 1.0, 1.0, 8.0, 3.0, 3.0]]],
        ],
        dtype=torch.float64,
    )
    reference_aabbs = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]],
                [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 6.0, 6.0, 6.0]],
                [[1.0, 1.0, 1.0, 3.0, 3.0, 3.0], [7.0, 7.0, 7.0, 8.0, 8.0, 8.0]],
            ]
        ],
        dtype=torch.float64,
    )
    assert query_aabbs.shape == (2, 1, 2, 6)
    assert reference_aabbs.shape == (1, 3, 2, 6)
    output = aabb.overlap_lengths_all_axes(query_aabbs, reference_aabbs)
    assert output.shape == (2, 3, 2, 2, 3)
    for environment_index in range(2):
        for world_index in range(3):
            expected = aabb.overlap_lengths_all_axes(query_aabbs[environment_index, 0], reference_aabbs[0, world_index])
            torch.testing.assert_close(output[environment_index, world_index], expected)

    distances = aabb.signed_distances(query_aabbs, reference_aabbs, AABBFace.Z_MIN)
    assert distances.shape == (2, 3, 2, 2)
    distance_tolerances = torch.tensor([[[0.0, 0.5]], [[1.0, 1.5]]], dtype=torch.float64)
    minimum_crossings = torch.tensor([[[0.0, 0.5]], [[1.0, 1.5]]], dtype=torch.float64)
    patches, face_contact_mask = aabb.query_face_contact_patches(
        query_aabbs, reference_aabbs, distance_tolerance=distance_tolerances, minimum_face_crossing=minimum_crossings
    )
    assert patches.shape == (2, 3, 2, 2, 6, 6)
    assert face_contact_mask.shape == (2, 3, 2, 2, 6)
    for environment_index in range(2):
        for world_index in range(3):
            expected_patches, expected_mask = aabb.query_face_contact_patches(
                query_aabbs[environment_index, 0],
                reference_aabbs[0, world_index],
                distance_tolerance=distance_tolerances[environment_index, 0],
                minimum_face_crossing=minimum_crossings[environment_index, 0],
            )
            torch.testing.assert_close(patches[environment_index, world_index], expected_patches)
            assert torch.equal(face_contact_mask[environment_index, world_index], expected_mask)


def test_thresholds_accept_python_scalar_scalar_tensor_and_per_query_tensor() -> None:
    """Broadcast scalar and query-aligned thresholds across reference rows."""
    query_aabbs = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 2.0, 1.0, 1.0], [1.2, 0.0, 0.0, 2.2, 1.0, 1.0]]],
            [[[1.2, 0.0, 0.0, 2.2, 1.0, 1.0], [1.4, 0.0, 0.0, 2.4, 1.0, 1.0]]],
        ],
        dtype=torch.float64,
    )
    reference_aabbs = torch.tensor(
        [[[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]], dtype=torch.float64
    )
    maximum_distances = torch.tensor([[[0.0, 0.1]], [[0.25, 0.5]]], dtype=torch.float64)
    expected = torch.tensor([[[[True], [False]], [[True], [False]]], [[[True], [True]], [[True], [True]]]])

    for maximum_distance in (0.5, torch.tensor(0.5, dtype=torch.float64)):
        result = aabb.within_distance(
            query_aabbs, reference_aabbs, AABBFace.X_MIN, minimum_distance=0.0, maximum_distance=maximum_distance
        )
        assert result.shape == (2, 2, 2, 1)
    assert torch.equal(
        aabb.within_distance(
            query_aabbs,
            reference_aabbs,
            AABBFace.X_MIN,
            minimum_distance=torch.zeros(2, 1, 2, dtype=torch.float64),
            maximum_distance=maximum_distances,
        ),
        expected,
    )
    assert torch.equal(
        aabb.contact_mask(query_aabbs, reference_aabbs, AABBFace.X_MIN, distance_tolerance=maximum_distances), expected
    )
    assert torch.equal(
        aabb.projected_overlap_mask(
            query_aabbs,
            reference_aabbs,
            AABBFace.X_MIN,
            minimum_face_crossing=torch.tensor([[[0.5, 1.1]], [[0.5, 1.1]]], dtype=torch.float64),
        ),
        torch.tensor([[[[True], [False]], [[True], [False]]], [[[True], [False]], [[True], [False]]]]),
    )
    assert aabb.axis_overlap_all_axes(
        query_aabbs, reference_aabbs, minimum_face_crossing=torch.zeros(2, 1, 2, dtype=torch.float64)
    ).shape == torch.Size([2, 2, 2, 1, 3])
    patch_thresholds = torch.tensor([[[[0.9, 0.9], [0.1, 0.1]]], [[[0.9, 0.9], [0.1, 0.1]]]], dtype=torch.float64)
    patch_result = aabb.contact_mask(
        query_aabbs,
        reference_aabbs,
        AABBFace.X_MIN,
        distance_tolerance=maximum_distances,
        minimum_patch_lengths=patch_thresholds,
    )
    assert patch_result.shape == (2, 2, 2, 1)


def test_empty_degenerate_and_self_relations_keep_natural_shapes() -> None:
    """Support empty collections, zero-length axes, and diagonal self-relations."""
    one_aabb = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    empty_aabbs = torch.empty((0, 6), dtype=torch.float64)
    empty_query = aabb.contact_mask(empty_aabbs, one_aabb, AABBFace.X_MIN, distance_tolerance=0.1)
    empty_reference = aabb.contact_mask(one_aabb, empty_aabbs, AABBFace.X_MIN, distance_tolerance=0.1)
    assert empty_query.shape == (0, 1)
    assert empty_reference.shape == (1, 0)
    assert aabb.intersection_bounds_all_axes(empty_aabbs, empty_aabbs).shape == (0, 0, 3, 2)

    degenerate = torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    assert torch.equal(aabb.axis_overlap_all_axes(degenerate, degenerate), torch.tensor([[[False, True, True]]]))
    assert torch.equal(torch.diag(aabb.contained_by_mask(degenerate, degenerate)), torch.ones(1, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_float_dtypes_noncontiguous_inputs_and_input_preservation(dtype: torch.dtype) -> None:
    """Keep dtype, views, and source storage behavior ordinary for Torch."""
    source = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]],
        dtype=dtype,
        device="cpu",
    )
    query_aabbs = source[::2]
    reference_aabbs = source[1:]
    assert not query_aabbs.is_contiguous()
    query_before = query_aabbs.clone()
    reference_before = reference_aabbs.clone()
    result = aabb.overlap_lengths_all_axes(query_aabbs, reference_aabbs)
    assert result.dtype == dtype
    patches, face_contact_mask = aabb.query_face_contact_patches(query_aabbs, reference_aabbs, distance_tolerance=1.0)
    assert patches.dtype == dtype
    assert patches.device == source.device
    assert face_contact_mask.dtype == torch.bool
    assert face_contact_mask.device == source.device
    torch.testing.assert_close(query_aabbs, query_before)
    torch.testing.assert_close(reference_aabbs, reference_before)
