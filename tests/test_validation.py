"""Boundary and structural contracts beyond the geometric examples."""

import pytest
import torch

import aabb_primitives as aabb


def boxes(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return two valid boxes for validation and broadcasting cases."""
    return torch.tensor([[0, 0, 0, 2, 2, 2], [1, 1, 2, 3, 3, 3]], dtype=dtype)


@pytest.mark.parametrize("shape", [(2, 6), (3, 2, 6), (2, 1, 3, 4, 2, 6)])
def test_arbitrary_leading_dimensions_share_unbatched_references(shape: tuple[int, ...]) -> None:
    """Batch rank has no application meaning; one reference scene can be shared."""
    query = boxes().expand(shape)
    actual = aabb.contact_mask(query, boxes(), aabb.AABBFace.Z_MIN, distance_tolerance=0.0)
    expected = torch.tensor([[False, False], [True, False]])
    assert actual.shape == (*shape[:-2], 2, 2)
    assert torch.equal(actual, expected.expand(actual.shape))


def test_shared_reference_scenes_align_with_explicit_singleton_axis() -> None:
    """Each environment shares its own references across alternatives."""
    query = boxes().expand(2, 3, 2, 6).clone()
    reference = boxes().expand(2, 2, 6).clone()
    reference[1, :, 2] -= 10
    reference[1, :, 5] -= 10
    actual = aabb.contact_mask(query, reference[:, None], aabb.AABBFace.Z_MIN, distance_tolerance=0.0)
    assert actual.shape == (2, 3, 2, 2)
    assert actual[0, :, 1, 0].all()
    assert not actual[1].any()


@pytest.mark.parametrize("threshold", [torch.zeros(1, 2), torch.zeros(3, 2), torch.zeros(2, 1)])
def test_threshold_cannot_introduce_batch_or_broadcast_query_rows(threshold: torch.Tensor) -> None:
    """Unbatched boxes cannot acquire a batch axis through a threshold."""
    with pytest.raises((TypeError, ValueError)):
        aabb.contact_mask(boxes(), boxes(), aabb.AABBFace.Z_MIN, distance_tolerance=threshold)


def test_scalar_and_batch_aligned_thresholds_preserve_resolved_shape() -> None:
    """Thresholds align to query rows, never to reference rows."""
    query = boxes().expand(2, 1, 2, 6)
    reference = boxes().expand(1, 3, 2, 6)
    for tolerance in (0.0, torch.tensor(0.0), torch.zeros(2), torch.zeros(2, 1, 2), torch.zeros(2, 3, 2)):
        result = aabb.contact_mask(query, reference, aabb.AABBFace.Z_MIN, distance_tolerance=tolerance)
        assert result.shape == (2, 3, 2, 2)
        assert result[..., 1, 0].all()
    with pytest.raises((TypeError, ValueError)):
        aabb.contact_mask(query, reference, aabb.AABBFace.Z_MIN, distance_tolerance=torch.zeros(4, 2))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_unsupported_box_dtypes_are_rejected(dtype: torch.dtype) -> None:
    """The supported precision contract is explicit, not implicit casting."""
    with pytest.raises((TypeError, ValueError)):
        aabb.overlap_lengths_all_axes(boxes().to(dtype), boxes().to(dtype))
    with pytest.raises(TypeError):
        aabb.validate_aabbs(boxes().to(dtype))


def test_box_structure_dtype_device_and_enum_mismatches_are_rejected() -> None:
    """Fail at incompatible input boundaries without attempting transfers."""
    invalid_pairs = [
        (boxes(), boxes(torch.float32)),
        (boxes(), torch.empty(2, 6, device="meta", dtype=torch.float64)),
        (boxes().expand(2, 2, 6), boxes().expand(3, 2, 6)),
        (torch.zeros(6, dtype=torch.float64), boxes()),
        (torch.zeros(2, 5, dtype=torch.float64), boxes()),
    ]
    for query, reference in invalid_pairs:
        with pytest.raises((TypeError, ValueError)):
            aabb.overlap_lengths_all_axes(query, reference)
    with pytest.raises(TypeError):
        aabb.signed_distances(boxes(), boxes(), "Z_MIN")
    with pytest.raises(TypeError):
        aabb.axis_overlap(boxes(), boxes(), 0)
    with pytest.raises(ValueError, match="same device"):
        aabb.contact_mask(
            boxes(),
            boxes(),
            aabb.AABBFace.Z_MIN,
            distance_tolerance=torch.empty((), device="meta", dtype=torch.float64),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_aabb_validation_rejects_nonfinite_values(value: float) -> None:
    """Coordinate scans are available as an explicit boundary check."""
    invalid = boxes()
    invalid[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        aabb.validate_aabbs(invalid)


def test_aabb_validation_allows_empty_and_degenerate_without_mutation() -> None:
    """Zero extents are useful for floor and patch representations."""
    for value in (torch.zeros(2, 6), torch.empty(0, 6), torch.empty(2, 0, 3, 6)):
        before = value.clone()
        assert aabb.validate_aabbs(value) is None
        torch.testing.assert_close(value, before)
    invalid = boxes()
    invalid[0, 0] = 10
    with pytest.raises(ValueError, match="extents"):
        aabb.validate_aabbs(invalid)


@pytest.mark.parametrize("name", ["distance_tolerance", "minimum_face_crossing", "inset"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), torch.tensor([-0.1, 0.0])])
def test_nonnegative_threshold_domains(name: str, value: float | torch.Tensor) -> None:
    """Every supplied tolerance is checked explicitly for its numeric domain."""
    with pytest.raises(ValueError):
        aabb.validate_thresholds(**{name: value})


def test_threshold_validation_accepts_signed_ranges_and_rejects_inversions() -> None:
    """Distance intervals are signed and inclusive, unlike tolerances."""
    assert aabb.validate_thresholds() is None
    aabb.validate_thresholds(minimum_distance=-2.0, maximum_distance=-1.0)
    aabb.validate_thresholds(minimum_distance=torch.tensor([-2.0, 0.0]), maximum_distance=0.0)
    aabb.validate_thresholds(minimum_distance=0.0, maximum_distance=torch.zeros(2))
    aabb.validate_thresholds(minimum_patch_lengths=(0.0, 1.0), inset=torch.empty(0))
    with pytest.raises(ValueError, match="less than or equal"):
        aabb.validate_thresholds(minimum_distance=torch.tensor([-2.0, 1.0]), maximum_distance=0.0)
    with pytest.raises(ValueError, match="broadcastable"):
        aabb.validate_thresholds(minimum_distance=torch.zeros(2), maximum_distance=torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        aabb.validate_thresholds(minimum_distance=float("nan"))
    for value in ((-1.0, 0.0), (0.0, float("inf"))):
        with pytest.raises(ValueError):
            aabb.validate_thresholds(minimum_patch_lengths=value)
    for value in (1.0, (0.0,), ((0.0, 0.0), (0.0, 0.0)), torch.zeros(2), torch.zeros(1, 3)):
        with pytest.raises((TypeError, ValueError)):
            aabb.validate_thresholds(minimum_patch_lengths=value)


def test_invalid_numeric_inputs_are_not_silently_repaired() -> None:
    """Unchecked operations have preconditions, while explicit checks reject violations."""
    invalid = boxes()
    invalid[0, 0] = float("nan")
    result = aabb.signed_distances(invalid, boxes(), aabb.AABBFace.X_MIN)
    assert torch.isnan(result[0]).all()
    with pytest.raises(ValueError):
        aabb.validate_aabbs(invalid)
