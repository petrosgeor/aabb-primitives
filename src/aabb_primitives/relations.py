"""Pairwise geometry for broadcastable Torch AABB collections.

Rows use [xmin, ymin, zmin, xmax, ymax, zmax]. Leading dimensions broadcast;
query and reference rows form the final pairwise axes. Operations check
structure only: use validate_aabbs and validate_thresholds to check numerical
preconditions at an input boundary. No inputs are stored, cloned, or mutated.

Batch symbols are independent in annotations. Torch metadata checks establish
broadcast compatibility; Jaxtyping's # broadcast annotations would require
NumPy at runtime, which is deliberately an optional plotting dependency.
"""

from dataclasses import dataclass
from enum import Enum, auto

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from aabb_primitives._validation import _check_aabbs, _check_pair, _check_threshold

__all__ = [
    "AABBAxis",
    "AABBFace",
    "signed_distances",
    "signed_distances_all_faces",
    "axis_overlap",
    "axis_overlap_all_axes",
    "contained_by_mask",
    "intersection_bounds",
    "intersection_bounds_all_axes",
    "overlap_lengths",
    "overlap_lengths_all_axes",
    "query_face_contact_patches",
    "tangential_overlap_lengths",
    "projected_overlap_mask",
    "inward_projected_overlap_mask",
    "inward_axis_overlap_all_axes",
    "projected_overlap_areas",
    "query_face_areas",
    "projected_intersection_bounds",
    "within_distance",
    "contact_mask",
]


class AABBAxis(Enum):
    """Name one coordinate axis without using an array index."""

    X = auto()
    Y = auto()
    Z = auto()


class AABBFace(Enum):
    """Name one face of a query AABB."""

    X_MIN = auto()
    X_MAX = auto()
    Y_MIN = auto()
    Y_MAX = auto()
    Z_MIN = auto()
    Z_MAX = auto()


@dataclass(frozen=True, slots=True)
class _FaceAxes:
    """Store the axes and direction used by a query face.

    Attributes:
        normal_axis (AABBAxis): Axis that points out of the face.
        tangential_axes (tuple[AABBAxis, AABBAxis]): Two axes that lie in the
            face, in their output order.
        is_min_face (bool): Whether this is the minimum face on its axis.
    """

    normal_axis: AABBAxis
    tangential_axes: tuple[AABBAxis, AABBAxis]
    is_min_face: bool


_AXIS_INDICES: dict[AABBAxis, int] = {AABBAxis.X: 0, AABBAxis.Y: 1, AABBAxis.Z: 2}
_FACE_AXES: dict[AABBFace, _FaceAxes] = {
    AABBFace.X_MIN: _FaceAxes(AABBAxis.X, (AABBAxis.Y, AABBAxis.Z), True),
    AABBFace.X_MAX: _FaceAxes(AABBAxis.X, (AABBAxis.Y, AABBAxis.Z), False),
    AABBFace.Y_MIN: _FaceAxes(AABBAxis.Y, (AABBAxis.X, AABBAxis.Z), True),
    AABBFace.Y_MAX: _FaceAxes(AABBAxis.Y, (AABBAxis.X, AABBAxis.Z), False),
    AABBFace.Z_MIN: _FaceAxes(AABBAxis.Z, (AABBAxis.X, AABBAxis.Y), True),
    AABBFace.Z_MAX: _FaceAxes(AABBAxis.Z, (AABBAxis.X, AABBAxis.Y), False),
}


def _add_reference_axis(threshold: float | torch.Tensor, *, add_feature_axis: bool = False) -> float | torch.Tensor:
    """Add pairwise singleton axes to a non-scalar query threshold.

    Args:
        threshold (float | torch.Tensor): Scalar or per-query threshold.
        add_feature_axis (bool): Whether to add an additional singleton axis
            for an output feature dimension.

    Returns:
        (float | torch.Tensor): A threshold ready for pairwise broadcasting.
    """
    if isinstance(threshold, torch.Tensor) and threshold.ndim:
        threshold = threshold[..., None]
        if add_feature_axis:
            threshold = threshold[..., None]
    return threshold


@jaxtyped(typechecker=beartype)
def signed_distances(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
) -> Float[torch.Tensor, "*batch query reference"]:
    """Return signed distances from one query face to its opposite.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face to measure.

    Returns:
        (Float[torch.Tensor, "*batch query reference"]): Signed distances.
            Positive values are gaps, zero aligns faces, and negative
            values mean that selected faces have crossed.
    """
    _check_pair(query_aabbs, reference_aabbs)
    face_axes = _FACE_AXES[face]
    axis_index = _AXIS_INDICES[face_axes.normal_axis]
    if face_axes.is_min_face:
        return query_aabbs[..., :, None, axis_index] - reference_aabbs[..., None, :, axis_index + 3]
    return reference_aabbs[..., None, :, axis_index] - query_aabbs[..., :, None, axis_index + 3]


@jaxtyped(typechecker=beartype)
def signed_distances_all_faces(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
) -> Float[torch.Tensor, "*batch query reference 6"]:
    """Return signed distances for all six query faces.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.

    Returns:
        (Float[torch.Tensor, "*batch query reference 6"]): Distances in
            X-minimum, X-maximum, Y-minimum, Y-maximum, Z-minimum,
            Z-maximum order.
    """
    _check_pair(query_aabbs, reference_aabbs)
    query_minimum = query_aabbs[..., :, None, :3]
    query_maximum = query_aabbs[..., :, None, 3:]
    reference_minimum = reference_aabbs[..., None, :, :3]
    reference_maximum = reference_aabbs[..., None, :, 3:]
    query_minimum_minus_reference_maximum = query_minimum - reference_maximum
    reference_minimum_minus_query_maximum = reference_minimum - query_maximum
    return torch.stack(
        (
            query_minimum_minus_reference_maximum[..., 0],
            reference_minimum_minus_query_maximum[..., 0],
            query_minimum_minus_reference_maximum[..., 1],
            reference_minimum_minus_query_maximum[..., 1],
            query_minimum_minus_reference_maximum[..., 2],
            reference_minimum_minus_query_maximum[..., 2],
        ),
        dim=-1,
    )


@jaxtyped(typechecker=beartype)
def axis_overlap(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    axis: AABBAxis,
    *,
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check whether each pair overlaps strictly along one axis.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        axis (AABBAxis): Axis to check.
        minimum_face_crossing (float | Float[torch.Tensor, ""] |
            Float[torch.Tensor, "*threshold_batch query"]): Minimum actual overlap
            length. Equality is accepted.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Strict overlap mask.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    result_overlap_lengths = overlap_lengths(query_aabbs, reference_aabbs, axis)
    crossing_threshold = _add_reference_axis(minimum_face_crossing)
    return (result_overlap_lengths > 0.0) & (result_overlap_lengths >= crossing_threshold)


@jaxtyped(typechecker=beartype)
def axis_overlap_all_axes(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    *,
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
) -> Bool[torch.Tensor, "*batch query reference 3"]:
    """Check strict overlap along X, Y, and Z.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        minimum_face_crossing (float | Float[torch.Tensor, ""] |
            Float[torch.Tensor, "*threshold_batch query"]): Minimum actual overlap
            length on each axis. Equality is accepted.

    Returns:
        (Bool[torch.Tensor, "*batch query reference 3"]): X, Y, Z masks.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    result_overlap_lengths = overlap_lengths_all_axes(query_aabbs, reference_aabbs)
    crossing_threshold = _add_reference_axis(minimum_face_crossing, add_feature_axis=True)
    return (result_overlap_lengths > 0.0) & (result_overlap_lengths >= crossing_threshold)


@jaxtyped(typechecker=beartype)
def contained_by_mask(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check whether each query AABB is contained by each reference.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Inclusive
            containment mask.
    """
    _check_pair(query_aabbs, reference_aabbs)
    query_minimum = query_aabbs[..., :, None, :3]
    query_maximum = query_aabbs[..., :, None, 3:]
    reference_minimum = reference_aabbs[..., None, :, :3]
    reference_maximum = reference_aabbs[..., None, :, 3:]
    return torch.all((query_minimum >= reference_minimum) & (query_maximum <= reference_maximum), dim=-1)


@jaxtyped(typechecker=beartype)
def intersection_bounds(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    axis: AABBAxis,
) -> Float[torch.Tensor, "*batch query reference 2"]:
    """Return raw intersection bounds along one axis.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        axis (AABBAxis): Axis to measure.

    Returns:
        (Float[torch.Tensor, "*batch query reference 2"]): Minimum and
            maximum intersection coordinates. Separate intervals retain
            inverted bounds.
    """
    _check_pair(query_aabbs, reference_aabbs)
    axis_index = _AXIS_INDICES[axis]
    intersection_minimum = torch.maximum(
        query_aabbs[..., :, None, axis_index], reference_aabbs[..., None, :, axis_index]
    )
    intersection_maximum = torch.minimum(
        query_aabbs[..., :, None, axis_index + 3], reference_aabbs[..., None, :, axis_index + 3]
    )
    return torch.stack((intersection_minimum, intersection_maximum), dim=-1)


@jaxtyped(typechecker=beartype)
def intersection_bounds_all_axes(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
) -> Float[torch.Tensor, "*batch query reference 3 2"]:
    """Return raw intersection bounds along X, Y, and Z.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.

    Returns:
        (Float[torch.Tensor, "*batch query reference 3 2"]): Bounds in
            axis then minimum, maximum order.
    """
    _check_pair(query_aabbs, reference_aabbs)
    query_minimum = query_aabbs[..., :, None, :3]
    query_maximum = query_aabbs[..., :, None, 3:]
    reference_minimum = reference_aabbs[..., None, :, :3]
    reference_maximum = reference_aabbs[..., None, :, 3:]
    return torch.stack(
        (torch.maximum(query_minimum, reference_minimum), torch.minimum(query_maximum, reference_maximum)), dim=-1
    )


@jaxtyped(typechecker=beartype)
def overlap_lengths(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    axis: AABBAxis,
) -> Float[torch.Tensor, "*batch query reference"]:
    """Return non-negative overlap lengths along one axis.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        axis (AABBAxis): Axis to measure.

    Returns:
        (Float[torch.Tensor, "*batch query reference"]): Overlap lengths.
    """
    _check_pair(query_aabbs, reference_aabbs)
    bounds = intersection_bounds(query_aabbs, reference_aabbs, axis)
    return torch.clamp(bounds[..., 1] - bounds[..., 0], min=0.0)


@jaxtyped(typechecker=beartype)
def overlap_lengths_all_axes(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
) -> Float[torch.Tensor, "*batch query reference 3"]:
    """Return non-negative overlap lengths along X, Y, and Z.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.

    Returns:
        (Float[torch.Tensor, "*batch query reference 3"]): X, Y, Z lengths.
    """
    _check_pair(query_aabbs, reference_aabbs)
    bounds = intersection_bounds_all_axes(query_aabbs, reference_aabbs)
    return torch.clamp(bounds[..., 1] - bounds[..., 0], min=0.0)


@jaxtyped(typechecker=beartype)
def query_face_contact_patches(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    *,
    distance_tolerance: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*distance_tolerance_batch query"],
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
) -> tuple[Float[torch.Tensor, "*batch query reference 6 6"], Bool[torch.Tensor, "*batch query reference 6"]]:
    """Return every query-face contact patch for every pair.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        distance_tolerance (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inclusive absolute-distance limit for a
            query face and its opposing reference face.
        minimum_face_crossing (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Minimum positive overlap length required on
            both tangential axes. Equality is accepted.

    Returns:
        (tuple[Float[torch.Tensor, "*batch query reference 6 6"],
            Bool[torch.Tensor, "*batch query reference 6"]]): Contact
            patches and their authoritative query-face contact mask, both
            in X-minimum, X-maximum, Y-minimum, Y-maximum, Z-minimum,
            Z-maximum face order. Patch coordinates use ordinary AABB
            order, and non-qualifying faces contain an all-zero AABB.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(distance_tolerance, query_aabbs, batch_shape, name="distance_tolerance", patch=False)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    result_signed_distances = signed_distances_all_faces(query_aabbs, reference_aabbs)
    absolute_distances = torch.abs(result_signed_distances)
    result_overlap_lengths = overlap_lengths_all_axes(query_aabbs, reference_aabbs)
    crossing_threshold = _add_reference_axis(minimum_face_crossing, add_feature_axis=True)
    sufficient_overlap = (result_overlap_lengths > 0.0) & (result_overlap_lengths >= crossing_threshold)
    tangential_overlap = torch.stack(
        (
            sufficient_overlap[..., 1] & sufficient_overlap[..., 2],
            sufficient_overlap[..., 1] & sufficient_overlap[..., 2],
            sufficient_overlap[..., 0] & sufficient_overlap[..., 2],
            sufficient_overlap[..., 0] & sufficient_overlap[..., 2],
            sufficient_overlap[..., 0] & sufficient_overlap[..., 1],
            sufficient_overlap[..., 0] & sufficient_overlap[..., 1],
        ),
        dim=-1,
    )
    distance_threshold = _add_reference_axis(distance_tolerance, add_feature_axis=True)
    face_contact_mask = tangential_overlap & (absolute_distances <= distance_threshold)
    result_intersection_bounds = intersection_bounds_all_axes(query_aabbs, reference_aabbs)
    query_face_coordinates = torch.stack(
        (
            query_aabbs[..., :, 0],
            query_aabbs[..., :, 3],
            query_aabbs[..., :, 1],
            query_aabbs[..., :, 4],
            query_aabbs[..., :, 2],
            query_aabbs[..., :, 5],
        ),
        dim=-1,
    )
    query_face_coordinates = torch.broadcast_to(query_face_coordinates[..., :, None, :], face_contact_mask.shape)
    normal_axes = (torch.arange(6, device=query_aabbs.device) // 2).expand(face_contact_mask.shape)
    normal_indices = torch.stack((normal_axes, normal_axes + 3), dim=-1)
    normal_coordinates = query_face_coordinates.unsqueeze(-1).expand(*face_contact_mask.shape, 2)
    intersection_aabbs = torch.cat((result_intersection_bounds[..., 0], result_intersection_bounds[..., 1]), dim=-1)
    patches = intersection_aabbs.unsqueeze(-2).expand(*face_contact_mask.shape, 6)
    patches = patches.scatter(-1, normal_indices, normal_coordinates)
    patches = torch.where(face_contact_mask.unsqueeze(-1), patches, torch.zeros_like(patches))
    return (patches, face_contact_mask)


@jaxtyped(typechecker=beartype)
def tangential_overlap_lengths(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
) -> Float[torch.Tensor, "*batch query reference 2"]:
    """Return overlap lengths along the two axes in a query face.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face that selects the axes.

    Returns:
        (Float[torch.Tensor, "*batch query reference 2"]): Tangential
            lengths in the face's documented axis order.
    """
    _check_pair(query_aabbs, reference_aabbs)
    first_axis, second_axis = _FACE_AXES[face].tangential_axes
    first_overlap = overlap_lengths(query_aabbs, reference_aabbs, first_axis)
    second_overlap = overlap_lengths(query_aabbs, reference_aabbs, second_axis)
    return torch.stack((first_overlap, second_overlap), dim=-1)


@jaxtyped(typechecker=beartype)
def projected_overlap_mask(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
    *,
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check strict overlap along both axes in a query face.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face whose tangential axes are checked.
        minimum_face_crossing (float | Float[torch.Tensor, ""] |
            Float[torch.Tensor, "*threshold_batch query"]): Minimum actual overlap
            length on both axes. Equality is accepted.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Projected-overlap
            mask.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    result_overlap_lengths = tangential_overlap_lengths(query_aabbs, reference_aabbs, face)
    crossing_threshold = _add_reference_axis(minimum_face_crossing, add_feature_axis=True)
    sufficient_overlap = (result_overlap_lengths > 0.0) & (result_overlap_lengths >= crossing_threshold)
    return torch.all(sufficient_overlap, dim=-1)


@jaxtyped(typechecker=beartype)
def inward_projected_overlap_mask(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
    *,
    inset: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*inset_batch query"],
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check strict overlap across inward bounds on two selected axes.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face whose tangential axes are checked.
        inset (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inward query-bound offset.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Inward-overlap
            mask for the selected tangential axes.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(inset, query_aabbs, batch_shape, name="inset", patch=False)
    result_axis_overlap = inward_axis_overlap_all_axes(query_aabbs, reference_aabbs, inset=inset)
    first_axis, second_axis = _FACE_AXES[face].tangential_axes
    return torch.all(result_axis_overlap[..., [_AXIS_INDICES[first_axis], _AXIS_INDICES[second_axis]]], dim=-1)


@jaxtyped(typechecker=beartype)
def inward_axis_overlap_all_axes(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    *,
    inset: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*inset_batch query"],
) -> Bool[torch.Tensor, "*batch query reference 3"]:
    """Check strict overlap across inward query bounds on every axis.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        inset (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inward query-bound offset.

    Returns:
        (Bool[torch.Tensor, "*batch query reference 3"]): X, Y, Z masks.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(inset, query_aabbs, batch_shape, name="inset", patch=False)
    query_minimum_minus_reference_maximum = query_aabbs[..., :, None, :3] - reference_aabbs[..., None, :, 3:]
    reference_minimum_minus_query_maximum = reference_aabbs[..., None, :, :3] - query_aabbs[..., :, None, 3:]
    inset_threshold = _add_reference_axis(inset, add_feature_axis=True)
    return (query_minimum_minus_reference_maximum < -inset_threshold) & (
        reference_minimum_minus_query_maximum < -inset_threshold
    )


@jaxtyped(typechecker=beartype)
def projected_overlap_areas(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
) -> Float[torch.Tensor, "*batch query reference"]:
    """Return overlap area after projecting onto a query face.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face that selects the projection plane.

    Returns:
        (Float[torch.Tensor, "*batch query reference"]): Projected areas.
    """
    _check_pair(query_aabbs, reference_aabbs)
    tangential_lengths = tangential_overlap_lengths(query_aabbs, reference_aabbs, face)
    return tangential_lengths[..., 0] * tangential_lengths[..., 1]


@jaxtyped(typechecker=beartype)
def query_face_areas(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"], face: AABBFace
) -> Float[torch.Tensor, "*query_batch query"]:
    """Return the full area of one face on every query AABB.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        face (AABBFace): Face to measure.

    Returns:
        (Float[torch.Tensor, "*batch query"]): Face areas.
    """
    _check_aabbs(query_aabbs, name="query_aabbs")
    first_axis, second_axis = _FACE_AXES[face].tangential_axes
    first_axis_index = _AXIS_INDICES[first_axis]
    second_axis_index = _AXIS_INDICES[second_axis]
    first_length = query_aabbs[..., first_axis_index + 3] - query_aabbs[..., first_axis_index]
    second_length = query_aabbs[..., second_axis_index + 3] - query_aabbs[..., second_axis_index]
    return first_length * second_length


@jaxtyped(typechecker=beartype)
def projected_intersection_bounds(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
) -> Float[torch.Tensor, "*batch query reference 4"]:
    """Return raw intersection bounds in query-face coordinates.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face that selects axes and their order.

    Returns:
        (Float[torch.Tensor, "*batch query reference 4"]): First-axis
            minimum, second-axis minimum, first-axis maximum, second-axis
            maximum. Separate intervals retain inverted bounds.
    """
    _check_pair(query_aabbs, reference_aabbs)
    first_axis, second_axis = _FACE_AXES[face].tangential_axes
    first_bounds = intersection_bounds(query_aabbs, reference_aabbs, first_axis)
    second_bounds = intersection_bounds(query_aabbs, reference_aabbs, second_axis)
    return torch.stack(
        (first_bounds[..., 0], second_bounds[..., 0], first_bounds[..., 1], second_bounds[..., 1]), dim=-1
    )


@jaxtyped(typechecker=beartype)
def within_distance(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
    *,
    minimum_distance: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*minimum_distance_batch query"],
    maximum_distance: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*maximum_distance_batch query"],
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check references within a signed distance range of a query face.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face to measure from.
        minimum_distance (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inclusive lower distance limit.
        maximum_distance (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inclusive upper distance limit.
        minimum_face_crossing (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Minimum tangential overlap length.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Distance mask.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(maximum_distance, query_aabbs, batch_shape, name="maximum_distance", patch=False)
    _check_threshold(minimum_distance, query_aabbs, batch_shape, name="minimum_distance", patch=False)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    distances = signed_distances(query_aabbs, reference_aabbs, face)
    projected_overlap = projected_overlap_mask(
        query_aabbs, reference_aabbs, face, minimum_face_crossing=minimum_face_crossing
    )
    minimum_threshold = _add_reference_axis(minimum_distance)
    maximum_threshold = _add_reference_axis(maximum_distance)
    return projected_overlap & (distances >= minimum_threshold) & (distances <= maximum_threshold)


@jaxtyped(typechecker=beartype)
def contact_mask(
    query_aabbs: Float[torch.Tensor, "*query_batch query 6"],
    reference_aabbs: Float[torch.Tensor, "*reference_batch reference 6"],
    face: AABBFace,
    *,
    distance_tolerance: float | Float[torch.Tensor, ""] | Float[torch.Tensor, "*distance_tolerance_batch query"],
    minimum_face_crossing: float
    | Float[torch.Tensor, ""]
    | Float[torch.Tensor, "*minimum_face_crossing_batch query"] = 0.0,
    minimum_patch_lengths: tuple[float, float] | Float[torch.Tensor, "*minimum_patch_lengths_batch query 2"] = (
        0.0,
        0.0,
    ),
) -> Bool[torch.Tensor, "*batch query reference"]:
    """Check whether faces are close and share a positive-area patch.

    Args:
        query_aabbs (Float[torch.Tensor, "*query_batch query 6"]): Query AABBs.
        reference_aabbs (Float[torch.Tensor, "*reference_batch reference 6"]):
            Reference AABBs.
        face (AABBFace): Query face to check.
        distance_tolerance (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Inclusive absolute-distance limit.
        minimum_face_crossing (float | Float[torch.Tensor, ""] | Float[torch.Tensor,
            "*threshold_batch query"]): Minimum tangential overlap length.
        minimum_patch_lengths (tuple[float, float] | Float[torch.Tensor,
            "*threshold_batch query 2"]): Minimum length for each tangential axis.

    Returns:
        (Bool[torch.Tensor, "*batch query reference"]): Contact mask.
    """
    batch_shape = _check_pair(query_aabbs, reference_aabbs)
    _check_threshold(distance_tolerance, query_aabbs, batch_shape, name="distance_tolerance", patch=False)
    _check_threshold(minimum_face_crossing, query_aabbs, batch_shape, name="minimum_face_crossing", patch=False)
    _check_threshold(minimum_patch_lengths, query_aabbs, batch_shape, name="minimum_patch_lengths", patch=True)
    distances = signed_distances(query_aabbs, reference_aabbs, face)
    tangential_lengths = tangential_overlap_lengths(query_aabbs, reference_aabbs, face)
    crossing_threshold = _add_reference_axis(minimum_face_crossing, add_feature_axis=True)
    projected_overlap = torch.all((tangential_lengths > 0.0) & (tangential_lengths >= crossing_threshold), dim=-1)
    if isinstance(minimum_patch_lengths, tuple):
        patch_threshold = torch.tensor(minimum_patch_lengths, dtype=torch.float64, device=tangential_lengths.device)
    else:
        patch_threshold = minimum_patch_lengths[..., :, None, :]
    distance_threshold = _add_reference_axis(distance_tolerance)
    inside_distance_tolerance = torch.abs(distances) <= distance_threshold
    patch_meets_minimum_lengths = torch.all(tangential_lengths >= patch_threshold, dim=-1)
    return projected_overlap & inside_distance_tolerance & patch_meets_minimum_lengths
