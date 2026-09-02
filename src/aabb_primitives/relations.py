"""Calculate pairwise geometry for unbatched or environment-world AABBs.

Inputs have shape ``(Q, 6)`` and ``(R, 6)``, or the explicit batched-world
shapes ``(B, K, Q, 6)`` and ``(B, K, R, 6)``. Pairwise arrays put queries
first, so ``result[q, r]`` or ``result[b, k, q, r]`` compares query ``q``
with reference ``r`` inside one independent world. The calculations do not
add packing rules, rounding, snapping, or coverage rules.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from aabb_primitives._thresholds import _normalize_query_threshold, _validate_minimum_face_crossing

__all__ = ["AABBAxis", "AABBFace", "PairwiseAABBRelations"]


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
    AABBFace.X_MIN: _FaceAxes(normal_axis=AABBAxis.X, tangential_axes=(AABBAxis.Y, AABBAxis.Z), is_min_face=True),
    AABBFace.X_MAX: _FaceAxes(normal_axis=AABBAxis.X, tangential_axes=(AABBAxis.Y, AABBAxis.Z), is_min_face=False),
    AABBFace.Y_MIN: _FaceAxes(normal_axis=AABBAxis.Y, tangential_axes=(AABBAxis.X, AABBAxis.Z), is_min_face=True),
    AABBFace.Y_MAX: _FaceAxes(normal_axis=AABBAxis.Y, tangential_axes=(AABBAxis.X, AABBAxis.Z), is_min_face=False),
    AABBFace.Z_MIN: _FaceAxes(normal_axis=AABBAxis.Z, tangential_axes=(AABBAxis.X, AABBAxis.Y), is_min_face=True),
    AABBFace.Z_MAX: _FaceAxes(normal_axis=AABBAxis.Z, tangential_axes=(AABBAxis.X, AABBAxis.Y), is_min_face=False),
}


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PairwiseAABBRelations:
    """Store pairwise relationships between query and reference AABBs.

    Use :meth:`from_aabbs` to create an instance. Each input row has the form
    ``[xmin, ymin, zmin, xmax, ymax, zmax]``. Pairwise results use query-first
    order. Inputs are either rank two, ``(Q, 6)`` and ``(R, 6)``, or rank
    four, ``(B, K, Q, 6)`` and ``(B, K, R, 6)``. Rank-four query and reference
    inputs must have identical environment and world dimensions; the class
    does not broadcast independent worlds.

    :class:`AABBAxis` selects an axis. :class:`AABBFace` selects a face of the
    query AABB and compares it with the opposite face of the reference AABB.
    A signed face distance is positive for a gap, zero when the faces line up,
    and negative after they cross. A negative value is not always the depth of
    the overlap.

    For X faces, tangential results use Y, Z order. Y faces use X, Z order,
    and Z faces use X, Y order.

    Construction takes O(Q * R) time per world. It copies both inputs and
    stores the copies and signed-distance arrays as read-only arrays. Selecting
    one signed-distance array is an O(1) read-only view. Bounds, lengths, areas,
    and masks take O(Q * R) time per world each time they are requested.
    Results are not cached.

    Attributes:
        _reference_aabbs (np.ndarray): Read-only reference snapshots.
        _query_aabbs (np.ndarray): Read-only query snapshots.
        _query_min_minus_reference_max (np.ndarray): Query minimum minus
            reference maximum in X, Y, Z order. The array is read-only.
        _reference_min_minus_query_max (np.ndarray): Reference minimum minus
            query maximum in X, Y, Z order. The array is read-only.
    """

    _reference_aabbs: (
        Float[np.ndarray, "reference 6"] | Float[np.ndarray, "reference_environment reference_world reference 6"]
    ) = field(repr=False)
    _query_aabbs: Float[np.ndarray, "query 6"] | Float[np.ndarray, "query_environment query_world query 6"] = field(
        repr=False
    )
    _query_min_minus_reference_max: (
        Float[np.ndarray, "query reference 3"] | Float[np.ndarray, "environment world query reference 3"]
    ) = field(repr=False)
    _reference_min_minus_query_max: (
        Float[np.ndarray, "query reference 3"] | Float[np.ndarray, "environment world query reference 3"]
    ) = field(repr=False)

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_aabbs(
        cls,
        reference_aabbs: Float[np.ndarray, "reference 6"]
        | Float[np.ndarray, "reference_environment reference_world reference 6"],
        query_aabbs: Float[np.ndarray, "query 6"] | Float[np.ndarray, "query_environment query_world query 6"],
    ) -> "PairwiseAABBRelations":
        """Build pairwise relationships from two AABB arrays.

        Each row must be ``[xmin, ymin, zmin, xmax, ymax, zmax]``. Both inputs
        must have rank two or both must have rank four. Rank-four inputs must
        have matching ``B`` and ``K`` dimensions. Coordinates must be finite,
        and each maximum must be greater than or equal to its minimum. Empty
        dimensions and zero-length axes are valid. This method copies both
        inputs, so changing them later does not change the result.

        Args:
            reference_aabbs (np.ndarray): Reference rows with shape
                ``(R, 6)`` or ``(B, K, R, 6)``.
            query_aabbs (np.ndarray): Query rows with shape ``(Q, 6)`` or
                ``(B, K, Q, 6)``.

        Returns:
            (PairwiseAABBRelations): Relations for every query and reference
                pair.
        """
        if reference_aabbs.ndim != query_aabbs.ndim:
            raise ValueError("reference_aabbs and query_aabbs must have the same rank")
        if reference_aabbs.ndim == 4 and reference_aabbs.shape[:2] != query_aabbs.shape[:2]:
            raise ValueError(
                "rank-four reference_aabbs and query_aabbs must have matching environment and world dimensions"
            )
        if not np.all(np.isfinite(reference_aabbs)):
            raise ValueError("reference_aabbs must contain only finite coordinates")
        if not np.all(np.isfinite(query_aabbs)):
            raise ValueError("query_aabbs must contain only finite coordinates")
        if np.any(reference_aabbs[..., :3] > reference_aabbs[..., 3:]):
            raise ValueError("reference_aabbs must have non-negative extents")
        if np.any(query_aabbs[..., :3] > query_aabbs[..., 3:]):
            raise ValueError("query_aabbs must have non-negative extents")

        reference_aabb_snapshot = reference_aabbs.copy()
        query_aabb_snapshot = query_aabbs.copy()
        reference_aabb_snapshot.setflags(write=False)
        query_aabb_snapshot.setflags(write=False)

        reference_min = reference_aabb_snapshot[..., :3]
        reference_max = reference_aabb_snapshot[..., 3:]
        query_min = query_aabb_snapshot[..., :3]
        query_max = query_aabb_snapshot[..., 3:]

        query_min_minus_reference_max = query_min[..., :, None, :] - reference_max[..., None, :, :]
        reference_min_minus_query_max = reference_min[..., None, :, :] - query_max[..., :, None, :]
        query_min_minus_reference_max.setflags(write=False)
        reference_min_minus_query_max.setflags(write=False)
        return cls(
            _reference_aabbs=reference_aabb_snapshot,
            _query_aabbs=query_aabb_snapshot,
            _query_min_minus_reference_max=query_min_minus_reference_max,
            _reference_min_minus_query_max=reference_min_minus_query_max,
        )

    @jaxtyped(typechecker=beartype)
    def signed_distances(
        self, face: AABBFace
    ) -> Float[np.ndarray, "query reference"] | Float[np.ndarray, "environment world query reference"]:
        """Return the signed distance from one query face to its opposite.

        A minimum query face is measured against the maximum reference face on
        the same axis. A maximum query face is measured against the minimum
        reference face. The returned array is a read-only view. Positive
        values are gaps, zero means that the faces line up, and negative values
        mean that they have crossed.

        Args:
            face (AABBFace): Face to measure on every query AABB.

        Returns:
            (Float[np.ndarray]): Signed distances with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        face_axes = _FACE_AXES[face]
        axis_index = _AXIS_INDICES[face_axes.normal_axis]
        if face_axes.is_min_face:
            return self._query_min_minus_reference_max[..., axis_index]
        return self._reference_min_minus_query_max[..., axis_index]

    @jaxtyped(typechecker=beartype)
    def signed_distances_all_faces(
        self,
    ) -> Float[np.ndarray, "query reference 6"] | Float[np.ndarray, "environment world query reference 6"]:
        """Return signed distances for all six query faces.

        The last dimension is ordered X-minimum, X-maximum, Y-minimum,
        Y-maximum, Z-minimum, Z-maximum.

        Returns:
            (Float[np.ndarray]): Signed distances with shape ``(Q, R, 6)`` or
                ``(B, K, Q, R, 6)``.
        """
        signed_distances = np.stack(
            (
                self._query_min_minus_reference_max[..., 0],
                self._reference_min_minus_query_max[..., 0],
                self._query_min_minus_reference_max[..., 1],
                self._reference_min_minus_query_max[..., 1],
                self._query_min_minus_reference_max[..., 2],
                self._reference_min_minus_query_max[..., 2],
            ),
            axis=-1,
        )
        return signed_distances

    @jaxtyped(typechecker=beartype)
    def axis_overlap(
        self, axis: AABBAxis, *, minimum_face_crossing: float = 0.0
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check whether each pair overlaps strictly along one axis.

        The actual overlap length must be positive and greater than or equal to
        ``minimum_face_crossing``. With a threshold of zero, touching,
        separated, and zero-length intersections do not overlap.

        Args:
            axis (AABBAxis): Axis to check.
            minimum_face_crossing (float): Smallest actual overlap length to
                accept. Equality is accepted. The value must be finite and
                non-negative.

        Returns:
            (Bool[np.ndarray]): Overlap mask with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        _validate_minimum_face_crossing(minimum_face_crossing)
        overlap_lengths = self.overlap_lengths(axis)
        # Reject zero-length contact at the zero default; >= still accepts threshold equality.
        axis_overlap = (overlap_lengths > 0.0) & (overlap_lengths >= minimum_face_crossing)
        return axis_overlap

    @jaxtyped(typechecker=beartype)
    def axis_overlap_all_axes(
        self, *, minimum_face_crossing: float = 0.0
    ) -> Bool[np.ndarray, "query reference 3"] | Bool[np.ndarray, "environment world query reference 3"]:
        """Check strict overlap along X, Y, and Z.

        Each actual overlap length must be positive and greater than or equal
        to ``minimum_face_crossing``.

        Args:
            minimum_face_crossing (float): Smallest actual overlap length to
                accept on each axis. Equality is accepted. The value must be
                finite and non-negative.

        Returns:
            (Bool[np.ndarray]): X, Y, Z overlap mask with shape ``(Q, R, 3)``
                or ``(B, K, Q, R, 3)``.
        """
        _validate_minimum_face_crossing(minimum_face_crossing)
        overlap_lengths = self.overlap_lengths_all_axes()
        # Reject zero-length contact at the zero default; >= still accepts threshold equality.
        axis_overlap = (overlap_lengths > 0.0) & (overlap_lengths >= minimum_face_crossing)
        return axis_overlap

    @jaxtyped(typechecker=beartype)
    def contained_by_mask(
        self,
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check whether each query AABB is contained by each reference AABB.

        Containment is inclusive on all six bounds, so identical AABBs contain
        one another. The method applies no tolerance, rounding, or
        self-exclusion. Empty batches return their natural query-by-reference
        shape.

        Returns:
            (Bool[np.ndarray]): Containment mask with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        query_minimum = self._query_aabbs[..., :, None, :3]
        query_maximum = self._query_aabbs[..., :, None, 3:]
        reference_minimum = self._reference_aabbs[..., None, :, :3]
        reference_maximum = self._reference_aabbs[..., None, :, 3:]

        contained_by = np.all((query_minimum >= reference_minimum) & (query_maximum <= reference_maximum), axis=-1)
        return contained_by

    @jaxtyped(typechecker=beartype)
    def intersection_bounds(
        self, axis: AABBAxis
    ) -> Float[np.ndarray, "query reference 2"] | Float[np.ndarray, "environment world query reference 2"]:
        """Return the raw intersection interval along one axis.

        The last dimension contains the minimum and maximum coordinates. If
        two intervals are separate, the minimum is greater than the maximum.
        The method does not clip or replace that inverted interval.

        Args:
            axis (AABBAxis): Axis to measure.

        Returns:
            (Float[np.ndarray]): Intersection bounds with shape ``(Q, R, 2)``
                or ``(B, K, Q, R, 2)``.
        """
        axis_index = _AXIS_INDICES[axis]
        intersection_minimum = np.maximum(
            self._query_aabbs[..., :, None, axis_index], self._reference_aabbs[..., None, :, axis_index]
        )
        intersection_maximum = np.minimum(
            self._query_aabbs[..., :, None, axis_index + 3], self._reference_aabbs[..., None, :, axis_index + 3]
        )
        intersection_bounds = np.stack((intersection_minimum, intersection_maximum), axis=-1)
        return intersection_bounds

    @jaxtyped(typechecker=beartype)
    def intersection_bounds_all_axes(
        self,
    ) -> Float[np.ndarray, "query reference 3 2"] | Float[np.ndarray, "environment world query reference 3 2"]:
        """Return raw intersection intervals along X, Y, and Z.

        The final two dimensions are ``(axis, bound)``. Axes use X, Y, Z order,
        and bounds use minimum, maximum order. Separate intervals keep their
        inverted bounds.

        Returns:
            (Float[np.ndarray]): Intersection bounds with shape
                ``(Q, R, 3, 2)`` or ``(B, K, Q, R, 3, 2)``.
        """
        intersection_minimum = np.maximum(self._query_aabbs[..., :, None, :3], self._reference_aabbs[..., None, :, :3])
        intersection_maximum = np.minimum(self._query_aabbs[..., :, None, 3:], self._reference_aabbs[..., None, :, 3:])
        intersection_bounds = np.stack((intersection_minimum, intersection_maximum), axis=-1)
        return intersection_bounds

    @jaxtyped(typechecker=beartype)
    def overlap_lengths(
        self, axis: AABBAxis
    ) -> Float[np.ndarray, "query reference"] | Float[np.ndarray, "environment world query reference"]:
        """Return the overlap length along one axis.

        Touching or separate intervals have length zero.

        Args:
            axis (AABBAxis): Axis to measure.

        Returns:
            (Float[np.ndarray]): Non-negative lengths with shape ``(Q, R)``
                or ``(B, K, Q, R)``.
        """
        intersection_bounds = self.intersection_bounds(axis)
        overlap_lengths = np.maximum(intersection_bounds[..., 1] - intersection_bounds[..., 0], 0.0)
        return overlap_lengths

    @jaxtyped(typechecker=beartype)
    def overlap_lengths_all_axes(
        self,
    ) -> Float[np.ndarray, "query reference 3"] | Float[np.ndarray, "environment world query reference 3"]:
        """Return overlap lengths along X, Y, and Z.

        Returns:
            (Float[np.ndarray]): X, Y, Z lengths with shape ``(Q, R, 3)`` or
                ``(B, K, Q, R, 3)``.
        """
        intersection_bounds = self.intersection_bounds_all_axes()
        overlap_lengths = np.maximum(intersection_bounds[..., 1] - intersection_bounds[..., 0], 0.0)
        return overlap_lengths

    @jaxtyped(typechecker=beartype)
    def tangential_overlap_lengths(
        self, face: AABBFace
    ) -> Float[np.ndarray, "query reference 2"] | Float[np.ndarray, "environment world query reference 2"]:
        """Return the two overlap lengths that lie in a query face.

        Args:
            face (AABBFace): Query face that sets the two axes to measure.

        Returns:
            (Float[np.ndarray]): Non-negative lengths with shape ``(Q, R, 2)``
                or ``(B, K, Q, R, 2)``. X faces use Y, Z order. Y faces use X,
                Z order. Z faces use X, Y order.
        """
        first_axis, second_axis = _FACE_AXES[face].tangential_axes
        first_overlap = self.overlap_lengths(first_axis)
        second_overlap = self.overlap_lengths(second_axis)
        tangential_overlap_lengths = np.stack((first_overlap, second_overlap), axis=-1)
        return tangential_overlap_lengths

    @jaxtyped(typechecker=beartype)
    def projected_overlap_mask(
        self, face: AABBFace, *, minimum_face_crossing: float = 0.0
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check overlap along both axes that lie in a query face.

        Both actual overlap lengths must be positive and greater than or equal
        to ``minimum_face_crossing``. The same scalar threshold is applied
        independently to both axes.

        Args:
            face (AABBFace): Query face that sets the two axes to check.
            minimum_face_crossing (float): Smallest actual overlap length to
                accept on each axis. Equality is accepted. The value must be
                finite and non-negative.

        Returns:
            (Bool[np.ndarray]): Projected-overlap mask with shape ``(Q, R)``
                or ``(B, K, Q, R)``.
        """
        _validate_minimum_face_crossing(minimum_face_crossing)
        overlap_lengths = self.tangential_overlap_lengths(face)
        # Reject zero-length contact at the zero default; >= still accepts threshold equality.
        sufficient_overlap = (overlap_lengths > 0.0) & (overlap_lengths >= minimum_face_crossing)
        return np.all(sufficient_overlap, axis=-1)

    @jaxtyped(typechecker=beartype)
    def inward_projected_overlap_mask(
        self, face: AABBFace, *, inset: float
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check strict overlap across the inward bounds of a query face.

        Each query bound on both tangential axes is moved inward by ``inset``.
        A reference must cross both resulting open boundaries on both axes.

        Args:
            face (AABBFace): Query face that sets the two axes to check.
            inset (float): Finite non-negative inward offset applied to every
                tangential query bound.

        Returns:
            (Bool[np.ndarray]): Inward-overlap mask with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        axis_overlap = self.inward_axis_overlap_all_axes(inset=inset)
        first_axis, second_axis = _FACE_AXES[face].tangential_axes
        tangential_axis_indices = (_AXIS_INDICES[first_axis], _AXIS_INDICES[second_axis])
        return np.all(np.take(axis_overlap, tangential_axis_indices, axis=-1), axis=-1)

    @jaxtyped(typechecker=beartype)
    def inward_axis_overlap_all_axes(
        self, *, inset: float
    ) -> Bool[np.ndarray, "query reference 3"] | Bool[np.ndarray, "environment world query reference 3"]:
        """Check strict overlap across inward query bounds on every axis.

        Both bounds of each query interval are moved inward by ``inset``. A
        reference must strictly cross the resulting open interval. Results are
        returned in X, Y, Z order.

        Args:
            inset (float): Finite non-negative inward offset applied to every
                query bound.

        Returns:
            (Bool[np.ndarray]): X, Y, Z inward-overlap mask with shape
                ``(Q, R, 3)`` or ``(B, K, Q, R, 3)``.
        """
        if not np.isfinite(inset) or inset < 0.0:
            raise ValueError("inset must be finite and non-negative")

        crosses_lower_bounds = self._query_min_minus_reference_max < -inset
        crosses_upper_bounds = self._reference_min_minus_query_max < -inset
        return crosses_lower_bounds & crosses_upper_bounds

    @jaxtyped(typechecker=beartype)
    def projected_overlap_areas(
        self, face: AABBFace
    ) -> Float[np.ndarray, "query reference"] | Float[np.ndarray, "environment world query reference"]:
        """Return the overlap area after projecting onto a query face.

        Distance along the axis that points out of the face does not change the
        area.

        Args:
            face (AABBFace): Query face that sets the projection plane.

        Returns:
            (Float[np.ndarray]): Projected areas with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        tangential_lengths = self.tangential_overlap_lengths(face)
        projected_overlap_areas = tangential_lengths[..., 0] * tangential_lengths[..., 1]
        return projected_overlap_areas

    @jaxtyped(typechecker=beartype)
    def query_face_areas(
        self, face: AABBFace
    ) -> Float[np.ndarray, "query"] | Float[np.ndarray, "environment world query"]:  # noqa: F821
        """Return the full area of one face on every query AABB.

        Args:
            face (AABBFace): Face to measure.

        Returns:
            (Float[np.ndarray]): Query face areas with shape ``(Q,)`` or
                ``(B, K, Q)``.
        """
        first_axis, second_axis = _FACE_AXES[face].tangential_axes
        first_axis_index = _AXIS_INDICES[first_axis]
        second_axis_index = _AXIS_INDICES[second_axis]
        first_length = self._query_aabbs[..., first_axis_index + 3] - self._query_aabbs[..., first_axis_index]
        second_length = self._query_aabbs[..., second_axis_index + 3] - self._query_aabbs[..., second_axis_index]
        query_face_areas = first_length * second_length
        return query_face_areas

    @jaxtyped(typechecker=beartype)
    def projected_intersection_bounds(
        self, face: AABBFace
    ) -> Float[np.ndarray, "query reference 4"] | Float[np.ndarray, "environment world query reference 4"]:
        """Return raw intersection bounds in the coordinates of a query face.

        Separate intervals keep their inverted bounds. The method does not
        clip, round, filter, or replace values with ``NaN``.

        Args:
            face (AABBFace): Query face that sets the two axes and their order.

        Returns:
            (Float[np.ndarray]): Projected bounds with shape ``(Q, R, 4)`` or
                ``(B, K, Q, R, 4)``. Bounds are ordered as first-axis minimum,
                second-axis minimum, first-axis maximum, second-axis maximum.
        """
        first_axis, second_axis = _FACE_AXES[face].tangential_axes
        first_bounds = self.intersection_bounds(first_axis)
        second_bounds = self.intersection_bounds(second_axis)
        projected_bounds = np.stack(
            (first_bounds[..., 0], second_bounds[..., 0], first_bounds[..., 1], second_bounds[..., 1]), axis=-1
        )
        return projected_bounds

    @jaxtyped(typechecker=beartype)
    def within_distance(
        self,
        face: AABBFace,
        *,
        minimum_distance: float | Float[np.ndarray, "query"] | Float[np.ndarray, "environment world query"],  # noqa: F821
        maximum_distance: float | Float[np.ndarray, "query"] | Float[np.ndarray, "environment world query"],  # noqa: F821
        minimum_face_crossing: float = 0.0,
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check which references are within a distance range of a query face.

        A pair must overlap along both axes in the face. Its signed face
        distance must also be between the two limits, including both limits.
        A scalar limit applies to every pair. An array must have shape ``(Q,)``
        for rank-two relations or ``(B, K, Q)`` for rank-four relations. The
        limits are checked even when a dimension is empty.

        Args:
            face (AABBFace): Query face to measure from.
            minimum_distance (float | np.ndarray): Smallest signed
                distance to include.
            maximum_distance (float | np.ndarray): Largest signed
                distance to include.
            minimum_face_crossing (float): Smallest actual overlap length to
                accept on each of the two axes in the face. Equality is
                accepted. The value must be finite and non-negative.

        Returns:
            (Bool[np.ndarray]): Distance mask with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        query_shape = self._query_aabbs.shape[:-1]
        normalized_minimum = _normalize_query_threshold(
            minimum_distance, query_shape=query_shape, name="minimum_distance", require_non_negative=False
        )
        normalized_maximum = _normalize_query_threshold(
            maximum_distance, query_shape=query_shape, name="maximum_distance", require_non_negative=False
        )
        if np.any(normalized_minimum > normalized_maximum):
            raise ValueError("minimum_distance must be less than or equal to maximum_distance")

        distances = self.signed_distances(face)
        projected_overlap = self.projected_overlap_mask(face, minimum_face_crossing=minimum_face_crossing)
        in_distance_window = (distances >= normalized_minimum) & (distances <= normalized_maximum)
        return projected_overlap & in_distance_window

    @jaxtyped(typechecker=beartype)
    def contact_mask(
        self,
        face: AABBFace,
        *,
        distance_tolerance: float | Float[np.ndarray, "query"] | Float[np.ndarray, "environment world query"],  # noqa: F821
        minimum_face_crossing: float = 0.0,
        minimum_patch_lengths: tuple[float, float] = (0.0, 0.0),
    ) -> Bool[np.ndarray, "query reference"] | Bool[np.ndarray, "environment world query reference"]:
        """Check whether two faces are close and share a positive-area patch.

        The absolute signed distance must be less than or equal to
        ``distance_tolerance``. Both overlap lengths in the face must be
        greater than zero. They must also be greater than or equal to the two
        requested patch lengths. With ``(0.0, 0.0)``, any patch with positive
        area is accepted. Edge contact and zero-area patches are rejected.

        Args:
            face (AABBFace): Query face to check.
            distance_tolerance (float | np.ndarray): Finite non-negative
                limit for the absolute signed distance. An array must have
                shape ``(Q,)`` or ``(B, K, Q)`` matching the relation rank.
            minimum_face_crossing (float): Smallest actual overlap length to
                accept on each of the two axes in the face. Equality is
                accepted. The value must be finite and non-negative.
            minimum_patch_lengths (tuple[float, float]): Smallest allowed
                overlap length on each axis in the face. Both values must be
                finite and non-negative.

        Returns:
            (Bool[np.ndarray]): Contact mask with shape ``(Q, R)`` or
                ``(B, K, Q, R)``.
        """
        normalized_tolerance = _normalize_query_threshold(
            distance_tolerance,
            query_shape=self._query_aabbs.shape[:-1],
            name="distance_tolerance",
            require_non_negative=True,
        )
        minimum_patch_lengths_array = np.asarray(minimum_patch_lengths, dtype=np.float64)
        if not np.all(np.isfinite(minimum_patch_lengths_array)):
            raise ValueError("minimum_patch_lengths must be finite")
        if np.any(minimum_patch_lengths_array < 0.0):
            raise ValueError("minimum_patch_lengths must be non-negative")

        distances = self.signed_distances(face)
        tangential_lengths = self.tangential_overlap_lengths(face)
        projected_overlap = self.projected_overlap_mask(face, minimum_face_crossing=minimum_face_crossing)
        positive_actual_patch = np.all(tangential_lengths > 0.0, axis=-1)
        inside_distance_tolerance = np.abs(distances) <= normalized_tolerance
        patch_meets_minimum_lengths = np.all(tangential_lengths >= minimum_patch_lengths_array, axis=-1)
        return projected_overlap & positive_actual_patch & inside_distance_tolerance & patch_meets_minimum_lengths
