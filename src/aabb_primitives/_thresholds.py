"""Validate thresholds used by pairwise AABB relations."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def _validate_minimum_face_crossing(minimum_face_crossing: float) -> None:
    """Check a face-crossing threshold.

    Args:
        minimum_face_crossing (float): Value to check. It must be finite and
            non-negative.
    """
    if not np.isfinite(minimum_face_crossing) or minimum_face_crossing < 0.0:
        raise ValueError("minimum_face_crossing must be finite and non-negative")


@jaxtyped(typechecker=beartype)
def _normalize_query_threshold(
    threshold: float | Float[np.ndarray, "query"] | Float[np.ndarray, "environment world query"],  # noqa: F821
    *,
    query_shape: tuple[int] | tuple[int, int, int],
    name: str,
    require_non_negative: bool,
) -> float | Float[np.ndarray, "query 1"] | Float[np.ndarray, "environment world query 1"]:
    """Prepare a scalar or per-query threshold for pairwise comparisons.

    Args:
        threshold (float | Float[np.ndarray, "query"] | Float[np.ndarray,
            "environment world query"]): A finite scalar or one value per
            query in either an unbatched or rank-four relation.
        query_shape (tuple[int] | tuple[int, int, int]): Exact query shape,
            either ``(Q,)`` or ``(B, K, Q)``.
        name (str): Parameter name to show in error messages.
        require_non_negative (bool): If true, reject negative values.

    Returns:
        (float | Float[np.ndarray, "query 1"] | Float[np.ndarray,
            "environment world query 1"]): The original scalar, or the query
            array with one trailing reference dimension.
    """
    if isinstance(threshold, float):
        if not np.isfinite(threshold):
            raise ValueError(f"{name} must be finite")
        if require_non_negative and threshold < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return threshold

    if threshold.shape != query_shape:
        raise ValueError(f"{name} must be a scalar or have shape {query_shape}")
    if not np.all(np.isfinite(threshold)):
        raise ValueError(f"{name} must be finite")
    if require_non_negative and np.any(threshold < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return threshold[..., None]
