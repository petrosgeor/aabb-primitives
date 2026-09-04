"""Structural checks for operations and explicit numerical boundary checks."""

import math

import torch


def _check_aabbs(aabbs: torch.Tensor, *, name: str = "aabbs") -> None:
    """Check box metadata without reading coordinate values."""
    if not isinstance(aabbs, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if aabbs.ndim < 2 or aabbs.shape[-1] != 6:
        raise ValueError(f"{name} must have shape (*batch, rows, 6)")
    if aabbs.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if aabbs.layout != torch.strided:
        raise TypeError(f"{name} must use a dense strided layout")


def _check_pair(query_aabbs: torch.Tensor, reference_aabbs: torch.Tensor) -> torch.Size:
    """Check pair metadata and return the resolved leading batch shape."""
    _check_aabbs(query_aabbs, name="query_aabbs")
    _check_aabbs(reference_aabbs, name="reference_aabbs")
    if query_aabbs.dtype != reference_aabbs.dtype:
        raise TypeError("query_aabbs and reference_aabbs must have the same dtype")
    if query_aabbs.device != reference_aabbs.device:
        raise ValueError("query_aabbs and reference_aabbs must be on the same device")
    try:
        return torch.broadcast_shapes(query_aabbs.shape[:-2], reference_aabbs.shape[:-2])
    except RuntimeError as error:
        raise ValueError("query and reference batch dimensions must be broadcastable") from error


def _check_threshold(
    value: float | torch.Tensor | tuple[float, float],
    queries: torch.Tensor,
    batch_shape: torch.Size,
    *,
    name: str,
    patch: bool = False,
) -> None:
    """Check threshold metadata without allowing new batch or query axes."""
    if patch and isinstance(value, tuple):
        if len(value) != 2 or not all(isinstance(item, float) for item in value):
            raise TypeError(f"{name} must be a pair of floats or a query-aligned tensor")
        return
    if not patch and isinstance(value, float):
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a float or a floating-point tensor")
    if value.dtype not in (torch.float32, torch.float64) or value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense float32 or float64 tensor")
    if value.device != queries.device:
        raise ValueError(f"{name} must be on the same device as the AABBs")
    if value.ndim == 0 and not patch:
        return
    tail = (queries.shape[-2], 2) if patch else (queries.shape[-2],)
    if value.ndim < len(tail) or tuple(value.shape[-len(tail) :]) != tail:
        raise ValueError(f"{name} must have trailing dimensions {tail}")
    try:
        resolved = torch.broadcast_shapes(value.shape[: -len(tail)], batch_shape)
    except RuntimeError as error:
        raise ValueError(f"{name} batch dimensions must broadcast into {tuple(batch_shape)}") from error
    if resolved != batch_shape:
        raise ValueError(f"{name} must not introduce batch dimensions beyond {tuple(batch_shape)}")


def validate_aabbs(aabbs: torch.Tensor) -> None:
    """Check finite coordinates and non-negative extents without changing input.

    Zero extents and empty collections are valid. This explicit check scans
    values and may synchronize an accelerator; geometry operations do not
    call it automatically.

    Args:
        aabbs (torch.Tensor): Float32 or float64 boxes shaped (*batch, rows, 6).
    """
    _check_aabbs(aabbs)
    if not torch.isfinite(aabbs).all():
        raise ValueError("aabbs must contain only finite coordinates")
    if torch.any(aabbs[..., :3] > aabbs[..., 3:]):
        raise ValueError("aabbs must have non-negative extents")


def _validate_numeric(value: float | torch.Tensor | tuple[float, float], *, name: str, non_negative: bool) -> None:
    """Check one threshold's numerical domain without converting tensors."""
    if isinstance(value, tuple):
        if name != "minimum_patch_lengths" or len(value) != 2 or not all(isinstance(item, float) for item in value):
            raise TypeError(f"{name} must be a pair of floats or a floating-point tensor")
        for item in value:
            _validate_numeric(item, name=name, non_negative=non_negative)
        return
    if isinstance(value, float):
        finite = math.isfinite(value)
        negative = value < 0.0
    elif isinstance(value, torch.Tensor):
        if value.dtype not in (torch.float32, torch.float64) or value.layout != torch.strided:
            raise TypeError(f"{name} must be a dense float32 or float64 tensor")
        finite = bool(torch.isfinite(value).all())
        negative = bool(torch.any(value < 0.0))
    else:
        raise TypeError(f"{name} must contain floats or a floating-point tensor")
    if not finite:
        raise ValueError(f"{name} must be finite")
    if non_negative and negative:
        raise ValueError(f"{name} must be non-negative")


def validate_thresholds(
    *,
    distance_tolerance: float | torch.Tensor | None = None,
    minimum_face_crossing: float | torch.Tensor | None = None,
    inset: float | torch.Tensor | None = None,
    minimum_patch_lengths: tuple[float, float] | torch.Tensor | None = None,
    minimum_distance: float | torch.Tensor | None = None,
    maximum_distance: float | torch.Tensor | None = None,
) -> None:
    """Check supplied numerical thresholds without changing them.

    Omitted arguments are ignored. Signed distance limits may be negative;
    all other thresholds must be non-negative. When both distance limits are
    supplied, they must broadcast together and satisfy minimum <= maximum.
    Geometry calls separately check alignment with the input AABBs. This
    explicit value scan may synchronize an accelerator.

    Args:
        distance_tolerance (float | torch.Tensor | None): Absolute face-distance tolerance.
        minimum_face_crossing (float | torch.Tensor | None): Minimum actual tangential overlap length.
        inset (float | torch.Tensor | None): Inward query-bound offset.
        minimum_patch_lengths (tuple[float, float] | torch.Tensor | None): Two tangential length limits.
        minimum_distance (float | torch.Tensor | None): Inclusive signed lower limit.
        maximum_distance (float | torch.Tensor | None): Inclusive signed upper limit.
    """
    for name, value in (
        ("distance_tolerance", distance_tolerance),
        ("minimum_face_crossing", minimum_face_crossing),
        ("inset", inset),
        ("minimum_patch_lengths", minimum_patch_lengths),
        ("minimum_distance", minimum_distance),
        ("maximum_distance", maximum_distance),
    ):
        if value is not None:
            if name == "minimum_patch_lengths" and not isinstance(value, (tuple, torch.Tensor)):
                raise TypeError("minimum_patch_lengths must be a pair of floats or a query-aligned tensor")
            if name == "minimum_patch_lengths" and isinstance(value, torch.Tensor):
                if value.ndim < 2 or value.shape[-1] != 2:
                    raise ValueError("minimum_patch_lengths must have shape (*batch, query, 2)")
            _validate_numeric(value, name=name, non_negative=name not in ("minimum_distance", "maximum_distance"))
    if minimum_distance is not None and maximum_distance is not None:
        if isinstance(minimum_distance, torch.Tensor) and isinstance(maximum_distance, torch.Tensor):
            if minimum_distance.device != maximum_distance.device:
                raise ValueError("distance limits must be on the same device")
        try:
            inverted = minimum_distance > maximum_distance
        except RuntimeError as error:
            raise ValueError("distance limits must be broadcastable") from error
        any_inverted = bool(torch.any(inverted)) if isinstance(inverted, torch.Tensor) else inverted
        if any_inverted:
            raise ValueError("minimum_distance must be less than or equal to maximum_distance")
