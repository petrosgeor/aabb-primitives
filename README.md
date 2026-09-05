# aabb-primitives

`aabb-primitives` is a PyTorch library for pairwise geometry between 3D
axis-aligned bounding boxes (AABBs). It calculates face distances,
intersections, overlap, containment, and contact patches for collections of
boxes with broadcastable batch dimensions.

![Five stacked boxes with four downward contact patches](docs/assets/contact-patches.png)

*Five boxes and the four positive-area contacts between consecutive boxes.*

## Installation

Python 3.10 or newer is required. From a checkout, install the package with:

```shell
python -m pip install .
```

For local development, [uv](https://docs.astral.sh/uv/getting-started/)
installs the development and example dependencies:

```shell
uv sync --group dev --extra examples
```

## Example

The following compares one query box with two reference boxes. The query's
lower Z face touches the first reference and has no XY overlap with the
second.

```python
import torch
import aabb_primitives as aabb

query = torch.tensor([[0.0, 0.0, 1.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
references = torch.tensor(
    [
        [0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        [3.0, 0.0, 0.0, 4.0, 2.0, 1.0],
    ],
    dtype=torch.float64,
)

contacts = aabb.contact_mask(
    query, references, face=aabb.AABBFace.Z_MIN, distance_tolerance=0.0
)
areas = aabb.projected_overlap_areas(query, references, aabb.AABBFace.Z_MIN)

assert contacts.tolist() == [[True, False]]
assert areas[contacts].tolist() == [2.0]
```

`contact_mask` checks the selected face distance and requires positive overlap
along both axes in that face. `projected_overlap_areas` measures the overlap in
the selected face plane; it does not check the distance between the faces.
These are geometric tests; the package does not calculate forces or determine
whether a stack is stable.

## Box representation

Each row contains six coordinates in this order:

```text
[xmin, ymin, zmin, xmax, ymax, zmax]
```

Inputs are dense Torch tensors with shape `(*batch, rows, 6)` and dtype
`float32` or `float64`. Query and reference tensors must use the same dtype
and device. Only CPU execution is currently tested. CUDA and autograd have
not been verified. Operations do not convert inputs or move them between
devices.

Zero extents are valid. A row with one zero extent represents an axis-aligned
rectangle, a row with two zero extents represents an axis-aligned line
segment, and a row with three zero extents represents a point. For example,
this segment runs from `(0, 1, 2)` to `(3, 1, 2)`:

```python
segment = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0, 2.0]], dtype=torch.float64)
aabb.validate_aabbs(segment)
segment_overlap = aabb.overlap_lengths_all_axes(segment, segment)

assert segment_overlap.shape == (1, 1, 3)
assert segment_overlap[0, 0].tolist() == [3.0, 0.0, 0.0]
```

Only axis-aligned segments can be represented by this format; a diagonal
segment needs a different representation. A segment or point can be used with
signed face-distance, intersection, and containment operations, but it cannot
form a positive-area contact patch.

Every pairwise function takes a query collection and a reference collection.
The result at `[..., q, r]` describes query row `q` relative to reference row
`r`. The same tensor can fill both roles to compare every ordered pair; the
diagonal self-pairs remain in the result.

Faces are ordered `X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX`, `Z_MIN`, `Z_MAX`. A
selected query face is compared with the opposing reference face. Its signed
distance is positive for a gap, zero when the faces align, and negative after
they cross. A negative value describes the face separation along the normal
axis, not the full overlap depth.

## Available operations

The public functions are available directly from `aabb_primitives`. Use
`AABBAxis` and `AABBFace` instead of integer axis or face indices.

| Need | Functions | Output shape |
|---|---|---|
| Measure opposing face distances | `signed_distances`, `signed_distances_all_faces` | `(..., Q, R)` or `(..., Q, R, 6)` |
| Get raw interval intersections | `intersection_bounds`, `intersection_bounds_all_axes` | `(..., Q, R, 2)` or `(..., Q, R, 3, 2)` |
| Measure non-negative overlap lengths | `overlap_lengths`, `overlap_lengths_all_axes` | `(..., Q, R)` or `(..., Q, R, 3)` |
| Get projected bounds, lengths, or areas | `projected_intersection_bounds`, `tangential_overlap_lengths`, `projected_overlap_areas` | `(..., Q, R, 4)`, `(..., Q, R, 2)`, or `(..., Q, R)` |
| Measure a query face | `query_face_areas` | `(..., Q)` |
| Test overlap or containment | `axis_overlap`, `axis_overlap_all_axes`, `projected_overlap_mask`, `contained_by_mask` | Boolean pairwise results |
| Test inward overlap or distance ranges | `inward_projected_overlap_mask`, `inward_axis_overlap_all_axes`, `within_distance` | Boolean pairwise results |
| Test contact or return contact patches | `contact_mask`, `query_face_contact_patches` | A mask, or patches plus an all-face mask |

`intersection_bounds` returns raw bounds, so disjoint intervals may be
inverted. Contact requires positive overlap along both tangential axes; an
edge, point, or line contact therefore does not produce a contact patch.

The [reference documentation](docs/reference.md) describes threshold shapes,
validation, all-face patch output, computational cost, and the migration from
version 0.1.

## Batching

The final two input dimensions always mean rows and coordinates. All leading
dimensions follow Torch broadcasting, and the output keeps the resolved batch
dimensions before the query and reference axes.

| Query shape | Reference shape | Pairwise mask shape | Meaning |
|---|---|---|---|
| `(Q, 6)` | `(R, 6)` | `(Q, R)` | One comparison |
| `(S, Q, 6)` | `(S, R, 6)` | `(S, Q, R)` | Aligned scenes |
| `(S, Q, 6)` | `(R, 6)` | `(S, Q, R)` | Shared references |
| `(B, K, Q, 6)` | `(B, 1, R, 6)` | `(B, K, Q, R)` | Shared references within each environment |
| `(B, 1, Q, 6)` | `(1, K, R, 6)` | `(B, K, Q, R)` | Every combination of two batch axes |

Matching batch dimensions align. Singleton dimensions can create combinations,
and incompatible dimensions raise an error. Using `query` and `references`
from the example above, the following compares three candidate boxes with the
same two reference boxes in each of two environments:

```python
candidate_boxes = query.expand(2, 3, 1, 6)  # environments, candidates, queries
packed_boxes = references.expand(2, 2, 6)  # environments, references

shared_contacts = aabb.contact_mask(
    candidate_boxes,
    packed_boxes[:, None, :, :],
    face=aabb.AABBFace.Z_MIN,
    distance_tolerance=0.0,
)
assert shared_contacts.shape == (2, 3, 1, 2)
```

## Examples

The optional examples use Torch for the geometry and Plotly for local
visualisation. Run them from the repository root after installing the
`examples` extra:

```shell
uv run --extra examples python examples/01_stacked_contact_patches.py
uv run --extra examples python examples/02_three_scene_contact_patches.py
uv run --extra examples python examples/03_projected_intersection_bounds.py
```

The first script shows downward contacts in one scene, the second applies the
batched API to three scenes, and the third separates projected XY overlap from
three-dimensional contact. Detailed outputs and interpretation notes are in
the [reference documentation](docs/reference.md).

## Development

The `.python-version` file selects Python 3.12 for local development. From a
checkout:

```shell
uv sync --group dev --extra examples
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

Core dependencies are Torch, Jaxtyping, and Beartype. The `examples` extra
adds NumPy and Plotly. Version 0.2 is experimental and replaces the old 0.1
NumPy API with Torch tensors; see the [migration notes](docs/reference.md).

MIT licensed. See [LICENSE](LICENSE).
