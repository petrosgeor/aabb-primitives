# Reference

This page describes the details that are easy to miss when using
`aabb-primitives`. For the short introduction and installation instructions,
see the [README](../README.md).

## Inputs and pairwise outputs

Every AABB is one row of six coordinates:

```text
[xmin, ymin, zmin, xmax, ymax, zmax]
```

Inputs are dense, strided Torch tensors with `float32` or `float64` dtype and
shape `(*batch, rows, 6)`. The last two dimensions always mean rows and
coordinates. Query and reference tensors must have the same dtype and device.
The leading dimensions are broadcast with the usual Torch rules. The output
then has the resolved batch dimensions followed by a query row and a reference
row:

```text
queries     (*batch, Q, 6)
references  (*batch, R, 6)
pairwise    (*batch, Q, R)
```

`pairwise[..., q, r]` describes query row `q` relative to reference row `r`.
Passing one tensor twice compares every ordered pair, including the diagonal.
The functions do not remove self-pairs.
For boxes with positive extents and zero distance tolerance, opposing faces of
the same box do not contact. Zero extents or a larger tolerance can change that
result.

`validate_aabbs` checks that coordinates are finite and that every minimum is
less than or equal to its corresponding maximum. Empty row dimensions and
zero extents are valid. A zero extent lets the same representation describe an
axis-aligned rectangle, line segment, or point. A diagonal segment cannot be
represented by this format. The geometry functions themselves check tensor
metadata, but do not scan coordinate values; call the validator at an input
boundary when the data is externally supplied. Empty collections preserve their
natural pairwise shapes, such as `(0, 6)` against `(R, 6)` producing `(0, R)`.

For pairwise functions, the common output shapes are:

| Function group | Output shape after batch dimensions |
| --- | --- |
| `signed_distances` | `(Q, R)` |
| `signed_distances_all_faces` | `(Q, R, 6)` |
| `intersection_bounds` | `(Q, R, 2)` |
| `intersection_bounds_all_axes` | `(Q, R, 3, 2)` |
| `overlap_lengths` | `(Q, R)` |
| `overlap_lengths_all_axes` | `(Q, R, 3)` |
| `tangential_overlap_lengths` | `(Q, R, 2)` |
| `projected_intersection_bounds` | `(Q, R, 4)` |
| `projected_overlap_areas` | `(Q, R)` |
| `axis_overlap`, `projected_overlap_mask`, `contained_by_mask`, `inward_projected_overlap_mask`, `within_distance`, `contact_mask` | `(Q, R)` |
| `axis_overlap_all_axes`, `inward_axis_overlap_all_axes` | `(Q, R, 3)` |
| `query_face_contact_patches` | patches `(Q, R, 6, 6)`, mask `(Q, R, 6)` |

`query_face_areas` only takes query boxes, so its output is `(*query_batch, Q)`.

### Leading-dimension broadcasting

There is no reserved batch rank. Any number of leading dimensions can describe
scenes, environments, alternatives, or another grouping chosen by the caller.
Matching axes are compared together; a singleton axis is expanded; two
different non-singleton sizes are an error.

| Query shape | Reference shape | Pairwise mask shape | Use |
| --- | --- | --- | --- |
| `(Q, 6)` | `(R, 6)` | `(Q, R)` | One query collection against one reference collection |
| `(S, Q, 6)` | `(S, R, 6)` | `(S, Q, R)` | `S` aligned scenes |
| `(S, Q, 6)` | `(R, 6)` | `(S, Q, R)` | One reference collection shared by every scene |
| `(B, K, Q, 6)` | `(B, 1, R, 6)` | `(B, K, Q, R)` | References shared by `K` alternatives in each environment |
| `(B, 1, Q, 6)` | `(1, K, R, 6)` | `(B, K, Q, R)` | Cartesian combinations of two batch axes |

The last row compares each of the `B` query collections with each of the `K`
reference collections. Broadcasting expands the view used by
the calculation; it does not duplicate the shared input collection in memory.
The output still contains every pair:

```python
import torch
import aabb_primitives as aabb

queries = torch.zeros((2, 1, 1, 6), dtype=torch.float64)       # B=2, Q=1
references = torch.zeros((1, 3, 1, 6), dtype=torch.float64)   # K=3, R=1

overlaps = aabb.overlap_lengths_all_axes(queries, references)
assert overlaps.shape == (2, 3, 1, 1, 3)
```

## Faces and signed distance

The face enum is ordered as follows:

```text
X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX
```

Signed distance and contact functions compare the selected query face
with the opposite face of each reference box. For example, `Z_MIN` compares
a query's `zmin` with each reference's `zmax`; `Z_MAX` compares a reference's
`zmin` with the query's `zmax`. Projection functions use the face to select its
two tangential axes. `query_face_areas` measures only the selected query face.

`signed_distances` returns a positive value for a gap, zero when the two faces
are aligned, and a negative value when the selected faces have crossed. The
sign is tied to the query face and is useful for filtering by a directed
distance. A negative value is not, by itself, the depth of the three-dimensional
intersection.

`signed_distances_all_faces` returns the six values in the face order above.

## Bounds and overlap

`intersection_bounds` and `intersection_bounds_all_axes` return the raw
coordinate bounds formed from coordinate-wise maxima of the minima and minima
of the maxima. They do not discard separate intervals. For disjoint boxes the
returned interval is inverted, with its minimum greater than its maximum:

```python
import torch
import aabb_primitives as aabb

query = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
reference = torch.tensor([[2.0, 0.0, 0.0, 3.0, 1.0, 1.0]], dtype=torch.float64)

raw = aabb.intersection_bounds(query, reference, aabb.AABBAxis.X)
assert raw.tolist() == [[[2.0, 1.0]]]
```

`overlap_lengths` and `overlap_lengths_all_axes` clamp those inverted lengths
to zero. `axis_overlap` requires a strictly positive length on its selected
axis; `projected_overlap_mask` requires it on both tangential axes. A zero
overlap length fails the check on that axis. Boxes touching
along a full face can still have positive projected overlap on the face's two
tangential axes; their normal signed distance is zero, so they can satisfy
`contact_mask`. `minimum_face_crossing` adds an inclusive lower bound on the
actual overlap length, but a zero-length overlap remains false when the
threshold is zero.

The related functions are:

- `axis_overlap` checks one axis; `axis_overlap_all_axes` returns separate
  masks for X, Y, and Z.
- `tangential_overlap_lengths` returns the two lengths in the selected face's
  tangential-axis order.
- `projected_intersection_bounds` returns
  `[first_min, second_min, first_max, second_max]` in that same order. The
  axes are Y/Z for an X face, X/Z for a Y face, and X/Y for a Z face.
- `projected_overlap_mask` tests strict overlap on both tangential axes.
- `projected_overlap_areas` multiplies the two non-negative tangential
  lengths. It measures overlap in the selected projection plane; it does not
  require the boxes to meet along the normal axis.
- `query_face_areas` measures the full area of the selected query face.

This distinction matters when two boxes have overlapping XY footprints but are
separated in Z. Their `Z_MIN` projected area can be positive while
`contact_mask` is false.

## Containment and inward bounds

`contained_by_mask` tests whether every minimum and maximum of the query lies
inside the corresponding reference interval. Equality is included. The query
may be a point, a line segment, or a lower-dimensional rectangle, and a box is
considered contained by itself.

`inward_axis_overlap_all_axes` and `inward_projected_overlap_mask` use the
query bounds moved inward by `inset`. On each selected axis, the reference
bounds must satisfy:

```text
reference_max > query_min + inset
reference_min < query_max - inset
```

The all-axis function returns one mask for each of X, Y, and Z. The projected
function selects the two tangential axes for its face. These predicates answer
whether a reference crosses the inset query bounds; they are not containment
tests and are not interchangeable with a minimum overlap threshold.

## Distance windows and contact

`within_distance` combines two conditions:

1. The reference has positive projected overlap on the two tangential axes,
   subject to `minimum_face_crossing`.
2. The signed face distance lies between `minimum_distance` and
   `maximum_distance`, inclusive.

The distance window can include a gap, exact alignment, or a controlled
crossing. Signed distance limits may be negative. Use `contact_mask` when the
normal condition should be symmetric around alignment.

`contact_mask` is true only when all of these conditions hold:

- both tangential overlap lengths are positive;
- both lengths meet `minimum_face_crossing`, if supplied;
- the absolute signed distance is at most `distance_tolerance`;
- both tangential lengths meet the two values in `minimum_patch_lengths`, if
  supplied.

All comparisons are inclusive at their supplied limits, except that an overlap
length must still be strictly positive. The two values in
`minimum_patch_lengths` follow the selected face's tangential-axis order. For
example, a Z-face patch uses `(X length, Y length)`:

```python
import torch
import aabb_primitives as aabb

query = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
references = torch.tensor(
    [
        [0.0, 0.0, 1.125, 2.0, 2.0, 2.0],  # 0.125-unit gap; a 2 by 2 patch
        [0.0, 0.0, 1.0, 2.0, 1.5, 2.0],  # aligned; a 2 by 1.5 patch
    ],
    dtype=torch.float64,
)

mask = aabb.contact_mask(
    query,
    references,
    aabb.AABBFace.Z_MAX,
    distance_tolerance=0.125,
    minimum_patch_lengths=(1.0, 1.9),
)
assert mask.tolist() == [[True, False]]
```

Line segments and points are valid input geometry, but contact masks require a
positive-area patch on the selected face. A degenerate tangential extent
therefore cannot satisfy `contact_mask` or a face-contact mask on that face.

## All-face contact patches

`query_face_contact_patches` checks all six query faces at once. It returns:

```text
patches     (*batch, Q, R, 6, 6)
face_mask   (*batch, Q, R, 6)
```

The first `6` in the patch shape is the query face, in
`X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX` order. The final `6` is an AABB row
in `[xmin, ymin, zmin, xmax, ymax, zmax]` order. Each qualifying patch keeps
the query face coordinate along its normal axis and uses the raw intersection
limits on the other two axes. This places the patch on the query face even
when the distance tolerance accepts a small gap or penetration.

Use `face_mask` to select valid patches. Invalid entries contain six zeros. The
function accepts `distance_tolerance` and `minimum_face_crossing`; use
`contact_mask` when you also need separate minimum lengths for the two
tangential axes.

The following example shows the output layout and the tolerance plane:

```python
import torch
import aabb_primitives as aabb

query = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
references = torch.tensor(
    [
        [0.0, 0.0, 1.125, 2.0, 2.0, 2.0],  # accepted through the 0.125 gap
        [0.0, 0.0, 3.0, 2.0, 2.0, 4.0],  # no face within tolerance
    ],
    dtype=torch.float64,
)

patches, face_mask = aabb.query_face_contact_patches(
    query, references, distance_tolerance=0.125
)
assert patches.shape == (1, 2, 6, 6)
assert face_mask.shape == (1, 2, 6)
assert face_mask.tolist() == [[[False, False, False, False, False, True],
                               [False, False, False, False, False, False]]]

# Z_MAX is face index 5. Its patch lies on query zmax = 1.0, not at the
# reference's zmin = 1.125.
assert patches[0, 0, 5].tolist() == [0.0, 0.0, 1.0, 2.0, 2.0, 1.0]
assert torch.equal(patches[0, 1], torch.zeros((6, 6), dtype=torch.float64))
```

## Thresholds and validation

The public validators are explicit:

```python
import torch
import aabb_primitives as aabb

boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
aabb.validate_aabbs(boxes)
aabb.validate_thresholds(
    distance_tolerance=0.05,
    minimum_face_crossing=0.1,
    inset=0.2,
    minimum_patch_lengths=(0.5, 0.5),
    minimum_distance=-0.1,
    maximum_distance=0.2,
)
```

`validate_aabbs` checks tensor metadata, finite values, and non-negative
extents. `validate_thresholds` accepts the following keyword arguments:
`distance_tolerance`, `minimum_face_crossing`, `inset`,
`minimum_patch_lengths`, `minimum_distance`, and `maximum_distance`. Omitted
arguments are ignored. All supplied values must be finite. Every threshold
except the two signed distance limits must be non-negative. If both distance
limits are present, they must broadcast together and satisfy
`minimum_distance <= maximum_distance` element by element.

When passed to a geometry function, scalar-valued thresholds such as
`distance_tolerance`, `minimum_face_crossing`, `inset`, `minimum_distance`,
and `maximum_distance` can be Python floats or zero-dimensional floating
tensors. Query-dependent thresholds use these shapes:

| Threshold | Accepted tensor shape |
| --- | --- |
| `distance_tolerance`, `minimum_face_crossing`, `inset` | `(*threshold_batch, Q)` |
| `minimum_distance`, `maximum_distance` | `(*threshold_batch, Q)` |
| `minimum_patch_lengths` | `(*threshold_batch, Q, 2)` |

`minimum_patch_lengths` can also be a pair of Python floats. Threshold tensors
must be dense, strided `float32` or `float64` tensors on the same device as the
AABBs. They may use a different supported floating dtype from the boxes. The
trailing query dimension must match `Q` exactly. Threshold batch dimensions may
broadcast into the resolved batch shape of the two box inputs, but they may
not introduce a new output batch dimension. For example, a threshold with
shape `(B, 1, Q)` can apply to all of `K` aligned alternatives in an output
with batch shape `(B, K)`.

The validators scan values and may synchronize an accelerator. They do not
convert, copy, round, or repair inputs. Geometry operations also do not perform
automatic dtype conversion or device transfer. Re-run the validators after
mutating data that came from outside the application.

## Visual examples

The example scripts run from the repository root:

```shell
uv run --extra examples python examples/01_stacked_contact_patches.py
uv run --extra examples python examples/02_three_scene_contact_patches.py
uv run --extra examples python examples/03_projected_intersection_bounds.py
```

Each script prints the selected geometry and opens one interactive local
Plotly viewer with labeled boxes and hoverable results. The geometry and
filtering run in Torch; NumPy is used only when data crosses into the plotting
helpers. The examples need a desktop browser, save no output files, and do not
use a Plotly account or hosted service. The viewer is single-use: rerun the
script if it is closed or reloaded. Download and cloud-sharing toolbar buttons
are disabled.

The expected printed geometry is:

| Script | Expected result |
| --- | --- |
| `01_stacked_contact_patches.py` | Query/reference pairs `(1, 0)`, `(2, 1)`, `(3, 2)`, `(4, 3)` with areas `9`, `6.25`, `4`, and `3`. |
| `02_three_scene_contact_patches.py` | Offset stack: four pairs `(1, 0)`, `(2, 1)`, `(3, 2)`, `(4, 3)` with areas `9`, `6.25`, `4`, `3`. Bridge: `(2, 0)`, `(2, 1)`, `(3, 2)`, `(4, 2)` with areas `4.5`, `4.5`, `2.5`, `2.5`. Separate towers: `(1, 0)`, `(2, 1)`, `(4, 3)`, all with area `3`. |
| `03_projected_intersection_bounds.py` | Query/reference pairs `(0, 0)` and `(0, 1)` have projected areas `2.25` and `3`. Every reference is two units below the query, so there are no three-dimensional contacts. |

The pair indices are local row indices within each query and reference
collection. The three-scene example keeps the scene dimension in one batched
calculation; it does not add an artificial world axis.

## Cost and scope

For each resolved batch entry, a dense pairwise operation does work
proportional to `Q * R`. Broadcasting avoids copying shared input collections,
but pairwise output tensors still contain every query/reference pair. The
all-face patch operation stores 36 coordinates per pair in addition to its
boolean mask, so its output can become large.

Functions keep no state and no cache. Calling a function again recalculates
the result. There is no spatial index, sparse candidate search, implicit
rounding, snapping, or automatic self-exclusion. The package reports AABB
geometry and predicates; it does not model stability, forces, packing
feasibility, or ground contact. CPU execution is tested. CUDA and autograd have
not been verified for this release.

## Migrating from 0.1

Version 0.2 is experimental and intentionally breaks the old NumPy API. The
old relation-object call:

```python
# 0.1: NumPy arrays and a relation object
# relations = PairwiseAABBRelations.from_aabbs(
#     query_aabbs=queries,
#     reference_aabbs=references,
# )
# contacts = relations.contact_mask(AABBFace.Z_MAX, distance_tolerance=0.0)
```

becomes direct calls with Torch tensors:

```python
# 0.2: Torch tensors and module-level functions
import torch
import aabb_primitives as aabb

queries = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
references = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 2.0]], dtype=torch.float64)

contacts = aabb.contact_mask(
    queries,
    references,
    aabb.AABBFace.Z_MAX,
    distance_tolerance=0.0,
)
assert contacts.tolist() == [[True]]
```

Convert a NumPy array explicitly with `torch.from_numpy()` when needed. That
conversion shares storage with the NumPy array, so changes to either the NumPy
array or the tensor affect the shared data. Replace constructor-time
validation with explicit `validate_aabbs` and `validate_thresholds` calls.
Remove singleton dimensions that existed only to satisfy the old
`(B, K, Q, 6)` interface; leading dimensions now use ordinary Torch
broadcasting.

There is no compatibility relation class, NumPy backend, or implicit input
snapshot in 0.2.
