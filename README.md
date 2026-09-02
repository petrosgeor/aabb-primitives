# aabb-primitives

`aabb-primitives` calculates vectorized pairwise geometric relationships
between axis-aligned bounding boxes represented as NumPy arrays.

> [!WARNING]
> The API is under development and may change before a stable release.

## Installation and development

Python 3.10 or newer is required. The development environment is pinned to
Python 3.12 in `.python-version`.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```shell
uv sync --group dev
uv run pytest
```

This installs the package in editable mode along with the development tools.
There is no need to activate the virtual environment when using `uv run`.
UV can download the required Python version if it is not already installed.

Check lint rules and formatting:

```shell
uv run ruff check src tests
uv run ruff format --check src tests
```

Apply formatting with:

```shell
uv run ruff format src tests
```

Ruff uses a 120-character line-width target. The commands above cover only
the library and tests, not presentation artifacts or generated environments.

## Representation

Each AABB is one row in minimum/maximum coordinate order:

```text
[xmin, ymin, zmin, xmax, ymax, zmax]
```

`PairwiseAABBRelations` compares query AABBs with reference AABBs. Pairwise
outputs are query-first: `result[q, r]` describes query `q` relative to
reference `r`.

The unbatched inputs have shapes `(Q, 6)` and `(R, 6)`, producing pairwise
outputs whose leading dimensions are `(Q, R)`.

```python
import numpy as np

from aabb_primitives import AABBAxis, PairwiseAABBRelations

queries = np.array(
    [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]],
    dtype=np.float64,
)
references = np.array(
    [
        [1.0, 0.0, 0.0, 3.0, 2.0, 2.0],
        [3.0, 0.0, 0.0, 4.0, 2.0, 2.0],
    ],
    dtype=np.float64,
)

relations = PairwiseAABBRelations.from_aabbs(
    reference_aabbs=references,
    query_aabbs=queries,
)
x_overlap_lengths = relations.overlap_lengths(AABBAxis.X)
```

## Explicit environments and worlds

The batched inputs have shapes `(B, K, Q, 6)` and `(B, K, R, 6)`, where `B`
is the environment dimension and `K` is the world dimension. Each `(b, k)`
pair is independent, and the query and reference inputs must have identical
`B` and `K` dimensions. Batched pairwise outputs begin with `(B, K, Q, R)`.

```python
import numpy as np

from aabb_primitives import AABBFace, PairwiseAABBRelations

query_world = np.array(
    [[0.0, 0.0, 1.0, 1.0, 1.0, 2.0]],
    dtype=np.float64,
)
reference_world = np.array(
    [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
    dtype=np.float64,
)

queries = np.broadcast_to(query_world, (2, 4, 1, 6)).copy()
references = np.broadcast_to(reference_world, (2, 4, 1, 6)).copy()

relations = PairwiseAABBRelations.from_aabbs(
    reference_aabbs=references,
    query_aabbs=queries,
)
z_min_contacts = relations.contact_mask(
    AABBFace.Z_MIN,
    distance_tolerance=0.0,
)
```
