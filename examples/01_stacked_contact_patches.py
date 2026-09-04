"""Show the four horizontal contact patches in a five-cuboid stack.

Run from the repository root:
    uv run --extra examples python examples/01_stacked_contact_patches.py

This is geometric contact detection, not a stability or force calculation.
"""

from typing import TYPE_CHECKING

import torch
from plotting import draw_scene, show_figure, style_figure

import aabb_primitives as aabb
from aabb_primitives import AABBFace

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def build_scene() -> torch.Tensor:
    """Return five float cuboids, shape (5, 6), numbered from bottom to top."""
    return torch.tensor(
        [
            [0, 0, 0, 4, 4, 1],
            [0.5, 0.5, 1, 3.5, 3.5, 2],
            [1, 0, 2, 4, 3, 3],
            [0.5, 1, 3, 3, 4, 4],
            [1.5, 1.5, 4, 3.5, 3.5, 5],
        ],
        dtype=torch.float64,
    )


def find_contacts(cuboids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pairs (N, 2), XY bounds (N, 4), heights (N,), and areas (N,).

    Input cuboids have shape (Q, 6). Each pair is (query, reference), with the
    upper cuboid as the query and the lower cuboid as the reference.
    """
    aabb.validate_aabbs(cuboids)
    face = AABBFace.Z_MIN

    # Only downward faces: checking Z_MAX as well would count each interface twice.
    # With positive-height cuboids and exact contact, self-pairs cannot match.
    mask = aabb.contact_mask(cuboids, cuboids, face, distance_tolerance=0.0)
    pairs = torch.argwhere(mask)

    # Raw projected bounds alone do not imply contact: filter using the mask.
    # On a Z face, bounds are [xmin, ymin, xmax, ymax].
    bounds = aabb.projected_intersection_bounds(cuboids, cuboids, face)[mask]
    areas = aabb.projected_overlap_areas(cuboids, cuboids, face)[mask]
    heights = cuboids[pairs[:, 0], 2]
    return pairs, bounds, heights, areas


def print_contacts(pairs: torch.Tensor, bounds: torch.Tensor, heights: torch.Tensor, areas: torch.Tensor) -> None:
    """Print one row per contact, in query-first order."""
    print("Horizontal contacts (query = upper cuboid, reference = lower cuboid)")
    print("query  reference  [xmin, ymin, xmax, ymax]       z     area")
    for (query, reference), patch, z, area in zip(pairs, bounds, heights, areas, strict=True):
        coordinates = ", ".join(f"{coordinate:g}" for coordinate in patch)
        print(f"{query:5d}  {reference:9d}  [{coordinates}]  z={z:g}  area={area:g}")


def plot_scene(
    cuboids: torch.Tensor, pairs: torch.Tensor, bounds: torch.Tensor, heights: torch.Tensor, areas: torch.Tensor
) -> "Figure":
    """Build a rotatable figure with hoverable cuboids and contact rectangles."""
    import plotly.graph_objects as go

    # NumPy is used only at the drawing boundary.
    cuboids, pairs, bounds, heights, areas = (
        value.detach().cpu().numpy() for value in (cuboids, pairs, bounds, heights, areas)
    )
    figure = go.Figure()
    draw_scene(figure, cuboids, pairs, bounds, heights, areas)
    style_figure(
        figure, "Five cuboids, four contact patches", "Downward faces (Z_MIN) · exact contact · no ground plane"
    )
    figure.update_layout(
        scene=dict(domain=dict(x=[0.04, 0.78], y=[0, 1])),
        legend=dict(x=0.8, y=0.8, title=dict(text="Query / reference"), itemclick=False, itemdoubleclick=False),
    )
    figure.add_annotation(
        text="Drag to rotate · hover over an edge or coloured patch for geometry",
        x=0.5,
        y=-0.09,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="#64748b"),
    )
    return figure


def main() -> None:
    cuboids = build_scene()
    contacts = find_contacts(cuboids)
    print_contacts(*contacts)
    show_figure(plot_scene(cuboids, *contacts))


if __name__ == "__main__":
    main()
