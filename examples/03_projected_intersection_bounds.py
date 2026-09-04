"""Show how separated cuboids can have intersecting XY projections.

Run from the repository root:
    uv run --extra examples python examples/03_projected_intersection_bounds.py

The coloured rectangles are projected intersections, not physical contact patches.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
from plotting import draw_cuboid_wireframe, draw_rectangle, scene_layout, show_figure, style_figure

import aabb_primitives as aabb
from aabb_primitives import AABBFace

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def build_scene() -> tuple[torch.Tensor, torch.Tensor]:
    """Return float query (1, 6) and reference (3, 6) cuboids with a vertical gap."""
    queries = torch.tensor([[0, 0, 3, 4, 4, 4]], dtype=torch.float64)
    references = torch.tensor(
        [[-1, 0.5, 0, 1.5, 2, 1], [2.5, 2, 0, 5, 4.5, 1], [5, 0.5, 0, 6, 2, 1]], dtype=torch.float64
    )
    return queries, references


def find_intersections(
    queries: torch.Tensor, references: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query/reference pairs (N, 2), XY bounds (N, 4), and areas (N,)."""
    aabb.validate_aabbs(queries)
    aabb.validate_aabbs(references)
    face = AABBFace.Z_MIN  # A Z face selects the XY plane; it does not require contact.

    # Outputs are query-first: bounds[q, r] describes query q and reference r.
    bounds = aabb.projected_intersection_bounds(queries, references, face)  # (Q, R, 4)
    mask = aabb.projected_overlap_mask(queries, references, face)  # (Q, R)
    areas = aabb.projected_overlap_areas(queries, references, face)  # (Q, R)

    # Raw bounds can be inverted for disjoint projections. Draw only positive-area intersections.
    return torch.argwhere(mask), bounds[mask], areas[mask]


def print_intersections(
    references: torch.Tensor, pairs: torch.Tensor, bounds: torch.Tensor, areas: torch.Tensor
) -> None:
    """Report the single query's projected intersections, including excluded references."""
    print("XY projected intersections (query/reference IDs belong to separate input arrays)")
    print("query  reference  [xmin, ymin, xmax, ymax]       area")
    for (query, reference), rectangle, area in zip(pairs, bounds, areas, strict=True):
        coordinates = ", ".join(f"{coordinate:g}" for coordinate in rectangle)
        print(f"{query:5d}  {reference:9d}  [{coordinates}]  area={area:g}")
    for reference in torch.arange(len(references))[~torch.isin(torch.arange(len(references)), pairs[:, 1])]:
        print(f"Query 0 / Reference {reference}: no projected overlap")
    print("In this scene every reference has a 2-unit vertical gap from the query: no 3D intersection or contact.")


def plot_scene(
    queries: torch.Tensor, references: torch.Tensor, pairs: torch.Tensor, bounds: torch.Tensor, areas: torch.Tensor
) -> "Figure":
    """Build a 3D scene and matching XY view; the display plane has no geometric role."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Plotly helpers receive NumPy only after all geometric filtering.
    queries, references, pairs, bounds, areas = (
        value.detach().cpu().numpy() for value in (queries, references, pairs, bounds, areas)
    )
    query_color = "#172554"
    reference_colors = ("#008577", "#c66b08", "#8755b5")
    plane_z = -0.6  # Display only: not a contact surface or ground.
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        horizontal_spacing=0.09,
        subplot_titles=["1. Separate in 3D: a 2-unit gap", "2. Intersect the XY footprints"],
    )
    cuboids = np.concatenate((queries, references))
    lower = cuboids[:, :3].min(axis=0) - 0.6
    upper = cuboids[:, 3:].max(axis=0) + 0.6
    figure.add_trace(
        go.Mesh3d(
            x=[lower[0], upper[0], upper[0], lower[0]],
            y=[lower[1], lower[1], upper[1], upper[1]],
            z=[plane_z] * 4,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#e2e8f0",
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="display_plane"),
        ),
        row=1,
        col=1,
    )
    labels = ["Query cuboid 0", *[f"Reference cuboid {index}" for index in range(len(references))]]
    colors = [query_color, *reference_colors]
    for index, (cuboid, color, label) in enumerate(zip(cuboids, colors, labels, strict=True)):
        corners = draw_cuboid_wireframe(
            figure, cuboid, label=label, color=color, linewidth=5 if index == 0 else 3, showlegend=True
        )
        xmin, ymin, zmin, xmax, ymax, zmax = cuboid
        guides = [[], [], []]
        for x, y, z in corners[:4]:
            guides[0].extend([x, x, None])
            guides[1].extend([y, y, None])
            guides[2].extend([plane_z, z, None])
        figure.add_trace(
            go.Scatter3d(
                x=guides[0],
                y=guides[1],
                z=guides[2],
                mode="lines",
                line=dict(color=color, dash="dash", width=2),
                opacity=0.25,
                hoverinfo="skip",
                showlegend=False,
                meta=dict(kind="guides"),
            )
        )
        footprint = corners[[0, 1, 2, 3, 0]]
        figure.add_trace(
            go.Scatter3d(
                x=footprint[:, 0],
                y=footprint[:, 1],
                z=[plane_z] * 5,
                mode="lines",
                line=dict(color=color, width=2),
                hoverinfo="skip",
                showlegend=False,
                meta=dict(kind="footprint"),
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[(xmin + xmax) / 2],
                y=[(ymin + ymax) / 2],
                z=[zmax + 0.2],
                mode="text",
                text=[label.replace("cuboid ", "")],
                textfont=dict(color=color, size=11),
                hoverinfo="skip",
                showlegend=False,
                meta=dict(kind="labels"),
            )
        )
        bounds_text = ", ".join(f"{value:g}" for value in cuboid)
        figure.add_trace(
            go.Scatter(
                x=footprint[:, 0],
                y=footprint[:, 1],
                mode="lines",
                line=dict(color=color, width=3 if index == 0 else 2),
                name=label,
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>3D bounds: [{bounds_text}]<extra></extra>",
                meta=dict(kind="footprint", label=label),
            ),
            row=1,
            col=2,
        )

    for (query, reference), rectangle, area in zip(pairs, bounds, areas, strict=True):
        xmin, ymin, xmax, ymax = rectangle
        color = reference_colors[reference]
        name = f"Query {query} / Reference {reference}"
        metadata = dict(
            kind="projected_intersection",
            query=int(query),
            reference=int(reference),
            bounds=rectangle.tolist(),
            area=float(area),
        )
        draw_rectangle(figure, rectangle, plane_z, color=color, name=name, meta=metadata, showlegend=False)
        bounds_text = ", ".join(f"{value:g}" for value in rectangle)
        figure.add_trace(
            go.Scatter(
                x=[xmin, xmax, xmax, xmin, xmin],
                y=[ymin, ymin, ymax, ymax, ymin],
                mode="lines",
                fill="toself",
                line=dict(color=color, width=2),
                fillcolor="rgba(0,133,119,0.22)" if reference == 0 else "rgba(198,107,8,0.22)",
                hoveron="fills",
                name=name,
                showlegend=False,
                meta=metadata,
                text=f"<b>{name}</b><br>XY bounds: [{bounds_text}]<br>Area: {area:g}<br>Not contact",
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )
        figure.add_annotation(
            x=(xmin + xmax) / 2,
            y=(ymin + ymax) / 2,
            text=f"Q{query} ∩ R{reference}<br>Area = {area:g}",
            showarrow=False,
            font=dict(size=14),
            row=1,
            col=2,
        )
    for x, y, text, color in [
        (0.1, 3.8, "Query 0 footprint", query_color),
        (-1, 2.2, "Reference 0", reference_colors[0]),
        (3.8, 4.7, "Reference 1", reference_colors[1]),
        (5.5, 1.25, "Reference 2<br>No XY<br>overlap", reference_colors[2]),
    ]:
        figure.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=11, color=color),
            xanchor="left" if text in ("Query 0 footprint", "Reference 0") else "center",
            row=1,
            col=2,
        )
    lower[2] = plane_z - 0.2
    figure.update_layout(scene=scene_layout(lower, upper))
    figure.update_xaxes(range=[lower[0], upper[0]], title_text="X", zeroline=False, row=1, col=2)
    figure.update_yaxes(
        range=[lower[1], upper[1]], title_text="Y", scaleanchor="x", scaleratio=1, zeroline=False, row=1, col=2
    )
    style_figure(
        figure,
        "Projected intersection does not mean contact",
        "One query · three references · two XY intersections · no 3D intersections",
    )
    figure.update_layout(
        margin=dict(l=25, r=35, t=130, b=130),
        legend=dict(orientation="h", x=0.5, y=-0.17, xanchor="center", itemclick=False, itemdoubleclick=False),
    )
    figure.add_annotation(
        text="XY projection plane — not a contact surface",
        x=0.22,
        y=-0.07,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#64748b"),
    )
    figure.add_annotation(
        text="projected_intersection_bounds(Z_MIN) → filter with projected_overlap_mask(Z_MIN)",
        x=0.5,
        y=-0.28,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#64748b"),
    )
    return figure


def main() -> None:
    queries, references = build_scene()
    intersections = find_intersections(queries, references)
    print_intersections(references, *intersections)
    show_figure(plot_scene(queries, references, *intersections))


if __name__ == "__main__":
    main()
