"""Shared Plotly drawing for the runnable examples; no relationship calculations."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

COLORS = ("#168aad", "#e58b25", "#319b75", "#9b66be")
DISPLAY_CONFIG = {
    "displaylogo": False,
    "showSendToCloud": False,
    "modeBarButtonsToRemove": ["toImage"],
    "responsive": True,
}


def show_figure(figure: "Figure") -> None:
    """Open a local browser tab, without export buttons or generated files."""
    figure.show(renderer="browser", config=DISPLAY_CONFIG)


def style_figure(figure: "Figure", title: str, subtitle: str) -> None:
    """Apply a shared, responsive presentation style."""
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=35, r=35, t=115, b=110),
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.5, xanchor="center", font=dict(size=24)),
        font=dict(family="Arial, sans-serif", size=13, color="#243247"),
        hoverlabel=dict(bgcolor="white", font_size=13),
        paper_bgcolor="white",
    )


def scene_layout(lower: np.ndarray, upper: np.ndarray) -> dict:
    """Use equal spatial scaling and one initial camera for all 3D panels."""
    axes = {}
    for index, axis in enumerate("xyz"):
        axes[f"{axis}axis"] = dict(
            title=axis.upper(),
            range=[float(lower[index]), float(upper[index])],
            backgroundcolor="#f8fafc",
            gridcolor="#dce3eb",
            showbackground=True,
            zeroline=False,
            showspikes=False,
        )
    return dict(
        **axes,
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=-2.0, z=1.25), projection=dict(type="orthographic")),
        dragmode="orbit",
    )


def draw_cuboid_wireframe(
    figure: "Figure",
    cuboid: np.ndarray,
    *,
    label: str,
    scene: str = "scene",
    color: str = "#526479",
    linewidth: float = 3,
    legend: str = "legend",
    showlegend: bool = False,
) -> np.ndarray:
    """Draw one hoverable cuboid and return its corners, lower face first."""
    import plotly.graph_objects as go

    xmin, ymin, zmin, xmax, ymax, zmax = cuboid
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    coordinates = [[], [], []]
    for start, end in edges:
        for axis in range(3):
            coordinates[axis].extend([float(corners[start, axis]), float(corners[end, axis]), None])
    bounds_text = ", ".join(f"{value:g}" for value in cuboid)
    figure.add_trace(
        go.Scatter3d(
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            mode="lines",
            scene=scene,
            line=dict(color=color, width=linewidth),
            name=label,
            legend=legend,
            showlegend=showlegend,
            meta=dict(kind="cuboid", label=label, bounds=cuboid.tolist()),
            hovertemplate=f"<b>{label}</b><br>Bounds: [{bounds_text}]<extra></extra>",
        )
    )
    return corners


def draw_rectangle(
    figure: "Figure",
    bounds: np.ndarray,
    height: float,
    *,
    color: str,
    name: str,
    meta: dict,
    scene: str = "scene",
    legend: str = "legend",
    showlegend: bool = True,
) -> None:
    """Draw an explicitly triangulated horizontal rectangle with geometry on hover."""
    import plotly.graph_objects as go

    xmin, ymin, xmax, ymax = bounds
    bounds_text = ", ".join(f"{value:g}" for value in bounds)
    details = f"<b>{name}</b><br>XY bounds: [{bounds_text}]<br>Area: {meta['area']:g}"
    if meta["kind"] == "contact":
        details += f"<br>Z: {height:g}"
    else:
        details += "<br>Projected overlap, not contact"
    figure.add_trace(
        go.Mesh3d(
            x=[xmin, xmax, xmax, xmin],
            y=[ymin, ymin, ymax, ymax],
            z=[height] * 4,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=color,
            opacity=0.85,
            lighting=dict(ambient=1, diffuse=0),
            flatshading=True,
            scene=scene,
            name=name,
            legend=legend,
            showlegend=showlegend,
            meta=meta,
            hovertemplate=details + "<extra></extra>",
        )
    )


def draw_scene(
    figure: "Figure",
    cuboids: np.ndarray,
    pairs: np.ndarray,
    bounds: np.ndarray,
    heights: np.ndarray,
    areas: np.ndarray,
    *,
    scene: str = "scene",
    legend: str = "legend",
    limits: tuple[np.ndarray, np.ndarray] | None = None,
    right_labels: tuple[int, ...] = (),
) -> None:
    """Add cuboid outlines, labels and precomputed contacts to one Plotly scene."""
    import plotly.graph_objects as go

    label_positions = []
    for index, cuboid in enumerate(cuboids):
        draw_cuboid_wireframe(figure, cuboid, label=f"Cuboid {index}", scene=scene)
        xmin, ymin, zmin, xmax, ymax, zmax = cuboid
        label_positions.append(
            [xmax + 0.25, ymax + 0.12, (zmin + zmax) / 2]
            if index in right_labels
            else [xmin + 0.4, ymin - 0.12, (zmin + zmax) / 2]
        )
    positions = np.array(label_positions)
    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="text",
            scene=scene,
            text=[f"Cuboid {index}" for index in range(len(cuboids))],
            textfont=dict(size=12, color="#243247"),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="labels"),
        )
    )
    for index, ((query, reference), rectangle, height, area) in enumerate(
        zip(pairs, bounds, heights, areas, strict=True)
    ):
        draw_rectangle(
            figure,
            rectangle,
            float(height),
            color=COLORS[index % len(COLORS)],
            name=f"Cuboid {query} / {reference} · area {area:g}",
            scene=scene,
            legend=legend,
            meta=dict(
                kind="contact",
                query=int(query),
                reference=int(reference),
                bounds=rectangle.tolist(),
                height=float(height),
                area=float(area),
            ),
        )
    lower, upper = (
        limits if limits is not None else (cuboids[:, :3].min(axis=0) - 0.6, cuboids[:, 3:].max(axis=0) + 0.6)
    )
    figure.update_layout(**{scene: scene_layout(lower, upper)})
