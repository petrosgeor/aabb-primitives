"""Compare contact patches in three independent scenes using the batched API.

Run from the repository root:
    uv run --extra examples python examples/02_three_scene_contact_patches.py

These are geometric contacts, not claims about stability, forces, or ground contact.
"""

from typing import TYPE_CHECKING

import torch
from plotting import draw_scene, show_figure, style_figure

import aabb_primitives as aabb
from aabb_primitives import AABBFace

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

SCENE_NAMES = ("Offset stack", "Bridge", "Separate towers")


def build_scenes() -> torch.Tensor:
    """Return float cuboids with shape (3, 5, 6): scenes, cuboids, coordinates."""
    scenes = torch.tensor(
        [
            [
                [0, 0, 0, 4, 4, 1],
                [0.5, 0.5, 1, 3.5, 3.5, 2],
                [1, 0, 2, 4, 3, 3],
                [0.5, 1, 3, 3, 4, 4],
                [1.5, 1.5, 4, 3.5, 3.5, 5],
            ],
            [
                [0, 0, 0, 1.5, 3, 1],
                [2.5, 0, 0, 4, 3, 1],
                [0, 0, 1, 4, 3, 2],
                [0.25, 0.5, 2, 1.5, 2.5, 3],
                [2.5, 0.5, 2, 3.75, 2.5, 3],
            ],
            [
                [0, 0, 0, 1.5, 2, 1],
                [0, 0, 1, 1.5, 2, 2],
                [0, 0, 2, 1.5, 2, 3],
                [2.5, 0, 0, 4, 2, 1],
                [2.5, 0, 1, 4, 2, 2],
            ],
        ],
        dtype=torch.float64,
    )
    return scenes


def find_contacts(scenes: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return one (pairs, XY bounds, heights, areas) tuple per scene.

    This example uses scenes of shape (S, Q, 6). Each result contains arrays
    of shape (N, 2), (N, 4), (N,), and (N,), where N varies between scenes.
    """
    aabb.validate_aabbs(scenes)
    face = AABBFace.Z_MIN
    mask = aabb.contact_mask(scenes, scenes, face, distance_tolerance=0.0)  # (S, Q, R)
    bounds = aabb.projected_intersection_bounds(scenes, scenes, face)  # (S, Q, R, 4)
    areas = aabb.projected_overlap_areas(scenes, scenes, face)  # (S, Q, R)

    contacts = []
    for scene_index in range(scenes.shape[0]):
        # Slice the scene first so pairs retain local, query-first cuboid IDs.
        scene_mask = mask[scene_index]
        pairs = torch.argwhere(scene_mask)
        patch_bounds = bounds[scene_index][scene_mask]
        patch_areas = areas[scene_index][scene_mask]
        heights = scenes[scene_index, pairs[:, 0], 2]
        contacts.append((pairs, patch_bounds, heights, patch_areas))
    return contacts


def print_contacts(contacts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
    """Print local cuboid IDs and patch geometry, grouped by scene."""
    print("Horizontal contacts (query = upper cuboid, reference = lower cuboid)")
    for scene_index, (name, (pairs, bounds, heights, areas)) in enumerate(zip(SCENE_NAMES, contacts, strict=True)):
        print(f"\n{name} | scene {scene_index} | {len(pairs)} contacts")
        print("query  reference  [xmin, ymin, xmax, ymax]       z     area")
        for (query, reference), patch, z, area in zip(pairs, bounds, heights, areas, strict=True):
            coordinates = ", ".join(f"{coordinate:g}" for coordinate in patch)
            print(f"{query:5d}  {reference:9d}  [{coordinates}]  z={z:g}  area={area:g}")


def plot_scenes(
    scenes: torch.Tensor, contacts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> "Figure":
    """Build three aligned panels with shared limits and independent rotation."""
    from plotly.subplots import make_subplots

    # Convert only completed geometry to the drawing representation.
    scenes = scenes.detach().cpu().numpy()
    contacts = [tuple(value.detach().cpu().numpy() for value in result) for result in contacts]
    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        horizontal_spacing=0.045,
        subplot_titles=[
            f"{name}<br>{len(result[0])} contact patches" for name, result in zip(SCENE_NAMES, contacts, strict=True)
        ],
    )
    lower = scenes[..., :3].min(axis=(0, 1)) - 0.6
    upper = scenes[..., 3:].max(axis=(0, 1)) + 0.6
    right_labels = ((), (1, 4), (3, 4))
    for scene_index, scene_contacts in enumerate(contacts):
        scene = "scene" if scene_index == 0 else f"scene{scene_index + 1}"
        legend = "legend" if scene_index == 0 else f"legend{scene_index + 1}"
        draw_scene(
            figure,
            scenes[scene_index],
            *scene_contacts,
            scene=scene,
            legend=legend,
            limits=(lower, upper),
            right_labels=right_labels[scene_index],
        )
        domain = figure.layout[scene].domain.x
        figure.update_layout(
            **{
                legend: dict(
                    x=(domain[0] + domain[1]) / 2,
                    y=-0.04,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=11),
                    itemclick=False,
                    itemdoubleclick=False,
                )
            }
        )
    style_figure(
        figure, "Three scenes, one batched calculation", "3 independent scenes × 5 cuboids · geometric contact only"
    )
    figure.update_layout(margin=dict(l=20, r=20, t=135, b=155))
    return figure


def main() -> None:
    scenes = build_scenes()
    contacts = find_contacts(scenes)
    print_contacts(contacts)
    show_figure(plot_scenes(scenes, contacts))


if __name__ == "__main__":
    main()
