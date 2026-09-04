"""Check advertised demo geometry and construct figures without a browser."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("numpy")
pytest.importorskip("plotly")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(filename: str):
    """Load a numbered script without executing its desktop entrypoint."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(filename[:-3], EXAMPLES / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_stacked_contact_demo() -> None:
    """The README stack contains four exact downward contacts."""
    example = load_example("01_stacked_contact_patches.py")
    scene = example.build_scene()
    contacts = example.find_contacts(scene)
    assert contacts[0].tolist() == [[1, 0], [2, 1], [3, 2], [4, 3]]
    assert contacts[3].tolist() == [9.0, 6.25, 4.0, 3.0]
    figure = example.plot_scene(scene, *contacts)
    assert figure.data


def test_three_scene_demo_has_no_artificial_world_axis() -> None:
    """One batched calculation preserves each scene's independent geometry."""
    example = load_example("02_three_scene_contact_patches.py")
    scenes = example.build_scenes()
    assert scenes.shape == (3, 5, 6)
    contacts = example.find_contacts(scenes)
    assert [result[0].tolist() for result in contacts] == [
        [[1, 0], [2, 1], [3, 2], [4, 3]],
        [[2, 0], [2, 1], [3, 2], [4, 2]],
        [[1, 0], [2, 1], [4, 3]],
    ]
    assert [result[3].tolist() for result in contacts] == [[9, 6.25, 4, 3], [4.5, 4.5, 2.5, 2.5], [3, 3, 3]]
    figure = example.plot_scenes(scenes, contacts)
    assert figure.layout.scene3


def test_projection_demo_is_separated_in_three_dimensions() -> None:
    """Projected intersections do not imply physical contact."""
    import aabb_primitives as aabb

    example = load_example("03_projected_intersection_bounds.py")
    queries, references = example.build_scene()
    result = example.find_intersections(queries, references)
    assert result[0].tolist() == [[0, 0], [0, 1]]
    torch.testing.assert_close(result[1], torch.tensor([[0, 0.5, 1.5, 2], [2.5, 2, 4, 4]], dtype=torch.float64))
    assert result[2].tolist() == [2.25, 3.0]
    assert not aabb.contact_mask(queries, references, aabb.AABBFace.Z_MIN, distance_tolerance=0.0).any()
    figure = example.plot_scene(queries, references, *result)
    assert figure.data
