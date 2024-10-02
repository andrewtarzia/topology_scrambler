"""Topology utilities."""

import logging
from collections import abc

import numpy as np
import stk
from rdkit import RDLogger

from .graphs import CGM12L24, UnalignedM1L2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def get_graph_type(stoichiometry: tuple[int, ...], multiplier: int) -> str:
    """Get underlying graph type."""
    if stoichiometry == (2, 1):
        return f"{1*multiplier}P{2*multiplier}"
    return None


def get_underyling_vertices(
    pair: str,
    multi: int,
) -> dict[int, list[stk.Vertex]]:
    """Get the vertex prototypes from stk."""
    underlying_topologies = {
        "lf_ls1": {
            1: UnalignedM1L2._vertex_prototypes,  # noqa: SLF001
            2: stk.cage.M2L4Lantern._vertex_prototypes,  # noqa: SLF001
            3: stk.cage.M3L6._vertex_prototypes,  # noqa: SLF001
        },
        "lf_ls9": {
            1: UnalignedM1L2._vertex_prototypes,  # noqa: SLF001
            2: stk.cage.M2L4Lantern._vertex_prototypes,  # noqa: SLF001
            3: stk.cage.M3L6._vertex_prototypes,  # noqa: SLF001
        },
        "la_st5": {
            1: stk.cage.M3L6._vertex_prototypes,  # noqa: SLF001
            2: stk.cage.M6L12Cube._vertex_prototypes,  # noqa: SLF001
            4: CGM12L24._vertex_prototypes,  # noqa: SLF001
        },
        "la_st52": {
            1: stk.cage.M3L6._vertex_prototypes,  # noqa: SLF001
            2: stk.cage.M6L12Cube._vertex_prototypes,  # noqa: SLF001
            4: CGM12L24._vertex_prototypes,  # noqa: SLF001
        },
    }
    return underlying_topologies[pair][multi]


def vmap_to_str(vertex_map: abc.Sequence[tuple[int, int]]) -> str:
    """Convert vertex map to str."""
    strs = sorted([f"{i[0]}-{i[1]}" for i in vertex_map])
    return "_".join(strs)


def points_on_sphere(
    sphere_radius: float,
    num_points: int,
    angle_rotation: float,
) -> np.ndarray:
    """Get the points on a sphere."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(num_points)
    z = np.linspace(
        1 - 1.0 / num_points,
        1.0 / num_points - 1.0,
        num_points,
    )
    radius = np.sqrt(1 - z * z)
    points = np.zeros((3, num_points))
    points[0, :] = sphere_radius * np.cos(theta) * radius
    points[1, :] = sphere_radius * np.sin(theta) * radius
    points[2, :] = z * sphere_radius

    axis = np.array((1.0, 0.0, 0.0))
    moving_points = points.T

    rot_mat = stk.rotation_matrix_arbitrary_axis(
        angle=np.radians(angle_rotation),
        axis=axis,
    )
    new_points = rot_mat @ moving_points.T
    new_points = new_points.T

    return np.array(new_points, dtype=np.float64)
