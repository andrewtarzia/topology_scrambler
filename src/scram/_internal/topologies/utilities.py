"""Script to generate and optimise CG models."""

import logging
from collections import abc

import stk
from rdkit import RDLogger

from .graphs import CGM12L24, UnalignedM1L2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


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
