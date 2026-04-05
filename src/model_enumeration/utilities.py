"""Utilities module."""

import os
from collections import abc, defaultdict

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import cgexplore as cgx
import matplotlib as mpl
import numpy as np
import stk
import stko
from matplotlib.colors import to_hex

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": to_hex(tstr_cmap(0.05)),
    "2": to_hex(tstr_cmap(0.0)),
    "3": to_hex(tstr_cmap(0.1)),
    "4": to_hex(tstr_cmap(0.2)),
    "5": to_hex(tstr_cmap(0.3)),
    "6": to_hex(tstr_cmap(0.4)),
    "7": to_hex(tstr_cmap(0.5)),
    "8": to_hex(tstr_cmap(0.6)),
    "9": to_hex(tstr_cmap(0.7)),
    "10": to_hex(tstr_cmap(0.8)),
    "11": to_hex(tstr_cmap(0.15)),
    "12": to_hex(tstr_cmap(0.9)),
}
topology_cmap = {
    "2P4": to_hex(tstr_cmap(0.0)),
    "3P6": to_hex(tstr_cmap(0.1)),
    "4P8": to_hex(tstr_cmap(0.2)),
    "4P82": to_hex(tstr_cmap(0.5)),
    "6P12": to_hex(tstr_cmap(0.9)),
    "6P122": to_hex(tstr_cmap(0.3)),
    "8P16": to_hex(tstr_cmap(0.4)),
    "8P162": to_hex(tstr_cmap(0.7)),
    "2P3": to_hex(tstr_cmap(0.05)),
    "4P6": to_hex(tstr_cmap(0.15)),
    "4P62": to_hex(tstr_cmap(0.25)),
    "6P9": to_hex(tstr_cmap(0.35)),
    "8P12": to_hex(tstr_cmap(0.45)),
}


core_bead = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)

core_bead2 = cgx.molecular.CgBead(
    element_string="O",
    bead_class="o",
    bead_type="o",
    coordination=2,
)


arm_bead = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)


binder_bead = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)


tetragonal_bead = cgx.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)

trigonal_bead = cgx.molecular.CgBead(
    element_string="C",
    bead_class="n",
    bead_type="n",
    coordination=3,
)


tetragonal_bead2 = cgx.molecular.CgBead(
    element_string="Cr",
    bead_class="y",
    bead_type="y",
    coordination=4,
)

trigonal_bead2 = cgx.molecular.CgBead(
    element_string="Ge",
    bead_class="x",
    bead_type="x",
    coordination=3,
)


def convert_topo(topo_str: str) -> str:
    """Convert topology to fancy name."""
    return {
        "2P4": r"Tet$^{2}$Di$^{4}$",
        "3P6": r"Tet$^{3}_{3}$Di$^{6}$",
        "4P8": r"Tet$^{4}_{4}$Di$^{8}$",
        "4P82": r"Tet$^{4}_{2}$Di$^{8}$",
        "6P12": r"Tet$^{6}$Di$^{12}$",
        "6P122": r"Tet$^{6}_{2}$Di$^{12}$",
        "8P16": r"Tet$^{8}$Di$^{16}$",
        "8P162": r"Tet$^{8}_{2}$Di$^{16}$",
        "2P3": r"Tri$^{2}$Di$^{3}$",
        "4P6": r"Tri$^{4}$Di$^{6}$",
        "4P62": r"Tri$^{4}_{2}$Di$^{6}$",
        "6P9": r"Tri$^{6}$Di$^{9}$",
        "8P12": r"Tri$^{8}$Di$^{12}$",
    }[topo_str]


def get_binder_vector_angles(
    conformer: cgx.molecular.Conformer,
) -> dict[str, list[float]]:
    """Extract the binder vector angles for each ligand."""
    ligands = stko.molecule_analysis.DecomposeMOC().decompose(
        molecule=conformer.molecule,
        metal_atom_nos=(46,),
    )
    va_dict = defaultdict(list)
    for lig in ligands:
        new_lig = stk.BuildingBlock.init_from_molecule(
            molecule=lig,
            functional_groups=stk.SmartsFunctionalGroupFactory(
                smarts="[Pb][Ba][*]",
                bonders=(0,),
                deleters=(),
                placers=(1,),
            ),
        )

        # Defined as the angle between the Pb-Ba vectors.
        vectors = [
            new_lig.get_centroid(atom_ids=fg.get_bonder_ids())
            - new_lig.get_centroid(atom_ids=fg.get_placer_ids())
            for fg in new_lig.get_functional_groups()
        ]
        binder_vector_angle = np.degrees(
            stko.vector_angle(vector1=vectors[0], vector2=vectors[1])
        )

        central_atom = list(
            next(iter(new_lig.get_functional_groups())).get_atoms()
        )[2].__class__.__name__

        va_dict[central_atom].append(binder_vector_angle)
    return va_dict


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def pore_str() -> str:
    """A unit str."""
    return r"pore size [$\mathrm{\AA}$]"


def isomer_energy() -> float:
    """Get constant."""
    return 0.3


def dihedral_state_threshold() -> float:
    """Get constant."""
    return 5.0


def cage_topology_options(
    study: str,
) -> abc.Sequence[tuple[str, abc.Callable]]:
    """Topology options."""
    match study:
        case "homoleptic_2p4":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", cgx.topologies.CGM4L8),
                ("4P82", cgx.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
                ("6P122", cgx.topologies.M6L122),
                ("8P16", stk.cage.EightPlusSixteen),
                ("8P162", cgx.topologies.M8L162),
            )

        case "shortened_homoleptic_2p4":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", cgx.topologies.CGM4L8),
                ("4P82", cgx.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
            )

        case "repr_6P122":
            topologies = (("6P122", cgx.topologies.M6L122),)

        case "homoleptic_2p3":
            topologies = (
                ("2P3", stk.cage.TwoPlusThree),
                ("4P6", stk.cage.FourPlusSix),
                ("4P62", stk.cage.FourPlusSix2),
                ("6P9", stk.cage.SixPlusNine),
                ("8P12", stk.cage.EightPlusTwelve),
            )

        case "homoleptic_3p4":
            topologies = (("6P8", stk.cage.SixPlusEight),)

        case "homoleptic_2p3_3x":
            topologies = (
                ("2P3", stk.cage.TwoPlusThree),
                ("4P6", stk.cage.FourPlusSix),
                ("4P62", stk.cage.FourPlusSix2),
                ("6P9", stk.cage.SixPlusNine),
                ("8P12", stk.cage.EightPlusTwelve),
            )

        case "homoleptic_2p4_4x":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", cgx.topologies.CGM4L8),
                ("4P82", cgx.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
                ("6P122", cgx.topologies.M6L122),
                ("8P16", stk.cage.EightPlusSixteen),
                ("8P162", cgx.topologies.M8L162),
            )

        case "heteroleptic":
            topologies = (("8P16", stk.cage.EightPlusSixteen),)

        case "li2023":
            topologies = (
                ("4P82", cgx.topologies.M4L82),
                ("6P122", cgx.topologies.M6L122),
                ("8P162", cgx.topologies.M8L162),
            )

        case _:
            msg = f"topology option {study} not defined"
            raise RuntimeError(msg)

    return topologies


def contains_parallels(topology_code: cgx.scram.TopologyCode) -> bool:
    """True if the graph contains "1-loops"."""
    weighted_graph = topology_code.get_weighted_graph()
    num_parallel_edges = len(
        [
            i
            for i in weighted_graph.edges()
            if i == 2  # noqa: PLR2004
        ]
    )

    return num_parallel_edges != 0


def percent_change(value: float, percent: float) -> float:
    """Get a percentage change."""
    return value * (percent / 100)
