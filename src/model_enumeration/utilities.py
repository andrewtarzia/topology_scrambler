"""Utilities module."""

import pathlib
from collections import abc, defaultdict

import cgexplore as cgx
import matplotlib as mpl
import numpy as np
import stk
import stko
from rmsd import check_reflections, int_atom, kabsch_rmsd, reorder_hungarian

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": tstr_cmap(0.05),
    "2": tstr_cmap(0.0),
    "3": tstr_cmap(0.1),
    "4": tstr_cmap(0.2),
    "5": tstr_cmap(0.3),
    "6": tstr_cmap(0.4),
    "7": tstr_cmap(0.5),
    "8": tstr_cmap(0.6),
    "9": tstr_cmap(0.7),
    "10": tstr_cmap(0.8),
    "11": tstr_cmap(0.15),
    "12": tstr_cmap(0.9),
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
    }[topo_str]


def create_zone(dmin: float, dmax: float, resolution: int) -> list[float]:
    """Create a higher resolution zone."""
    return list(range(dmin, dmax + 1, resolution))


def rmsd_checker(
    unopt_mol: stk.ConstructedMolecule,
    unopt_name: str,
    unopt_glob: list[pathlib.Path],
) -> bool:
    """Check if an un-optimised molecule has a low RMSD to another one."""
    if len(unopt_glob) == 0:
        return False

    p_coord = unopt_mol.with_centroid(
        np.array((0, 0, 0))
    ).get_position_matrix()

    rmsd_threshold = 1

    for other_mol in unopt_glob:
        if other_mol.name.replace(".mol", "") == unopt_name:
            continue

        p_atoms = np.array(
            [int_atom(i.__class__.__name__) for i in unopt_mol.get_atoms()]
        )

        q_mol = stk.BuildingBlock.init_from_file(str(other_mol))
        q_atoms = np.array(
            [int_atom(i.__class__.__name__) for i in q_mol.get_atoms()]
        )
        q_coord = q_mol.with_centroid(
            np.array((0, 0, 0))
        ).get_position_matrix()

        # Apply reorder and reflections.
        result_rmsd, q_swap, q_reflection, q_review = check_reflections(
            p_atoms,
            q_atoms,
            p_coord,
            q_coord,
            reorder_method=reorder_hungarian,
            rmsd_method=kabsch_rmsd,
        )

        if result_rmsd < rmsd_threshold:
            return True
    return False


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


def max_uniformity_threshold() -> float:
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
