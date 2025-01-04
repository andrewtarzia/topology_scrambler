"""Utilities module."""

import logging
import pathlib
from collections import defaultdict

import cgexplore as cgx
import numpy as np
import stk
import stko
from rmsd import check_reflections, int_atom, kabsch_rmsd, reorder_hungarian

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


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


class EnvVariables:
    """Define environment variables."""

    project_dir = pathlib.Path(
        "/home/atarzia/workingspace/model_enum_data/desymm_finder/"
    )
    project_dir.mkdir(exist_ok=True, parents=True)

    pymol_path = pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )

    cg_figures = project_dir / pathlib.Path("figures/")
    cg_figures.mkdir(exist_ok=True, parents=True)
    cg_structures = project_dir / pathlib.Path("structures/")
    cg_structures.mkdir(exist_ok=True, parents=True)
    cg_ligands = project_dir / pathlib.Path("ligands/")
    cg_ligands.mkdir(exist_ok=True, parents=True)
    cg_calculations = project_dir / pathlib.Path("calculations/")
    cg_calculations.mkdir(exist_ok=True, parents=True)
    cg_outputdata = project_dir / pathlib.Path("outputdata/")
    cg_outputdata.mkdir(exist_ok=True, parents=True)


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
