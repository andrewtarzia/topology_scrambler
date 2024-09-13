"""Utilities module."""

import itertools as it
import logging
import pathlib
from collections import Counter, abc, defaultdict
from copy import deepcopy
from typing import assert_never

import cgexplore
import numpy as np
import openmm
import stk
import stko
from rmsd import check_reflections, int_atom, kabsch_rmsd, reorder_hungarian
from topologies import (
    length_2_heteroleptic_bb_dicts,
    length_3_heteroleptic_bb_dicts,
    length_4_heteroleptic_bb_dicts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_potential_bb_dicts(
    tstr: str,
    ratio: tuple[int, int],
    bb_type: str,
) -> abc.Sequence[dict[int, abc.Sequence[int]]]:
    """Get potential building block dictionaries."""
    match bb_type:
        case "ditopic":
            possibilities, count_to_add = length_2_heteroleptic_bb_dicts(tstr)
            current_counter = max(
                [
                    max(possibilities[i])
                    for i in possibilities
                    if len(possibilities[i]) != 0
                ]
            )

        case "tritopic":
            possibilities, count_to_add = length_3_heteroleptic_bb_dicts(tstr)
            # Use minus one because of the +1 later on needed for other states.
            current_counter = -1

        case "tetratopic":
            possibilities, count_to_add = length_4_heteroleptic_bb_dicts(tstr)
            # Use minus one because of the +1 later on needed for other states.
            current_counter = -1

        case _ as unreachable:
            assert_never(unreachable)

    modifiable = [i for i in possibilities if len(possibilities[i]) == 0]

    saved = set()
    possible_dicts = []
    for combo in it.product(modifiable, repeat=count_to_add):
        counted = Counter(combo).values()
        current_ratio = [i / min(counted) for i in counted]
        if len(current_ratio) != len(ratio):
            continue

        if tuple(i for i in current_ratio) != ratio:
            continue
        if combo in saved:
            continue
        saved.add(combo)

        new_possibility = deepcopy(possibilities)
        for idx, bb in enumerate(combo):
            new_possibility[bb].append(current_counter + idx + 1)

        possible_dicts.append((len(possible_dicts), new_possibility))

    msg = "bring rmsd checker in here"
    logging.info(msg)
    msg = "use symmetry corrected RMSD on single-bead repr of tstr"
    logging.info(msg)

    return tuple(possible_dicts)


def create_zone(dmin: float, dmax: float, resolution: int) -> list[float]:
    """Create a higher resolution zone."""
    return list(range(dmin, dmax + 1, resolution))


def get_forcefield_dict(forcefield: cgexplore.forcefields.ForceField) -> dict:
    """Get the underlying forcefield dict."""
    # This is matched to the existing analysis code. I recommend
    # generalising in the future.
    ff_targets = forcefield.get_targets()
    k_dict = {}
    v_dict = {}

    for bt in ff_targets["bonds"]:
        cp = (bt.type1, bt.type2)
        k_dict["_".join(cp)] = bt.bond_k.value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer**2
        )
        v_dict["_".join(cp)] = bt.bond_r.value_in_unit(openmm.unit.angstrom)

    for at in ff_targets["angles"]:
        cp = (at.type1, at.type2, at.type3)
        try:
            k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.radian**2
            )
            v_dict["_".join(cp)] = at.angle.value_in_unit(openmm.unit.degrees)
        except TypeError:
            # Handle different angle types.
            k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                openmm.unit.kilojoule / openmm.unit.mole
            )
            v_dict["_".join(cp)] = (at.n, at.b)

    for at in ff_targets["torsions"]:
        cp = at.search_string
        k_dict["_".join(cp)] = at.torsion_k.value_in_unit(
            openmm.unit.kilojoules_per_mole
        )
        v_dict["_".join(cp)] = at.phi0.value_in_unit(openmm.unit.degrees)
    for at in ff_targets["nonbondeds"]:
        v_dict[at.bead_class] = at.sigma.value_in_unit(openmm.unit.angstrom)
        k_dict[at.bead_class] = at.epsilon.value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

    return {
        "ff_id": forcefield.get_identifier(),
        "ff_prefix": forcefield.get_prefix(),
        "k_dict": k_dict,
        "v_dict": v_dict,
    }


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
    conformer: cgexplore.molecular.Conformer
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
