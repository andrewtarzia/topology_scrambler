"""Utilities module."""

import cgexplore
import logging

import openmm

import stk
import numpy as np
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def element_from_type(
    bead_type: str,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> str:
    """Get element of cgbead from type of cgbead."""
    return next(i.element_string for i in present_beads if i.bead_type == bead_type)


def define_bond(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetBond:
    """Define target from a known structured list."""
    return cgexplore.terms.TargetBond(
        type1=interaction_key[0],
        type2=interaction_key[1],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        bond_r=openmm.unit.Quantity(
            value=interaction_list[1], unit=openmm.unit.angstrom
        ),
        bond_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer**2,
        ),
    )


def define_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetAngle:
    """Define target from a known structured list."""
    return cgexplore.terms.TargetAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        angle=openmm.unit.Quantity(value=interaction_list[1], unit=openmm.unit.degrees),
        angle_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2,
        ),
    )


def define_pyramid(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetPyramidAngle:
    """Define target from a known structured list."""
    angle = openmm.unit.Quantity(value=interaction_list[1], unit=openmm.unit.degrees)
    opposite_angle = openmm.unit.Quantity(
        value=cgexplore.utilities.convert_pyramid_angle(
            angle.value_in_unit(angle.unit)
        ),
        unit=angle.unit,
    )
    return cgexplore.terms.TargetPyramidAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        angle=angle,
        opposite_angle=opposite_angle,
        angle_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2,
        ),
    )


def define_cosine_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetCosineAngle:
    """Define target from a known structured list."""
    return cgexplore.terms.TargetCosineAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        n=interaction_list[1],
        b=interaction_list[2],
        angle_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoule / openmm.unit.mole,
        ),
    )


def define_torsion(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetTorsion:
    """Define target from a known structured list."""
    measured_atom_ids = tuple(int(i) for i in interaction_list[1])
    if len(measured_atom_ids) != 4:  # noqa: PLR2004
        msg = (
            f"trying to define torsion with measured atoms {measured_atom_ids}"
            ", should be 4"
        )
        raise RuntimeError(msg)
    return cgexplore.terms.TargetTorsion(
        search_string=tuple(i for i in interaction_key),
        search_estring=tuple(
            element_from_type(test, present_beads) for test in interaction_key
        ),
        measured_atom_ids=measured_atom_ids,
        phi0=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.degrees,
        ),
        torsion_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        torsion_n=interaction_list[4],
    )


def define_nonbonded(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
) -> cgexplore.terms.TargetNonbonded:
    """Define target from a known structured list."""
    return cgexplore.terms.TargetNonbonded(
        bead_class=interaction_key[0],
        bead_element=element_from_type(interaction_key[0], present_beads),
        epsilon=openmm.unit.Quantity(
            value=interaction_list[1],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        sigma=openmm.unit.Quantity(
            value=interaction_list[2], unit=openmm.unit.angstrom
        ),
        force="custom-excl-vol",
    )


# Diverging ligands.
cbead_d = cgexplore.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
abead_d = cgexplore.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)
ebead_d = cgexplore.molecular.CgBead(
    element_string="Mn",
    bead_class="f",
    bead_type="f",
    coordination=2,
)

# Converging ligands.
cbead_c = cgexplore.molecular.CgBead(
    element_string="Ni",
    bead_class="d",
    bead_type="d",
    coordination=2,
)
abead_c = cgexplore.molecular.CgBead(
    element_string="Fe",
    bead_class="e",
    bead_type="e",
    coordination=2,
)
ebead_c = cgexplore.molecular.CgBead(
    element_string="Ga",
    bead_class="g",
    bead_type="g",
    coordination=2,
)

# Constant.
binder_bead = cgexplore.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
tetra_bead = cgexplore.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)


def get_forcefield(
    identifier: str,
    prefix: str,
    vdw_bond_cutoff: int,
    present_beads: tuple[cgexplore.molecular.CgBead, ...],
    definer_dict: dict,
) -> cgexplore.forcefields.ForceField:  # noqa: C901
    """Get forcefield."""

    bond_terms: list = []
    angle_terms: list[
        cgexplore.terms.TargetAngle | cgexplore.terms.TargetCosineAngle
    ] = []
    torsion_terms: list = []
    nonbonded_terms: list = []
    for key_ in definer_dict:
        term = definer_dict[key_]  # type: ignore[assignment]

        if term[0] == "bond":
            bond_terms.append(
                define_bond(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "pyramid":
            angle_terms.append(
                define_pyramid(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "angle":
            angle_terms.append(
                define_angle(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "cosine":
            angle_terms.append(
                define_cosine_angle(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "tors":
            torsion_terms.append(
                define_torsion(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "nb":
            nonbonded_terms.append(
                define_nonbonded(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

    return cgexplore.forcefields.ForceField(
        identifier=identifier,
        prefix=prefix,
        present_beads=present_beads,
        bond_targets=tuple(bond_terms),
        angle_targets=tuple(angle_terms),
        torsion_targets=tuple(torsion_terms),
        nonbonded_targets=tuple(nonbonded_terms),
        vdw_bond_cutoff=vdw_bond_cutoff,
    )


present_beads = (
    cbead_d,
    abead_d,
    cbead_c,
    abead_c,
    ebead_c,
    ebead_d,
    binder_bead,
    tetra_bead,
)
constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.5, 1e5),  ####
    "ab": ("bond", 1.0, 1e5),  ####
    "gb": ("bond", 1.0, 1e5),  ####
    "fb": ("bond", 1.0, 1e5),  ####
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "mbg": ("angle", 180, 1e2),
    "mbf": ("angle", 180, 1e2),
    "aca": ("angle", 180, 1e2),
    "ede": ("angle", 180, 1e2),
    # Torsions.
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "d": ("nb", 10.0, 1.0),
    "e": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "g": ("nb", 10.0, 1.0),
    "f": ("nb", 10.0, 1.0),
}

definer_dict_lf_ls1 = deepcopy(constant_definer_dict)
definer_dict_lf_ls1["ac"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls1["bac"] = ("angle", 150, 1e2)
definer_dict_lf_ls1["bacab"] = ("tors", "0134", 180, 50, 1)  ####
definer_dict_lf_ls1["dd"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls1["ed"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls1["eg"] = ("bond", 1, 1e5)  ####
definer_dict_lf_ls1["bge"] = ("angle", 180, 1e2)  #### binder
definer_dict_lf_ls1["dde"] = ("angle", 130, 1e2)  ####
definer_dict_lf_ls1["deg"] = ("angle", 130, 1e2)  ####
definer_dict_lf_ls1["edde"] = ("tors", "0123", 180, 50, 1)  ####
# definer_dict_lf_ls1["geddeg"] = ("tors", "0145", 180, 0, 1)  ####
forcefield_lf_ls1 = get_forcefield(
    identifier="lfls1",
    prefix="min_opt",
    present_beads=present_beads,
    vdw_bond_cutoff=2,
    definer_dict=definer_dict_lf_ls1,
)


definer_dict_lf_ls9 = deepcopy(constant_definer_dict)
definer_dict_lf_ls9["ac"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls9["bac"] = ("angle", 165, 1e2)
definer_dict_lf_ls9["bacab"] = ("tors", "0134", 180, 50, 1)  ####
definer_dict_lf_ls9["dd"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls9["ed"] = ("bond", 2, 1e5)  ####
definer_dict_lf_ls9["eg"] = ("bond", 1, 1e5)  ####
definer_dict_lf_ls9["bge"] = ("angle", 180, 1e2)  #### binder
definer_dict_lf_ls9["dde"] = ("angle", 130, 1e2)  ####
definer_dict_lf_ls9["deg"] = ("angle", 130, 1e2)  ####
definer_dict_lf_ls9["edde"] = ("tors", "0123", 180, 50, 1)  ####
# definer_dict_lf_ls9["geddeg"] = ("tors", "0145", 180, 0, 1)  ####
forcefield_lf_ls9 = get_forcefield(
    identifier="lfls9",
    prefix="min_opt",
    present_beads=present_beads,
    vdw_bond_cutoff=2,
    definer_dict=definer_dict_lf_ls9,
)

definer_dict_la_st5 = deepcopy(constant_definer_dict)
definer_dict_la_st5["cc"] = ("bond", 2, 1e5)  ####
definer_dict_la_st5["ac"] = ("bond", 2, 1e5)  ####
definer_dict_la_st5["af"] = ("bond", 1, 1e5)  ####
definer_dict_la_st5["bfa"] = ("angle", 180, 1e2)  #### binder
definer_dict_la_st5["cca"] = ("angle", 130, 1e2)  ####
definer_dict_la_st5["caf"] = ("angle", 130, 1e2)  ####
definer_dict_la_st5["acca"] = ("tors", "0123", 180, 50, 1)  ####
# definer_dict_la_st5["faccaf"] = ("tors", "0145", 150, 50, 1)  ####
definer_dict_la_st5["dd"] = ("bond", 2, 1e5)  ####
definer_dict_la_st5["ed"] = ("bond", 2, 1e5)  ####
definer_dict_la_st5["eg"] = ("bond", 1, 1e5)  ####
definer_dict_la_st5["bge"] = ("angle", 180, 1e2)  #### binder
definer_dict_la_st5["dde"] = ("angle", 130, 1e2)  ####
definer_dict_la_st5["deg"] = ("angle", 130, 1e2)  ####
definer_dict_la_st5["edde"] = ("tors", "0123", 180, 50, 1)  ####
# definer_dict_la_st5["geddeg"] = ("tors", "0145", 180, 0, 1)  ####
forcefield_la_st5 = get_forcefield(
    identifier="last5",
    prefix="min_opt",
    present_beads=present_beads,
    vdw_bond_cutoff=2,
    definer_dict=definer_dict_la_st5,
)

definer_dict_la_st52 = deepcopy(constant_definer_dict)
definer_dict_la_st52["cc"] = ("bond", 2, 1e5)  ####
definer_dict_la_st52["ac"] = ("bond", 2, 1e5)  ####
definer_dict_la_st52["af"] = ("bond", 1, 1e5)  ####
definer_dict_la_st52["bfa"] = ("angle", 180, 1e2)  #### binder
definer_dict_la_st52["cca"] = ("angle", 130, 1e2)  ####
definer_dict_la_st52["caf"] = ("angle", 130, 1e2)  ####
definer_dict_la_st52["acca"] = ("tors", "0123", 180, 50, 1)  ####
# definer_dict_la_st52["faccaf"] = ("tors", "0145", 150, 50, 1)  ####
definer_dict_la_st52["dd"] = ("bond", 2, 1e5)  ####
definer_dict_la_st52["ed"] = ("bond", 2, 1e5)  ####
definer_dict_la_st52["eg"] = ("bond", 1, 1e5)  ####
definer_dict_la_st52["bge"] = ("angle", 180, 1e2)  #### binder
definer_dict_la_st52["dde"] = ("angle", 130, 1e2)  ####
definer_dict_la_st52["deg"] = ("angle", 130, 1e2)  ####
definer_dict_la_st52["edde"] = ("tors", "0123", 180, 50, 1)  ####
definer_dict_la_st52["geddeg"] = ("tors", "0145", 180, 0, 1)  ####
forcefield_la_st52 = get_forcefield(
    identifier="last52",
    prefix="min_opt",
    present_beads=present_beads,
    vdw_bond_cutoff=2,
    definer_dict=definer_dict_la_st52,
)


class SixBead(cgexplore.molecular.Precursor):
    """A Precursor."""

    def __init__(
        self,
        bead: cgexplore.molecular.CgBead,
        abead1: cgexplore.molecular.CgBead,
        abead2: cgexplore.molecular.CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"6C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead2.element_string}X1][{abead1.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{bead.element_string}]"
                f"[{abead1.element_string}][{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-6, 3, 0.2],
                    [-4, 2, 0],
                    [-2, 0.1, 0],
                    [2, 0, 0],
                    [4, 2, 0],
                    [6, 3, 0.2],
                ]
            ),
        )
