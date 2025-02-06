"""Utilities module."""

from copy import deepcopy

import cgexplore as cgx
import numpy as np
import stk


class StericTwoC1Arm(cgx.molecular.Precursor):
    """A `TwoC1Arm` Precursor."""

    def __init__(
        self,
        bead: cgx.molecular.CgBead,
        abead1: cgx.molecular.CgBead,
        steric_bead: cgx.molecular.CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = (
            f"s2C1{bead.bead_type}{abead1.bead_type}{steric_bead.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            steric_bead.bead_type: steric_bead,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{bead.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=f"[{abead1.element_string}][{bead.element_string}]"
            f"([{steric_bead.element_string}])[{abead1.element_string}]",
            functional_groups=new_fgs,
            position_matrix=np.array(
                [[-3, 0, 0], [0, 0, 0], [0, 1, 0], [3, 0, 0]]
            ),
        )


# Small ligands.
cbead_d = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
abead_d = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)

# Sixbead small ligands.
c2bead_d = cgx.molecular.CgBead(
    element_string="Zn",
    bead_class="z",
    bead_type="z",
    coordination=2,
)
a2bead_d = cgx.molecular.CgBead(
    element_string="Rh",
    bead_class="r",
    bead_type="r",
    coordination=2,
)
e2bead_d = cgx.molecular.CgBead(
    element_string="Mn",
    bead_class="f",
    bead_type="f",
    coordination=2,
)

# Large ligands.
cbead_c = cgx.molecular.CgBead(
    element_string="Ni",
    bead_class="d",
    bead_type="d",
    coordination=2,
)
abead_c = cgx.molecular.CgBead(
    element_string="Fe",
    bead_class="e",
    bead_type="e",
    coordination=2,
)
ebead_c = cgx.molecular.CgBead(
    element_string="Ga",
    bead_class="g",
    bead_type="g",
    coordination=2,
)

# Constant.
binder_bead = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
tetra_bead = cgx.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)
steric_bead = cgx.molecular.CgBead(
    element_string="S",
    bead_class="s",
    bead_type="s",
    coordination=1,
)

constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.0, 1e5),
    "cs": ("bond", 1.0, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "mbg": ("angle", 180, 1e2),
    "mbf": ("angle", 180, 1e2),
    "aca": ("angle", 180, 1e2),
    "acs": ("angle", 90, 1e2),
    # Torsions.
    "bacs": ("tors", "0123", 180, 50, 1),
    "bacab": ("tors", "0134", 180, 50, 1),
    "edde": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "mbge": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "rzzr": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "mbfr": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "d": ("nb", 10.0, 1.0),
    "e": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "g": ("nb", 10.0, 1.0),
    "s": ("nb", 10.0, 1.0),
    "z": ("nb", 10.0, 1.0),
    "f": ("nb", 10.0, 1.0),
    "r": ("nb", 10.0, 1.0),
}


def precursors_to_forcefield(  # noqa: PLR0913
    pair: str,
    large: cgx.molecular.Precursor,
    small: cgx.molecular.Precursor,
    large_meas: dict[str, float],
    small_meas: dict[str, float],
    vdw_bond_cutoff: int | None = None,
) -> cgx.forcefields.ForceField:
    """Get a forcefield from precursor definitions."""
    # Define bead libraries.
    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        c2bead_d,
        a2bead_d,
        e2bead_d,
        binder_bead,
        tetra_bead,
        steric_bead,
    )
    cgx.molecular.BeadLibrary(present_beads)

    definer_dict = deepcopy(constant_definer_dict)

    cg_scale = 2

    if isinstance(large, cgx.molecular.SixBead):
        beads = large.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        definer_dict["dd"] = ("bond", large_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", large_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", large_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", large_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", large_meas["dde"], 1e2)
        definer_dict["egb"] = ("angle", large_meas["egb"], 1e2)
        definer_dict["deg"] = ("angle", large_meas["deg"], 1e2)

    if isinstance(small, cgx.molecular.TwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads:
            raise RuntimeError
        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)

    elif isinstance(small, StericTwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads or "s" not in beads:
            raise RuntimeError

        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)
        definer_dict["s"] = ("nb", 10.0, small_meas["s"])

    elif isinstance(small, cgx.molecular.SixBead):
        beads = small.get_bead_set()
        if "z" not in beads or "r" not in beads or "f" not in beads:
            raise RuntimeError
        definer_dict["zz"] = ("bond", small_meas["dd"] / cg_scale, 1e5)
        definer_dict["zr"] = ("bond", small_meas["de"] / cg_scale, 1e5)
        definer_dict["rf"] = ("bond", small_meas["eg"] / cg_scale, 1e5)
        definer_dict["fb"] = ("bond", small_meas["gb"] / cg_scale, 1e5)
        definer_dict["zzr"] = ("angle", small_meas["dde"], 1e2)
        definer_dict["rfb"] = ("angle", small_meas["egb"], 1e2)
        definer_dict["zrf"] = ("angle", small_meas["deg"], 1e2)

    else:
        raise NotImplementedError

    return cgx.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=vdw_bond_cutoff,
        present_beads=present_beads,
        definer_dict=definer_dict,
    )
