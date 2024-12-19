"""Utilities module."""

import logging
from copy import deepcopy

import cgexplore as cgx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


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


# Capper.
capper_bead = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=1,
)

# Converging ligands.
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
constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.0, 1e5),
    "bc": ("bond", 1.0, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mbg": ("angle", 180, 1e2),
    "mbc": ("angle", 180, 1e2),
    "egb": ("angle", 120, 1e2),
    "deg": ("angle", 180, 1e2),
    # Torsions.
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "d": ("nb", 10.0, 1.0),
    "e": ("nb", 10.0, 1.0),
    "g": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
}


def precursors_to_forcefield(
    pair: str,
    ditopic: cgx.molecular.Precursor,
    ditopic_meas: dict[str, float],
) -> cgx.forcefields.ForceField:
    """Get a forcefield from precursor definitions."""
    # Define bead libraries.
    present_beads = (
        cbead_c,
        abead_c,
        ebead_c,
        capper_bead,
        binder_bead,
        tetra_bead,
    )
    cgx.molecular.BeadLibrary(present_beads)

    definer_dict = deepcopy(constant_definer_dict)

    cg_scale = 2

    if isinstance(ditopic, cgx.molecular.SixBead):
        beads = ditopic.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        definer_dict["dd"] = ("bond", ditopic_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", ditopic_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", ditopic_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", ditopic_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", ditopic_meas["dde"], 1e2)
        if "edde_v" in ditopic_meas:
            definer_dict["edde"] = (
                "tors",
                "0123",
                ditopic_meas["edde_v"],
                ditopic_meas["edde_k"],
                1,
            )
        else:
            definer_dict["edde"] = ("tors", "0123", 180, 50, 1)
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)

    return cgx.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=2,
        present_beads=present_beads,
        definer_dict=definer_dict,
    )
