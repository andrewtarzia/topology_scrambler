"""Utilities module."""

import logging
import pathlib
from collections import Counter
from copy import deepcopy

import cgexplore as cgx
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


# Diverging ligands.
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
ebead_d = cgx.molecular.CgBead(
    element_string="Mn",
    bead_class="f",
    bead_type="f",
    coordination=2,
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
steric_bead = cgx.molecular.CgBead(
    element_string="S",
    bead_class="s",
    bead_type="s",
    coordination=1,
)
inner_bead = cgx.molecular.CgBead(
    element_string="Ir",
    bead_class="i",
    bead_type="i",
    coordination=2,
)

constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.0, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "mbg": ("angle", 180, 1e2),
    "aca": ("angle", 180, 1e2),
    "egb": ("angle", 120, 1e2),
    "deg": ("angle", 180, 1e2),
    # Torsions.
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "d": ("nb", 10.0, 1.0),
    "e": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "g": ("nb", 10.0, 1.0),
}


def precursors_to_forcefield(  # noqa: PLR0913
    pair: str,
    diverging: cgx.molecular.Precursor,
    converging: cgx.molecular.Precursor,
    conv_meas: dict[str, float],
    dive_meas: dict[str, float],
    new_definer_dict: dict[str, tuple] | None = None,
) -> cgx.forcefields.ForceField:
    """Get a forcefield from precursor definitions."""
    # Define bead libraries.
    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        ebead_d,
        binder_bead,
        tetra_bead,
        steric_bead,
        inner_bead,
    )
    cgx.molecular.BeadLibrary(present_beads)

    if new_definer_dict is None:
        definer_dict = deepcopy(constant_definer_dict)
    else:
        definer_dict = deepcopy(new_definer_dict)

    cg_scale = 2

    if isinstance(converging, cgx.molecular.SixBead):
        beads = converging.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        if "d" in conv_meas:
            definer_dict["d"] = ("nb", 10.0, conv_meas["d"])
        definer_dict["dd"] = ("bond", conv_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", conv_meas["dde"], 1e2)
        definer_dict["edde"] = ("tors", "0123", 180, 50, 1)
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)

    elif isinstance(converging, cgx.molecular.StericSixBead):
        beads = converging.get_bead_set()

        if (
            "d" not in beads
            or "e" not in beads
            or "g" not in beads
            or "s" not in beads
        ):
            raise RuntimeError
        definer_dict["di"] = ("bond", conv_meas["dd"] / cg_scale / 2, 1e5)
        definer_dict["is"] = ("bond", conv_meas["is"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["ide"] = ("angle", conv_meas["ide"], 1e2)
        definer_dict["did"] = ("angle", 180, 1e2)
        definer_dict["dis"] = ("angle", 90, 1e2)
        definer_dict["edide"] = ("tors", "0134", 180, 50, 1)
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)
        definer_dict["s"] = ("nb", conv_meas["se"], conv_meas["s"])

    else:
        raise NotImplementedError

    if isinstance(diverging, cgx.molecular.TwoC1Arm):
        beads = diverging.get_bead_set()
        if "a" not in beads or "c" not in beads:
            raise RuntimeError
        definer_dict["ba"] = ("bond", dive_meas["ba"] / cg_scale, 1e5)
        ac = dive_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", dive_meas["bac"], 1e2)
        definer_dict["bacab"] = ("tors", "0134", dive_meas["bacab"], 50, 1)
    else:
        raise NotImplementedError

    return cgx.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=2,
        present_beads=present_beads,
        definer_dict=definer_dict,
    )


def plot_xy(
    xproperty: str,
    ensemble: dict[str:dict],
    min_energy: float,
    figure_dir: pathlib.Path,
    ligand_name: str,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if xproperty in ("binder_angles",):
        ax.scatter(
            [ensemble[i][xproperty][0] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )
        ax.scatter(
            [ensemble[i][xproperty][1] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    elif xproperty in ("torsion_state",):
        xs = [Counter(ensemble[i][xproperty]) for i in ensemble]

        xs = [i.get("b", 0) for i in xs]

        ax.scatter(
            xs,
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    else:
        ax.scatter(
            [ensemble[i][xproperty] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xproperty, fontsize=16)
    ax.set_ylabel("relative energy [kJ/mol]", fontsize=16)
    if xproperty == "binder_adjacent_torsion":
        ax.set_xlim(-180, 180)

    if xproperty == "binder_angles":
        ax.set_xlim(0, 180)

    if xproperty == "binder_binder_angle":
        ax.set_xlim(0, 180)

    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"xy_{xproperty}_{ligand_name}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
