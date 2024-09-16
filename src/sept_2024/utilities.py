"""Utilities module."""

import json
import logging
import pathlib
from copy import deepcopy

import cgexplore
import numpy as np
import stk

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
steric_bead = cgexplore.molecular.CgBead(
    element_string="S",
    bead_class="s",
    bead_type="s",
    coordination=1,
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


def precursors_to_forcefield(
    pair: str,
    diverging: cgexplore.molecular.Precursor,
    converging: cgexplore.molecular.Precursor,
    conv_meas: dict[str, float],
    dive_meas: dict[str, float],
) -> cgexplore.forcefields.ForceField:
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
    )
    cgexplore.molecular.BeadLibrary(present_beads)

    definer_dict = deepcopy(constant_definer_dict)

    cg_scale = 2

    if isinstance(converging, SixBead):
        beads = converging.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        definer_dict["dd"] = ("bond", conv_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", conv_meas["dde"], 1e2)
        definer_dict["edde"] = ("tors", "0123", 180, 50, 1)
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)

    elif isinstance(converging, StericSixBead):
        beads = converging.get_bead_set()

        if (
            "d" not in beads
            or "e" not in beads
            or "g" not in beads
            or "s" not in beads
        ):
            raise RuntimeError
        definer_dict["dd"] = ("bond", conv_meas["dd"] / cg_scale / 2, 1e5)
        definer_dict["ds"] = ("bond", 1.0, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", conv_meas["dde"], 1e2)
        definer_dict["ddd"] = ("angle", 180, 1e2)
        definer_dict["dds"] = ("angle", 90, 1e2)
        definer_dict["eddde"] = ("tors", "0134", 180, 50, 1)
        definer_dict["edds"] = ("tors", "0123", 180, 50, 1)
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)
        definer_dict["s"] = ("nb", 10.0, conv_meas["s"])

    else:
        raise NotImplementedError

    if isinstance(diverging, cgexplore.molecular.TwoC1Arm):
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

    return cgexplore.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=2,
        present_beads=present_beads,
        definer_dict=definer_dict,
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


class StericSixBead(cgexplore.molecular.Precursor):
    """A Precursor."""

    def __init__(
        self,
        bead: cgexplore.molecular.CgBead,
        abead1: cgexplore.molecular.CgBead,
        abead2: cgexplore.molecular.CgBead,
        sbead: cgexplore.molecular.CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._sbead = sbead
        self._name = (
            f"6S2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
            f"{sbead.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
            sbead.bead_type: sbead,
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
                f"([{sbead.element_string}])[{bead.element_string}]"
                f"[{abead1.element_string}][{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-6, 3, 0.2],
                    [-4, 2, 0],
                    [-2, 0.1, 0],
                    [0, 0.1, 0],
                    [0, 1, 0],
                    [2, 0, 0],
                    [4, 2, 0],
                    [6, 3, 0.2],
                ]
            ),
        )


def save_vertex_positions(
    name: str,
    calculation_dir: pathlib.Path,
    structure_dir: pathlib.Path,
    molecule: stk.ConstructedMolecule,
) -> None:
    """Save vertex positions of molecule to file."""
    vertex_file = calculation_dir / f"{name}_vertices.json"
    if not vertex_file.exists():
        constructed_molecule = molecule.with_structure_from_file(
            structure_dir / f"{name}_optc.mol"
        )

        bbs = {}
        for ai in constructed_molecule.get_atom_infos():
            bbid = ai.get_building_block_id()
            if bbid not in bbs:
                bbs[bbid] = []
            bbs[bbid].append(ai.get_atom().get_id())

        centroids = {
            i: tuple(
                float(i)
                for i in constructed_molecule.get_centroid(atom_ids=bbs[i])
            )
            for i in bbs
        }
        with vertex_file.open("w") as f:
            json.dump(centroids, f, indent=4)
