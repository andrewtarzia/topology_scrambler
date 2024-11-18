"""Utilities module."""

import json
import logging
import pathlib

import matplotlib as mpl
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": tstr_cmap(0.0),
    "2": tstr_cmap(0.05),
    "3": tstr_cmap(0.1),
    "4": tstr_cmap(0.15),
    "5": tstr_cmap(0.2),
    "6": tstr_cmap(0.25),
    "7": tstr_cmap(0.30),
    "8": tstr_cmap(0.35),
    "9": tstr_cmap(0.40),
    "10": tstr_cmap(0.45),
    "11": tstr_cmap(0.5),
    "12": tstr_cmap(0.55),
}


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


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
