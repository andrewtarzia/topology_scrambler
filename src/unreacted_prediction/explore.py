"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk
import stko
from explore_utilities import (
    abead_c,
    binder_bead,
    capper_bead,
    cbead_c,
    eb_str,
    ebead_c,
    isomer_energy,
    precursors_to_forcefield,
    tetra_bead,
)
from openmm import OpenMMException
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    systems = {
        ("cltp75", "1", "", "1"): {
            "name": "c-232-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "2", "", "1"): {
            "name": "c-464-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "2", "", "2"): {
            "name": "c-464-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "1", "nocap", "1"): {
            "name": "c-12-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "2", "nocap", "1"): {
            "name": "c-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "2", "nocap", "2"): {
            "name": "c-24-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "1", "", "1"): {
            "name": "c*2-232-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "2", "", "1"): {
            "name": "c*2-464-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "2", "", "2"): {
            "name": "c*2-464-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "1", "nocap", "1"): {
            "name": "c*2-12-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "2", "nocap", "1"): {
            "name": "c*2-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "2", "nocap", "2"): {
            "name": "c*2-24-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "1", "", "1"): {
            "name": "l-232-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "2", "", "1"): {
            "name": "l-464-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "2", "", "2"): {
            "name": "l-464-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "1", "nocap", "1"): {
            "name": "l-12-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "2", "nocap", "1"): {
            "name": "l-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "2", "nocap", "2"): {
            "name": "l-24-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "1", "", "1"): {
            "name": "l*2-232-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "2", "", "1"): {
            "name": "l*2-464-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "2", "", "2"): {
            "name": "l*2-464-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "1", "nocap", "1"): {
            "name": "l*2-12-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "2", "nocap", "1"): {
            "name": "l*2-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "2", "nocap", "2"): {
            "name": "l*2-24-2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        system = (
            entry.properties["ligname"],
            entry.properties["multiplier"],
            entry.properties["name_prefix"],
            entry.properties["allowed_num_components"],
        )
        if entry.properties["ligname"] not in (
            "ltp75",
            "ltp752",
            "cltp75",
            "cltp752",
        ):
            continue

        energy = entry.properties["energy_per_bb"]
        if energy < isomer_energy():
            logging.info("low energy system: %s", entry.key)

        if energy < systems[system]["min_energy"]:
            systems[system]["min_energy"] = energy
            systems[system]["min_key"] = entry.key

        systems[system]["data"].append(energy)

    rng = np.random.default_rng(seed=2)
    for i, system in enumerate(systems):
        if len(systems[system]["data"]) == 0:
            continue
        min_energy = min(systems[system]["data"])

        ax.scatter(
            [
                i + (2 * rng.random() - 1) * 0.3
                for j in range(len(systems[system]["data"]))
            ],
            systems[system]["data"],
            c="tab:blue",
            alpha=0.4,
            edgecolor="none",
            s=30,
            marker="o",
            zorder=1,
        )
        ax.scatter(
            i,
            min_energy,
            c="tab:orange",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
            zorder=2,
        )
        logging.info("%s_optc.mol", systems[system]["min_key"])

    ax.axvline(x=5 + 0.5, c="gray")
    ax.axvline(x=11 + 0.5, c="gray")
    ax.axvline(x=17 + 0.5, c="gray")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")

    ax.axhline(y=isomer_energy(), c="k", ls="--")

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def analyse_twist(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(16, 5))

    systems = {
        ("cltp0", "2", "nocap", "1"): {
            "name": "c-24-1-0",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp10", "2", "nocap", "1"): {
            "name": "c-24-1-10",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp20", "2", "nocap", "1"): {
            "name": "c-24-1-20",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp30", "2", "nocap", "1"): {
            "name": "c-24-1-30",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp40", "2", "nocap", "1"): {
            "name": "c-24-1-40",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp50", "2", "nocap", "1"): {
            "name": "c-24-1-50",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp60", "2", "nocap", "1"): {
            "name": "c-24-1-60",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp70", "2", "nocap", "1"): {
            "name": "c-24-1-70",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp75", "2", "nocap", "1"): {
            "name": "c-24-1-75",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp752", "2", "nocap", "1"): {
            "name": "c-24-1-75*2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp80", "2", "nocap", "1"): {
            "name": "c-24-1-80",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltp90", "2", "nocap", "1"): {
            "name": "c-24-1-90",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("cltpunr", "2", "nocap", "1"): {
            "name": "cu-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp0", "2", "nocap", "1"): {
            "name": "l-24-1-0",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp10", "2", "nocap", "1"): {
            "name": "l-24-1-10",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp20", "2", "nocap", "1"): {
            "name": "l-24-1-20",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp30", "2", "nocap", "1"): {
            "name": "l-24-1-30",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp40", "2", "nocap", "1"): {
            "name": "l-24-1-40",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp50", "2", "nocap", "1"): {
            "name": "l-24-1-50",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp60", "2", "nocap", "1"): {
            "name": "l-24-1-60",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp70", "2", "nocap", "1"): {
            "name": "l-24-1-70",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp75", "2", "nocap", "1"): {
            "name": "l-24-1-75",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp752", "2", "nocap", "1"): {
            "name": "l-24-1-75*2",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp80", "2", "nocap", "1"): {
            "name": "l-24-1-80",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltp90", "2", "nocap", "1"): {
            "name": "l-24-1-90",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
        ("ltpunr", "2", "nocap", "1"): {
            "name": "lu-24-1",
            "data": [],
            "min_key": None,
            "min_energy": float("inf"),
        },
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "ligname" not in entry.properties:
            continue
        system = (
            entry.properties["ligname"],
            entry.properties["multiplier"],
            entry.properties["name_prefix"],
            entry.properties["allowed_num_components"],
        )
        if system not in systems:
            continue

        energy = entry.properties["energy_per_bb"]
        if energy < systems[system]["min_energy"]:
            systems[system]["min_energy"] = energy
            systems[system]["min_key"] = entry.key

        systems[system]["data"].append(energy)

    rng = np.random.default_rng(seed=2)
    for i, system in enumerate(systems):
        if len(systems[system]["data"]) == 0:
            continue
        min_energy = min(systems[system]["data"])

        ax.scatter(
            [
                i + (2 * rng.random() - 1) * 0.3
                for j in range(len(systems[system]["data"]))
            ],
            systems[system]["data"],
            c="tab:blue",
            alpha=0.4,
            edgecolor="none",
            s=30,
            marker="o",
            zorder=1,
        )
        ax.scatter(
            i,
            min_energy,
            c="tab:orange",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
            zorder=2,
        )
        logging.info("%s_optc.mol", systems[system]["min_key"])

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.axhline(y=isomer_energy(), c="k", ls="--")

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
    logging.info("------------------")


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: cgx.scram.TopologyIterator,
    topology_code: cgx.scram.TopologyCode,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties

    if "name_prefix" not in properties:
        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        try:
            (
                ligname,
                multiplier,
                allowed_num_components,
                topology_idx,
                mash_idx,
            ) = name.split("_")
            name_prefix = ""
        except ValueError:
            (
                name_prefix,
                ligname,
                multiplier,
                allowed_num_components,
                topology_idx,
                mash_idx,
            ) = name.split("_")

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield.get_forcefield_dictionary(),
                "energy_per_bb": cgx.utilities.get_energy_per_bb(
                    energy_decomposition=properties["energy_decomposition"],
                    number_building_blocks=iterator.get_num_building_blocks(),
                ),
                "num_components": num_components,
                "ligname": ligname,
                "multiplier": multiplier,
                "allowed_num_components": allowed_num_components,
                "topology_idx": topology_idx,
                "mash_idx": mash_idx,
                "name_prefix": name_prefix,
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
                "min_distance": (
                    cgx.analysis.GeomMeasure().calculate_min_distance(
                        database.get_molecule(key=name)
                    )["min_distance"]
                ),
            },
        )


class Single(cgx.molecular.Precursor):
    """A single bead Precursor."""

    def __init__(self, bead: cgx.molecular.CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"S1{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}

        self._building_block = stk.BuildingBlock(
            smiles=f"[{bead.element_string}]",
            functional_groups=(
                stk.SingleAtom(
                    stk.Atom(
                        0,
                        charge=0,
                        atomic_number=cgx.molecular.periodic_table()[
                            bead.element_string
                        ],
                    )
                ),
            ),
            position_matrix=[[0, 0, 0]],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/unreacted/")
    figure_dir = wd / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    calculation_dir = wd / "explore_calculations"
    calculation_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = wd / "explore_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir = wd / "explore_ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    data_dir = wd / "explore_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "explore.db"

    ligand_measures = {
        # From prep.
        # With and without chiral torsion on a new dde definition.
        "cltp0": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180,
            "edde_k": 50,
        },
        "cltp10": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 10,
            "edde_k": 50,
        },
        "cltp20": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 20,
            "edde_k": 50,
        },
        "cltp30": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 30,
            "edde_k": 50,
        },
        "cltp40": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 40,
            "edde_k": 50,
        },
        "cltp50": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 50,
            "edde_k": 50,
        },
        "cltp60": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 60,
            "edde_k": 50,
        },
        "cltp70": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 70,
            "edde_k": 50,
        },
        "cltp75": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 75,
            "edde_k": 50,
        },
        "cltp80": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 80,
            "edde_k": 50,
        },
        "cltp90": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 90,
            "edde_k": 50,
        },
        "cltp752": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 75,
            "edde_k": 100,
        },
        "cltpunr": {
            "dd": 7.4,
            "de": 4.3,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180,
            "edde_k": 0,
        },
        # Rigid, with measured values, varying angle.
        "ltp0": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180,
            "edde_k": 50,
        },
        "ltp10": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 10,
            "edde_k": 50,
        },
        "ltp20": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 20,
            "edde_k": 50,
        },
        "ltp30": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 30,
            "edde_k": 50,
        },
        "ltp40": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 40,
            "edde_k": 50,
        },
        "ltp50": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 50,
            "edde_k": 50,
        },
        "ltp60": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 60,
            "edde_k": 50,
        },
        "ltp70": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 70,
            "edde_k": 50,
        },
        "ltp75": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 75,
            "edde_k": 50,
        },
        "ltp80": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 80,
            "edde_k": 50,
        },
        "ltp90": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 90,
            "edde_k": 50,
        },
        "ltp752": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 75,
            "edde_k": 100,
        },
        # Small, rigid, varying angle.
        # With flexibile backbone.
        "ltpunr": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180,
            "edde_k": 0,
        },
    }

    study_type = ((2, 1), (3, 2, 2))

    multipliers = (1, 2)
    num_components = (1, 2)

    if args.run:
        for (
            ligname,
            ditopic_meas,
        ), multiplier, num_component, stoichiometry_l_m_c in it.product(
            ligand_measures.items(), multipliers, num_components, study_type
        ):
            if stoichiometry_l_m_c == (2, 1) and num_component == 2:  # noqa: PLR2004
                continue
            if stoichiometry_l_m_c == (3, 2, 2) and ligname not in (
                "ltp75",
                "ltp752",
                "cltp75",
                "cltp752",
            ):
                continue

            ditopic = cgx.molecular.SixBead(
                bead=cbead_c, abead1=abead_c, abead2=ebead_c
            )
            capper = Single(bead=capper_bead)
            tetra = cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            )

            forcefield = precursors_to_forcefield(
                pair=ligname,
                ditopic=ditopic,
                ditopic_meas=ditopic_meas,
            )

            capper_name = f"{capper.get_name()}_f{forcefield.get_identifier()}"
            capper_bb = capper.get_building_block()
            capper_bb.write(str(ligand_dir / f"{capper_name}_optl.mol"))

            ditopic_name = (
                f"{ditopic.get_name()}_f{forcefield.get_identifier()}"
            )
            ditopic_bb = cgx.utilities.optimise_ligand(
                molecule=ditopic.get_building_block(),
                name=ditopic_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            ditopic_bb.write(str(ligand_dir / f"{ditopic_name}_optl.mol"))
            ditopic_bb = ditopic_bb.clone()

            tetra_name = f"{tetra.get_name()}_f{forcefield.get_identifier()}"
            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            tetra_bb.write(str(ligand_dir / f"{tetra_name}_optl.mol"))
            tetra_bb = tetra_bb.clone()

            # Currently, only testing the unreacted case.
            if stoichiometry_l_m_c == (3, 2, 2):
                graph_type = (
                    f"{stoichiometry_l_m_c[1]*multiplier}-4FG_"
                    f"{stoichiometry_l_m_c[0]*multiplier}-2FG_"
                    f"{stoichiometry_l_m_c[2]*multiplier}-1FG"
                )
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.IHomolepticTopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_m_c[1] * multiplier,
                        ditopic_bb: stoichiometry_l_m_c[0] * multiplier,
                        capper_bb: stoichiometry_l_m_c[2] * multiplier,
                    },
                    graph_type=graph_type,
                    graph_set="rx",
                    max_samples=int(1e5),
                    allowed_num_components=num_component,
                )

                name_prefix = ""

            elif stoichiometry_l_m_c == (2, 1):
                graph_type = (
                    f"{stoichiometry_l_m_c[1]*multiplier}P"
                    f"{stoichiometry_l_m_c[0]*multiplier}"
                )

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.IHomolepticTopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_m_c[1] * multiplier,
                        ditopic_bb: stoichiometry_l_m_c[0] * multiplier,
                    },
                    graph_type=graph_type,
                    graph_set="rx",
                    max_samples=None,
                    allowed_num_components=num_component,
                )

                name_prefix = "nocap_"

            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )

            for idx, topology_code in enumerate(iterator.yield_graphs()):
                # Do the construction.
                nx_graph = topology_code.get_nx_graph()
                # Handle problems with small topologies.
                try:
                    vertex_position_setters = (
                        None,
                        nx.spectral_layout(nx_graph, dim=3),
                        nx.spring_layout(nx_graph, dim=3),
                        nx.kamada_kawai_layout(nx_graph, dim=3),
                    )
                except ValueError:
                    vertex_position_setters = (None,)

                for mash_idx, nx_positions in enumerate(
                    vertex_position_setters
                ):
                    if nx_positions is not None:
                        vertex_positions = {
                            idx: np.array(nx_positions[idx]) * 10
                            for idx in topology_code.get_nx_graph().nodes
                        }
                        opt_function = cgx.scram.optimise_cage
                    else:
                        vertex_positions = None
                        opt_function = cgx.scram.graph_optimise_cage

                    # Do the construction.
                    constructed_molecule = cgx.scram.try_except_construction(
                        iterator=iterator,
                        topology_code=topology_code,
                        building_block_configuration=None,
                        vertex_positions=vertex_positions,
                    )

                    name = (
                        f"{name_prefix}{ligname}_{multiplier}_{num_component}_{idx}_"
                        f"{mash_idx}"
                    )

                    # The two components have to be the same.
                    if num_component == 2:  # noqa: PLR2004
                        network = stko.Network.init_from_molecule(
                            constructed_molecule
                        ).get_connected_components()
                        n_nodes = [comp.number_of_nodes() for comp in network]
                        n_edges = [comp.number_of_edges() for comp in network]
                        if (
                            n_nodes[0] != n_nodes[1]
                            or n_edges[0] != n_edges[1]
                        ):
                            continue

                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )
                    # Optimise and save.
                    logging.info("building %s", name)

                    try:
                        conformer = opt_function(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        if conformer is not None:
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=topology_code,
                        )

                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)
    analyse_twist(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="exp_2.png",
    )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="exp_1.png",
    )


if __name__ == "__main__":
    main()
