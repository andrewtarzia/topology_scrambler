"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
from collections import defaultdict

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    StericTwoC1Arm,
    a2bead_d,
    abead_c,
    abead_d,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)
from model_enumeration.utilities import (
    eb_str,
    isomer_energy,
    multi_cmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def convert_coordinates(
    molecule: stk.ConstructedMolecule,
    nx_positions: np.ndarray,
) -> stk.ConstructedMolecule:
    """Convert networkx coordinates to molecule position matrix."""
    # We allow these to independantly fail because the nx graphs can
    # be ridiculous, just get the first that passes.
    for scaler in (3, 5, 10, 15):
        pos_mat = np.array([nx_positions[i] for i in nx_positions])
        new_mol = molecule.with_position_matrix(pos_mat * scaler)
        break
    return new_mol.with_centroid(np.array((0.0, 0.0, 0.0)))


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    database.add_properties(key=name, property_dict={"lowest_e_of_mash": True})

    if "large_binder_binder_angles" not in properties:
        ss_dists = (
            stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
                molecule=final_molecule,
                metal_atom_nos=(16,),
            )
        )
        if len(ss_dists) != 0:
            min_ss_value = min(ss_dists.values())
            max_ss_value = max(ss_dists.values())

            database.add_properties(
                key=name,
                property_dict={
                    "min_ss_dist": min_ss_value,
                    "max_ss_dist": max_ss_value,
                },
            )

        # Get the bg angles.
        ligands = stko.molecule_analysis.DecomposeMOC().decompose(
            molecule=final_molecule,
            metal_atom_nos=(46,),
        )

        small_binder_binder_angles = []
        large_binder_binder_angles = []
        potential_smiles = [
            "[Pb]~[Ga]",  # Large.
            "[Pb]~[Ba]",  # Small.
            "[Pb]~[Mn]",  # Small.
        ]
        for lig in ligands:
            for smiles in potential_smiles:
                as_building_block = stk.BuildingBlock.init_from_molecule(
                    lig,
                    stk.SmartsFunctionalGroupFactory(
                        smarts=smiles, bonders=(0,), deleters=(1,)
                    ),
                )
                if as_building_block.get_num_functional_groups() == 2:  # noqa: PLR2004
                    large = smiles in ("[Pb]~[Ga]",)
                    break

            vectors = [
                as_building_block.get_centroid(atom_ids=fg.get_bonder_ids())
                - as_building_block.get_centroid(atom_ids=fg.get_deleter_ids())
                for fg in as_building_block.get_functional_groups()
            ]
            normed = [i / np.linalg.norm(i) for i in vectors]
            angle = np.degrees(
                stko.vector_angle(vector1=normed[0], vector2=normed[1])
            )
            if large:
                large_binder_binder_angles.append(angle)
            else:
                small_binder_binder_angles.append(angle)

        database.add_properties(
            key=name,
            property_dict={
                "large_binder_binder_angles": large_binder_binder_angles,
                "small_binder_binder_angles": small_binder_binder_angles,
                "min_distance": (
                    cgx.analysis.GeomMeasure().calculate_min_distance(
                        database.get_molecule(key=name)
                    )["min_distance"]
                ),
            },
        )


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 10))
    energies = {}

    xs = ["1", "2", "3", "4"]

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        pair = tuple(entry.key.split("_")[:2])
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        midx = entry.properties["mash_idx"]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), tidx, bidx, midx))

    # create the new map
    cmap = plt.cm.Blues_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = [0, 0.3, 1.0, 5.0, 10.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for (pair, multi), evalues in energies.items():
        sorted_energies = sorted(evalues, key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = pairs.index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap=cmap,
            norm=norm,
        )
        ax.text(
            x=x + 0.5,
            y=y,
            s=f"t:{min_energy[1]},b:{min_energy[2]}",
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="k",
            fontsize=10,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(pairs))))
    ax.set_yticklabels(["_".join(i) for i in pairs])
    ax.set_xlim(None, 4)

    ax.axvline(0.8, c="k", alpha=0.5)
    ax.axvline(1.8, c="k", alpha=0.5)
    ax.axvline(2.8, c="k", alpha=0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"1:1:1 {eb_str()}", fontsize=16)

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


def make_summary_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(16, 5))

    x_multi_mins = {i: defaultdict(float) for i in multi_cmap}
    min_at_all_xs = defaultdict(int)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        x = pairs.index((l1, l2))

        if "lowest_e_of_mash" not in entry.properties:
            continue
        energy = entry.properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        if x not in x_multi_mins[multi]:
            x_multi_mins[multi][x] = energy
        else:
            x_multi_mins[multi][x] = min((x_multi_mins[multi][x], energy))

        if x not in min_at_all_xs:
            min_at_all_xs[x] = energy
        else:
            min_at_all_xs[x] = min((min_at_all_xs[x], energy))

    for i in range(len(pairs) - 1):
        ax.axvline(x=i + 0.5, c="k", alpha=0.2)

    for multi in multi_cmap:
        if len(x_multi_mins[multi]) == 0:
            continue
        edict = x_multi_mins[multi]

        ax.plot(
            sorted(edict),
            [edict[i] for i in sorted(edict)],
            c="none",
            markerfacecolor=multi_cmap[multi],
            mec="k",
            marker="o",
            alpha=1,
            markersize=12,
        )

    ax.plot(
        sorted(min_at_all_xs),
        [min_at_all_xs[i] for i in sorted(min_at_all_xs)],
        c="k",
        alpha=1,
        zorder=-1,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(pairs))))
    ax.set_xticklabels(["_".join(i) for i in pairs], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)

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


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, (ax, ax1) = plt.subplots(
        ncols=2,
        figsize=(10, 5),
        width_ratios=[3, 1],
        sharey=True,
    )

    stages = (
        "opt1",
        "shifted",
        "smd",
        "nx00",
        "nx10",
        "nx20",
        "nx30",
        "nx01",
        "nx11",
        "nx21",
        "nx31",
        "nx02",
        "nx12",
        "nx22",
        "nx32",
        "ns",
    )

    sources = {i: 0 for i in stages}
    mashes = {}
    lowe_sources = {i: 0 for i in stages}  # Produces low energy structures.
    lowe_mashes = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        midx = entry.properties["mash_idx"]
        if midx not in mashes:
            mashes[midx] = 0
            lowe_mashes[midx] = 0

        sources[entry.properties["source"]] += 1
        mashes[midx] += 1
        energy = entry.properties["energy_per_bb"]
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1
            lowe_mashes[midx] += 1

    ax.bar(
        stages,
        [lowe_sources[i] for i in stages],
        color="#086788",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 1.0",
    )
    ax.bar(
        stages,
        [sources[i] for i in stages],
        color="none",
        edgecolor="k",
        lw=2,
        label="all",
    )

    ax1.bar(
        [int(i) for i in mashes],
        [lowe_mashes[i] for i in mashes],
        color="#086788",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 1.0",
    )
    ax1.bar(
        [int(i) for i in mashes],
        [mashes[i] for i in mashes],
        color="none",
        edgecolor="k",
        lw=2,
        label="all",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)  # , color=color)
    ax.legend(fontsize=16)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    ax.set_xlabel("stage", fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("mash idx", fontsize=16)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def sterics_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(tuple)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]

        x = entry.properties["forcefield_dict"]["v_dict"]["s"]

        if "min_ss_dist" not in entry.properties:
            continue
        y = entry.properties["min_ss_dist"]

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas[(multi, l1, l2)][x][1]
            ):
                datas[(multi, l1, l2)][x] = (
                    y,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas[(multi, l1, l2)][x] = (y, entry.properties["energy_per_bb"])

    xlbl = r"min. $r_{s-s}$ [$\AA$]"

    for (multi, l1, l2), xdict in datas.items():
        ax.scatter(
            list(xdict),
            [xdict[i][0] for i in xdict],
            alpha=1.0,
            c=multi_cmap[multi],
            ec="k",
            s=60,
            label=f"M{multi}" if (l1, l2) == ("lf", "ls3") else None,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel(xlbl, fontsize=16)
    ax.legend(ncol=1, fontsize=16)
    ax.set_ylim(0, None)

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


def binder_vector_angles_plot(  # noqa: C901, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    datas_lge: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    datas_sma: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]

        if "large_binder_binder_angles" not in entry.properties:
            continue
        ylge = entry.properties["large_binder_binder_angles"]
        ysma = entry.properties["small_binder_binder_angles"]

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas_lge[(multi, l1, l2)][1]
            ):
                datas_lge[(multi, l1, l2)] = (
                    ylge,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas_lge[(multi, l1, l2)] = (
                ylge,
                entry.properties["energy_per_bb"],
            )

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas_sma[(multi, l1, l2)][1]
            ):
                datas_sma[(multi, l1, l2)] = (
                    ysma,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas_sma[(multi, l1, l2)] = (
                ysma,
                entry.properties["energy_per_bb"],
            )

    lsdone = set()
    for (multi, l1, l2), xdict in datas_sma.items():
        ydict = datas_lge[(multi, l1, l2)]

        if xdict[1] > 1.0:
            alpha = 0.3
            zorder = -1
            c = "gray"
            ec = "none"
            s = 30
            label = None

        elif xdict[1] > 0.3:  # noqa: PLR2004
            alpha = 1
            zorder = 0
            c = multi_cmap[multi]
            ec = "none"
            s = 30
            label = f"M{multi}"
            if label in lsdone:
                label = None
            lsdone.add(label)

        else:
            alpha = 1
            zorder = 1
            c = multi_cmap[multi]
            ec = "k"
            s = 60
            label = None

        ax.scatter(
            xdict[0],
            ydict[0],
            alpha=alpha,
            marker="o",
            c=c,
            ec=ec,
            s=s,
            label=label,
            zorder=zorder,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("mean small binder angle [$^\\circ$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylabel("mean large binder angle [$^\\circ$]", fontsize=16)
    ax.plot((0, 180), (0, 180), c="k", zorder=-1)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=16)

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


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgenknown_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgenknown_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgenknown_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgenknown_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgenknown"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgenknown.db"

    ligand_measures = {
        "lf-fake": {
            "egb": 120,
            "deg": 180,
            "dd": 8.0,
            "de": 4.0,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
        },
        "ls10-fake": {"ba": 2.8, "aa": 5.4, "bac": 165, "s": 0.0},
        "lf-xrd": {
            "egb": 120,
            "deg": 180,
            "dd": 7.87,
            "de": 4.25,
            "dde": 126.9,
            "eg": 2.75 / 2,
            "gb": 2.75 / 2,
        },
        "ls10-xrd": {"ba": 2.8, "aa": 5.25, "bac": 166, "s": 0.0},
        # From prep.
        "lf": {
            "egb": 120,
            "deg": 180,
            "dd": 8.0,
            "de": 4.3,
            "dde": 133,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e10": {
            "egb": 120,
            "deg": 180,
            "dd": 5.9,
            "de": 4.1,
            "dde": 139,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e11": {
            "egb": 120,
            "deg": 180,
            "dd": 6.9,
            "de": 1.4,
            "dde": 170,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e12": {
            "egb": 120,
            "deg": 180,
            "dd": 7.0,
            "de": 1.5,
            "dde": 167,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e13": {
            "egb": 120,
            "deg": 180,
            "dd": 10.5,
            "de": 1.4,
            "dde": 151,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e14": {
            "egb": 120,
            "deg": 180,
            "dd": 5.9,
            "de": 4.1,
            "dde": 143,
            "eg": 1.4,
            "gb": 1.4,
        },
        # From optl.
        "l2": {"ba": 2.8, "aa": 4.9, "bac": 150, "s": 0.0},
        "ls2": {"ba": 2.8, "aa": 4.7, "bac": 155, "s": 0.0},
        "ls3": {"ba": 2.8, "aa": 5.0, "bac": 145, "s": 0.5},
        "ls4": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 1.0},
        "ls5": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 0.5},
        "ls7": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 1.5},
        "ls8": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 2.0},
        "l3": {"ba": 2.8, "aa": 5.3, "bac": 165, "s": 0.0},
        "ls10": {"ba": 2.8, "aa": 5.4, "bac": 167, "s": 0.0},
        "l1": {"ba": 2.8, "aa": 8.2, "bac": 136, "s": 0.0},
        "e16": {"ba": 2.8, "aa": 7.3, "bac": 121, "s": 0.0},
        "e18": {"ba": 2.8, "aa": 10.0, "bac": 121, "s": 0.0},
        "e17": {
            "egb": 90,
            "deg": 150,
            "dd": 7.3,
            "de": 4.0,
            "dde": 151,
            "eg": 2.4,
            "gb": 2.8,
        },
        "la": {
            "egb": 90,
            "deg": 150,
            "dd": 6.1,
            "de": 8.3,
            "dde": 136,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lb": {
            "egb": 90,
            "deg": 150,
            "dd": 6.7,
            "de": 5.7,
            "dde": 148,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lc": {
            "egb": 90,
            "deg": 150,
            "dd": 7.0,
            "de": 5.7,
            "dde": 150,
            "eg": 2.4,
            "gb": 2.8,
        },
        "ld": {
            "egb": 90,
            "deg": 150,
            "dd": 7.5,
            "de": 5.7,
            "dde": 165,
            "eg": 2.4,
            "gb": 2.8,
        },
    }

    ligand_types = {
        "lf": "sixbead",
        "lf-fake": "sixbead",
        "lf-xrd": "sixbead",
        "e10": "sixbead",
        "e11": "sixbead",
        "e12": "sixbead",
        "e13": "sixbead",
        "e14": "sixbead",
        "e17": "sixbead",
        "la": "sixbead",
        "lb": "sixbead",
        "lc": "sixbead",
        "ld": "sixbead",
        "e16": "twoarm",
        "e18": "twoarm",
        "l2": "twoarm",
        "ls10-fake": "twoarm",
        "ls10-xrd": "twoarm",
        "ls2": "twoarm",
        "ls3": "stwoarm",
        "ls4": "stwoarm",
        "ls5": "stwoarm",
        "ls7": "stwoarm",
        "ls8": "stwoarm",
        "ls9": "twoarm",
        "ls10": "twoarm",
        "l1": "twoarm",
        "l3": "twoarm",
    }

    pairs_to_predict = [
        # large, small.
        ("lf", "l2"),
        ("lf-fake", "ls10-fake"),
        ("lf-xrd", "ls10-xrd"),
        ("lf", "ls2"),
        ("lf", "ls3"),
        ("lf", "ls4"),
        ("lf", "ls5"),
        ("lf", "ls7"),
        ("lf", "ls8"),
        ("lf", "l3"),
        ("lf", "ls10"),
        ("la", "l1"),
        ("lb", "l1"),
        ("lc", "l1"),
        ("ld", "l1"),
        ("la", "l2"),
        ("lb", "l2"),
        ("lc", "l2"),
        ("ld", "l2"),
        ("la", "l3"),
        ("lb", "l3"),
        ("lc", "l3"),
        ("ld", "l3"),
        ("e10", "e16"),
        ("e17", "e16"),
        ("e17", "e10"),
        ("e10", "e11"),
        ("e14", "e16"),
        ("e14", "e18"),
        ("e10", "e18"),
        ("e10", "e12"),
        ("e14", "e11"),
        ("e14", "e12"),
        ("e13", "e11"),
        ("e13", "e12"),
        ("e14", "e13"),
        ("e12", "e11"),
    ]

    pairs = {}
    for large, small in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        else:
            msg = large
            raise NotImplementedError(msg)

        if ligand_types[small] == "twoarm":
            small_prec = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
        elif ligand_types[small] == "stwoarm":
            small_prec = StericTwoC1Arm(
                bead=cbead_d, abead1=abead_d, steric_bead=steric_bead
            )
        elif ligand_types[small] == "sixbead":
            small_prec = cgx.molecular.SixBead(
                bead=c2bead_d,
                abead1=a2bead_d,
                abead2=e2bead_d,
            )
        else:
            msg = small
            raise NotImplementedError(msg)

        multi = (2, 3, 4) if large in ("lf", "lf-fake", "lf-xrd") else (2, 3)
        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "multipliers": multi,
            "vdw_cutoff": 2,
        }

    if args.run:
        for pair in pairs:
            large_name = pairs[pair]["large_name"]
            small_name = pairs[pair]["small_name"]

            large = pairs[pair]["large"]
            small = pairs[pair]["small"]
            tetra = pairs[pair]["tetra"]

            forcefield = precursors_to_forcefield(
                pair=pair,
                large=large,
                small=small,
                large_meas=ligand_measures[large_name],
                small_meas=ligand_measures[small_name],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
            )

            small_bb = cgx.utilities.optimise_ligand(
                molecule=small.get_building_block(),
                name=f"{pair}_{small.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            small_bb.write(
                str(ligand_dir / f"{pair}_{small.get_name()}_optl.mol")
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(str(ligand_dir / f"{tetra.get_name()}_optl.mol"))

            large_bb = cgx.utilities.optimise_ligand(
                molecule=large.get_building_block(),
                name=f"{pair}_{large.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            large_bb.write(
                str(ligand_dir / f"{pair}_{large.get_name()}_optl.mol")
            )

            pair_lowest_energy = float("inf")

            for multiplier in pairs[pair]["multipliers"]:
                logging.info("doing: pair %s, multi %s", pair, multiplier)

                attempts = (
                    "spring",
                    "kamada",
                    1.1,
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                )

                generated_conformers = []
                for midx, scale in enumerate(attempts):
                    actual_scale = 1 if not isinstance(scale, float) else scale

                    name = f"{pair}_{multiplier}_stk_{midx}_bstk"
                    if multiplier == 2:  # noqa: PLR2004
                        constructed_molecule = stk.ConstructedMolecule(
                            stk.cage.M2L4Lantern(
                                building_blocks={
                                    tetra_bb: (0, 1),
                                    large_bb: (2, 3),
                                    small_bb: (4, 5),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 6

                    elif multiplier == 3:  # noqa: PLR2004
                        constructed_molecule = stk.ConstructedMolecule(
                            stk.cage.M3L6(
                                building_blocks={
                                    tetra_bb: (0, 1, 2),
                                    large_bb: (3, 5, 7),
                                    small_bb: (4, 6, 8),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 9

                    elif multiplier == 4:  # noqa: PLR2004
                        constructed_molecule = stk.ConstructedMolecule(
                            cgx.topologies.CGM4L8(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    large_bb: (4, 6, 8, 10),
                                    small_bb: (5, 7, 9, 11),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 12

                    else:
                        raise NotImplementedError

                    if scale == "spring":
                        stko_graph = stko.Network.init_from_molecule(
                            constructed_molecule
                        )
                        nx_positions = nx.spring_layout(
                            stko_graph.get_graph(), dim=3
                        )
                        constructed_molecule = convert_coordinates(
                            constructed_molecule, nx_positions
                        )

                    if scale == "kamada":
                        stko_graph = stko.Network.init_from_molecule(
                            constructed_molecule
                        )
                        nx_positions = nx.kamada_kawai_layout(
                            stko_graph.get_graph(), dim=3
                        )
                        constructed_molecule = convert_coordinates(
                            constructed_molecule, nx_positions
                        )

                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )
                    # Optimise and save.
                    logging.info("building %s", name)

                    try:
                        potential_names = [
                            f"{pair}_{multiplier}_stk_{nmash_idx}_bstk"
                            for nmash_idx in range(len(attempts))
                        ]

                        conformer = cgx.scram.optimise_cage(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                            potential_names=potential_names,
                        )

                        if conformer is not None:
                            num_components = len(
                                stko.Network.init_from_molecule(
                                    conformer.molecule
                                ).get_connected_components()
                            )
                            energy_per_bb = cgx.utilities.get_energy_per_bb(
                                energy_decomposition=(
                                    conformer.energy_decomposition
                                ),
                                number_building_blocks=num_bbs,
                            )

                            properties = {
                                "forcefield_dict": (
                                    forcefield.get_forcefield_dictionary()
                                ),
                                "energy_per_bb": energy_per_bb,
                                "l1": large_name,
                                "l2": small_name,
                                "pair": pair,
                                "num_components": num_components,
                                "num_bbs": num_bbs,
                                "multiplier": multiplier,
                                "topology_idx": "stk",
                                "mash_idx": midx,
                                "bb_config_idx": "stk",
                            }
                            cgx.utilities.AtomliteDatabase(
                                database_path
                            ).add_properties(
                                key=name,
                                property_dict=properties,
                            )
                            generated_conformers.append(
                                (
                                    name,
                                    conformer.molecule.with_centroid(
                                        (0, 0, 0)
                                    ),
                                    energy_per_bb,
                                )
                            )

                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

                    try:
                        current_energy = (
                            cgx.utilities.AtomliteDatabase(database_path)
                            .get_entry(name)
                            .properties["energy_per_bb"]
                        )
                        pair_lowest_energy = min(
                            (pair_lowest_energy, current_energy)
                        )

                    except RuntimeError:
                        pass

                min_energy_conformer = sorted(
                    generated_conformers, key=lambda p: p[2]
                )[0]
                min_energy_name, min_energy_structure, _ = min_energy_conformer

                min_energy_structure.write(
                    str(structure_dir / f"{min_energy_name}_optc.mol")
                )

                analyse_cage(database_path=database_path, name=min_energy_name)

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
        pairs=pairs_to_predict,
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
        pairs=pairs_to_predict,
    )
    sterics_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_6.png",
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_5.png",
    )

    binder_vector_angles_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_7.png",
    )


if __name__ == "__main__":
    main()
