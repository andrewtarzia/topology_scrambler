"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import os
import pathlib
import shutil
from collections import defaultdict

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import atomlite
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    abead_c,
    abead_d,
    analyse_cage,
    binder_bead,
    cbead_c,
    cbead_d,
    constant_definer_dict,
    ebead_c,
    get_regraphed_molecule,
    get_stk_topology_code,
    get_vertexset_molecule,
    optimise_cage,
    passes_graph_bb_iso,
    precursors_to_forcefield,
    tetra_bead,
)
from model_enumeration.utilities import (
    contains_parallels,
    eb_str,
    isomer_energy,
    multi_cmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
attempts = (
    None,
    "regraphed-spring-10",
    "regraphed-kamada-10",
    "set-kamada-10",
    "set-spring-10",
    "set-spectral-10",
)


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
    width_height: tuple[float, float] = (7, 10),
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=width_height)
    energies = {}

    xs = []

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        if multi not in xs:
            xs.append(multi)

        pair = tuple(entry.properties["pair"].split("_"))
        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

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
        y = [i[0] for i in pairs].index(pair)

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
    ax.set_yticklabels(["_".join(i) for i in [i[0] for i in pairs]])

    for i in list(range(len(xs))):
        ax.axvline(int(i) + 0.8, c="k", alpha=0.5)

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


def make_summary_plot2(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    structure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, (axx, ax) = plt.subplots(
        nrows=2,
        figsize=(16, 6),
        height_ratios=[1, 3],
        sharex=True,
    )

    name_converter = {
        ("la", "st52"): "L1-l\n4:2:3",
        ("la", "st52_11"): "L1-l\n1:1:1",
        ("la", "st52_243"): "L1-l\n2:4:3",
        ("la", "st52_153"): "L1-l\n1:5:3",
        ("la", "st52_513"): "L1-l\n5:1:3",
        ("la", "st52_132"): "L1-l\n1:3:2",
        ("la", "st52_312"): "L1-l\n3:1:2",
        ("la", "st5_11"): "L1-s\n1:1:1",
        ("la", "st5"): "L1-s\n4:2:3",
        ("la", "st5_243"): "L1-s\n2:4:3",
        ("la", "st5_153"): "L1-s\n1:5:3",
        ("la", "st5_513"): "L1-s\n5:1:3",
        ("la", "st5_132"): "L1-s\n1:3:2",
        ("la", "st5_312"): "L1-s\n3:1:2",
        ("la", "c1"): "L1b\n4:2:3",
        ("la", "c1_11"): "L1b\n1:1:1",
        ("la", "c1_243"): "L1b\n2:4:3",
        ("la", "c1_153"): "L1b\n1:5:3",
        ("la", "c1_513"): "L1b\n5:1:3",
        ("la", "c1_132"): "L1b\n1:3:2",
        ("la", "c1_312"): "L1b\n3:1:2",
    }

    x_multi_mins = {i: defaultdict(float) for i in multi_cmap}
    x_count = {i: defaultdict(int) for i in multi_cmap}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])

        pair = tuple(entry.properties["pair"].split("_"))

        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

        if pair not in name_converter:
            continue

        x = [i[0] for i in pairs].index(pair)
        x_count[multi][x] += 1
        energy = entry.properties["energy_per_bb"]

        if energy < 1:
            stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            ).write(structure_dir / f"{entry.key}_optc.mol")

        if entry.properties["num_components"] > 1:
            continue

        if x not in x_multi_mins[multi]:
            x_multi_mins[multi][x] = energy
        else:
            x_multi_mins[multi][x] = min((x_multi_mins[multi][x], energy))

    for i in range(len(pairs) - 1):
        ax.axvline(x=i + 0.5, c="k", alpha=0.2)
        axx.axvline(x=i + 0.5, c="k", alpha=0.2)

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
            label=f"$m=${multi}",
        )
        axx.plot(
            list(x_count[multi]),
            [x_count[multi][i] for i in x_count[multi]],
            c="none",
            markerfacecolor=multi_cmap[multi],
            mec="k",
            marker="o",
            zorder=2,
            markersize=12,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(name_converter))))
    ax.set_xticklabels(
        [
            name_converter[j]
            for j in [i[0] for i in pairs]
            if j in name_converter
        ],
    )
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)

    ax.legend(fontsize=16)

    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_ylabel("calcs", fontsize=16)

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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run starship case study studying Pd(II) heteroleptic systems."""
    run = _parse_args().run

    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")

    run_prefix = "starships"
    calculation_dir = wd / f"{run_prefix}_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / f"{run_prefix}_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / f"{run_prefix}_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / f"{run_prefix}_data"
    data_dir.mkdir(exist_ok=True)
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / f"{run_prefix}"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / f"{run_prefix}.db"

    ligand_measures = {
        "la": {
            "dd": 7.0,
            "de": 1.5,
            "dde": 170,
            "eg": 1.4,
            "gb": 1.4,
            "egb": 120,
            "deg": 180,
        },
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
    }

    pairs = {
        "la_st5_11": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (1, 1, 1),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
            "vdw_cutoff": 2,
        },
        "la_st5_132": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (1, 3, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5_312": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (3, 1, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5_243": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (2, 4, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5_153": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (1, 5, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5_513": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (5, 1, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52_11": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (1, 1, 1),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
            "vdw_cutoff": 2,
        },
        "la_st52_132": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (1, 3, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52_312": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (3, 1, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52_243": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (2, 4, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52_153": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (1, 5, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52_513": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (5, 1, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1_11": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (1, 1, 1),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
            "vdw_cutoff": 2,
        },
        "la_c1_132": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (1, 3, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1_312": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (3, 1, 2),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1_243": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (2, 4, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1_153": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (1, 5, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1_513": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (5, 1, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
    }
    pairs_to_predict = [
        (
            (i.split("_")[0], "_".join(i.split("_")[1:])),
            pairs[i]["multipliers"],
        )
        for i in pairs
    ]

    for pair in pairs:
        forcefield = precursors_to_forcefield(
            pair=pair,
            large=pairs[pair]["large"],
            small=pairs[pair]["small"],
            large_meas=ligand_measures[pairs[pair]["large_name"]],
            small_meas=ligand_measures[pairs[pair]["small_name"]],
            vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
            constant_definer_dict=constant_definer_dict,
        )

        if run:
            small_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["small"].get_building_block(),
                name=f"{pair}_{pairs[pair]['small'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            small_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['small'].get_name()}_optl.mol"
                )
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["tetra"].get_building_block(),
                name=pairs[pair]["tetra"].get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(
                str(ligand_dir / f"{pairs[pair]['tetra'].get_name()}_optl.mol")
            )

            large_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["large"].get_building_block(),
                name=f"{pair}_{pairs[pair]['large'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            large_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['large'].get_name()}_optl.mol"
                )
            )

            for multiplier in pairs[pair]["multipliers"]:
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                nmetals = pairs[pair]["stoichiometry_L_L_M"][2] * multiplier
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: pairs[pair]["stoichiometry_L_L_M"][2]
                        * multiplier,
                        large_bb: pairs[pair]["stoichiometry_L_L_M"][0]
                        * multiplier,
                        small_bb: pairs[pair]["stoichiometry_L_L_M"][1]
                        * multiplier,
                    },
                    graph_type=f"{1 * nmetals}P{2 * nmetals}",
                    graph_set="rx",
                )
                logging.info(
                    "graph iteration has %s graphs", iterator.count_graphs()
                )

                possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                    iterator=iterator
                )
                logging.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )

                # Use known topology codes.
                stk_topology_code, stk_positions = get_stk_topology_code(
                    graph_type=f"{1 * nmetals}P{2 * nmetals}",
                )

                vertex_positions = {
                    nidx: np.array(stk_positions[nidx]) * 10
                    for nidx in stk_topology_code.get_nx_graph().nodes
                }
                sidx = -1
                midx = 0
                run_topology_codes = []
                for bb_config in possible_bbdicts:
                    name = (
                        f"{pair}_{multiplier}_{sidx}_{midx}_b{bb_config.idx}"
                    )

                    if not passes_graph_bb_iso(
                        topology_code=stk_topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append((stk_topology_code, bb_config))
                    try:
                        constructed_molecule = stk.ConstructedMolecule(
                            cgx.topologies.CustomTopology(  # type: ignore[arg-type]
                                building_blocks=bb_config.get_building_block_dictionary(),
                                vertex_prototypes=iterator.get_vertex_prototypes(
                                    unaligning=False
                                ),
                                # Convert to edge prototypes.
                                edge_prototypes=stk_topology_code.edges_from_connection(
                                    iterator.get_vertex_prototypes(
                                        unaligning=False
                                    )
                                ),
                                vertex_alignments=None,
                                vertex_positions=vertex_positions,
                                scale_multiplier=iterator.scale_multiplier,
                                optimizer=stk.MCHammer(),
                            )
                        )
                    except ValueError:
                        continue
                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )

                    # Optimise and save.
                    logging.info("building %s", name)
                    try:
                        conformer = optimise_cage(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                            potential_names=[],
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
                                number_building_blocks=(
                                    iterator.get_num_building_blocks()
                                ),
                            )

                            properties = {
                                "forcefield_dict": (
                                    forcefield.get_forcefield_dictionary()
                                ),
                                "energy_per_bb": energy_per_bb,
                                "l1": pairs[pair]["large_name"],
                                "l2": pairs[pair]["small_name"],
                                "pair": pair,
                                "num_components": num_components,
                                "num_bbs": (
                                    iterator.get_num_building_blocks()
                                ),
                                "multiplier": multiplier,
                                "topology_idx": sidx,
                                "mash_idx": midx,
                                "topology_code_vmap": tuple(
                                    (int(i[0]), int(i[1]))
                                    for i in stk_topology_code.vertex_map
                                ),
                                "bb_config_idx": bb_config.idx,
                            }
                            cgx.utilities.AtomliteDatabase(
                                database_path
                            ).add_properties(
                                key=name,
                                property_dict=properties,
                            )

                            analyse_cage(
                                database_path=database_path,
                                name=name,
                                forcefield=forcefield,
                            )
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )
                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

                run_topology_codes = []
                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Filter graphs for 1-loops.
                    if contains_parallels(topology_code):
                        continue

                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append((topology_code, bb_config))

                    generated_conformers = []
                    for midx, scale in enumerate(attempts):
                        name = (
                            f"{pair}_{multiplier}_{idx}_{midx}"
                            f"_b{bb_config.idx}"
                        )

                        try:
                            if isinstance(scale, str) and "regraphed" in scale:
                                constructed_molecule = get_regraphed_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=bb_config,
                                )

                            else:
                                constructed_molecule = get_vertexset_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=bb_config,
                                )
                        except ValueError:
                            continue

                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )
                        # Optimise and save.
                        logging.info("building %s", name)

                        try:
                            potential_names = [
                                f"{pair}_{multiplier}_{idx}_"
                                f"{nmash_idx}_b{bb_config.idx}"
                                for nmash_idx in range(len(attempts))
                            ]
                            if scale is None:
                                conformer = cgx.scram.graph_optimise_cage(
                                    molecule=constructed_molecule,
                                    name=name,
                                    output_dir=calculation_dir,
                                    forcefield=forcefield,
                                    platform=None,
                                    database_path=database_path,
                                )
                                # Copy the file over.
                                wipfinal = (
                                    calculation_dir / f"{name}_wipfinal.mol"
                                )
                                wipfinal_new = (
                                    calculation_dir / f"{name}_final.mol"
                                )
                                if wipfinal.exists():
                                    shutil.copy(wipfinal, wipfinal_new)

                            else:
                                conformer = optimise_cage(
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
                                energy_per_bb = (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                )

                                properties = {
                                    "forcefield_dict": (
                                        forcefield.get_forcefield_dictionary()
                                    ),
                                    "energy_per_bb": energy_per_bb,
                                    "l1": pairs[pair]["large_name"],
                                    "l2": pairs[pair]["small_name"],
                                    "pair": pair,
                                    "num_components": num_components,
                                    "num_bbs": (
                                        iterator.get_num_building_blocks()
                                    ),
                                    "multiplier": multiplier,
                                    "topology_idx": idx,
                                    "mash_idx": midx,
                                    "topology_code_vmap": tuple(
                                        (int(i[0]), int(i[1]))
                                        for i in topology_code.vertex_map
                                    ),
                                    "bb_config_idx": bb_config.idx,
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

                    min_energy_conformer = sorted(
                        generated_conformers, key=lambda p: p[2]
                    )[0]
                    min_energy_name, min_energy_structure, _ = (
                        min_energy_conformer
                    )

                    min_energy_structure.write(
                        str(structure_dir / f"{min_energy_name}_optc.mol")
                    )

                    analyse_cage(
                        database_path=database_path,
                        name=min_energy_name,
                        forcefield=forcefield,
                    )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="starship_1.png",
        pairs=pairs_to_predict,
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="starship_2.png",
        pairs=pairs_to_predict,
        structure_dir=structure_dir,
    )


if __name__ == "__main__":
    main()
