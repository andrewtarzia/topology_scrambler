"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import shutil
from collections import defaultdict

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
    analyse_cage,
    constant_definer_dict,
    define_pairs,
    get_regraphed_molecule,
    get_stk_topology_code,
    get_vertexset_molecule,
    make_opt_plot,
    optimise_cage,
    passes_graph_bb_iso,
    precursors_to_forcefield,
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

tarzia_result = {
    # large, small.
    ("la", "l1"): (0.54, "tab:orange"),
    ("lb", "l1"): (0.35, "tab:blue"),
    ("lc", "l1"): (0.34, "tab:blue"),
    ("ld", "l1"): (0.32, "tab:orange"),
    ("la", "l2"): (0.96, "tab:orange"),
    ("lb", "l2"): (0.73, "tab:orange"),
    ("lc", "l2"): (0.7, "tab:orange"),
    ("ld", "l2"): (0.74, "tab:orange"),
    ("la", "l3"): (1.19, "tab:orange"),
    ("lb", "l3"): (0.94, "tab:orange"),
    ("lc", "l3"): (0.92, "tab:orange"),
    ("ld", "l3"): (0.96, "tab:orange"),
    ("e10", "e16"): (0.47, "tab:blue"),
    ("e17", "e16"): (0.25, "tab:blue"),
    ("e17", "e10"): (0.35, "tab:orange"),
    ("e10", "e11"): (0.61, "tab:blue"),
    ("e14", "e16"): (0.44, "tab:blue"),
    ("e14", "e18"): (0.55, "tab:blue"),
    ("e10", "e18"): (0.59, "tab:blue"),
    ("e10", "e12"): (0.61, "tab:blue"),
    ("e14", "e11"): (0.57, "tab:blue"),
    ("e14", "e12"): (0.57, "tab:blue"),
    ("e13", "e11"): (0.5, "tab:blue"),
    ("e13", "e12"): (0.5, "tab:blue"),
    ("e14", "e13"): (0.65, "tab:orange"),
    ("e12", "e11"): (0.66, "tab:orange"),
}
have_alkynes = {"e14", "lb", "e17", "lc", "e18", "e10", "ld", "la"}


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


def parity_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

    ys = {i: float("inf") for i in tarzia_result}
    max_blue_g = 0
    max_blue_e = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = str(entry.properties["l1"])
        l2 = str(entry.properties["l2"])
        if multi != "2" or (l1, l2) not in tarzia_result:
            continue

        x, c = tarzia_result[(l1, l2)]
        y = entry.properties["energy_per_bb"]

        ys[(l1, l2)] = min((y, ys[(l1, l2)]))

    for (l1, l2), (x, c) in tarzia_result.items():
        if c == "tab:blue":
            max_blue_g = max((x, max_blue_g))
            if ys[(l1, l2)] != float("inf"):
                max_blue_e = max((ys[(l1, l2)], max_blue_e))

    rng = np.random.default_rng(12345)
    for (l1, l2), (x, c) in tarzia_result.items():
        if c == "tab:blue":
            xval = 0

        elif c == "tab:orange":
            xval = 1

        ax.scatter(
            (rng.random() - 0.5) * 0.6 + xval,
            x,
            c=c,
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
        )
        ax1.scatter(
            (rng.random() - 0.5) * 0.6 + xval,
            ys[(l1, l2)],
            c=c,
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"$g_{\mathrm{avg}}$", fontsize=16)
    ax1.set_ylabel(f"$m=2$ {eb_str()}", fontsize=16)
    ax.set_ylim(0, None)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["forms $cis$ cage", "not"])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["forms $cis$ cage", "not"])
    ax.axvline(x=0.5, c="k")
    ax1.axvline(x=0.5, c="k")
    ax.set_xlim(-0.5, 1.5)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_yscale("log")
    ax1.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)

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


def make_summary_plot2(  # noqa: C901, PLR0912, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    structure_dir: pathlib.Path,
    filename: str,
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(16, 5))

    x_multi_mins = {i: defaultdict(float) for i in multi_cmap}
    x_count = {i: defaultdict(int) for i in multi_cmap}
    min_at_all_xs = defaultdict(int)
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

    for i, pair in enumerate(pairs):
        if pair[0] not in tarzia_result:
            continue
        if tarzia_result[pair[0]][1] == "tab:blue":
            ax.plot((i - 0.2, i + 0.2), (10, 10), c="k")
        if pair[0][0] in have_alkynes or pair[0][1] in have_alkynes:
            ax.plot((i - 0.2, i + 0.2), (9, 9), c="r")

    ax.plot(
        sorted(min_at_all_xs),
        [min_at_all_xs[i] for i in sorted(min_at_all_xs)],
        c="k",
        alpha=1,
        zorder=-1,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(pairs))))
    ax.set_xticklabels(
        ["_".join(i) for i in [i[0] for i in pairs]], rotation=90
    )
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def binder_vector_angles_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(16, 5))

    datas_lge: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    datas_sma: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    xpos = {(l1, l2): i for i, (l1, l2) in enumerate(tarzia_result)}

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

    for (multi, l1, l2), xdict in datas_sma.items():
        ydict = datas_lge[(multi, l1, l2)]
        if (l1, l2) not in tarzia_result:
            continue

        res = tarzia_result[(l1, l2)]

        ec = "none" if xdict[1] > isomer_energy() else "k"

        ax.scatter(
            xpos[(l1, l2)],
            np.mean(ydict[0]) - np.mean(xdict[0]),
            alpha=1,
            marker="o",
            c=res[1],
            ec=ec,
            s=160,
            zorder=2,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axhline(y=0, c="k", zorder=-2)
    ax.set_ylabel("$\\Delta$ mean binder angle [$^\\circ$]", fontsize=16)
    ax.set_xticks(list(range(len(xpos))))
    ax.set_xticklabels(["_".join(i) for i in xpos], rotation=90)

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


def deviation_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(16, 5))

    paths = [
        # ("forcefield_dict.v_dict.m_b", "bond_data.Pb_Pd"), # noqa: ERA001
        # ("forcefield_dict.v_dict.e_g", "bond_data.Fe_Ga"), # noqa: ERA001
        # ("forcefield_dict.v_dict.d_e", "bond_data.Fe_Ni"), # noqa: ERA001
        # ("forcefield_dict.v_dict.d_d", "bond_data.Ni_Ni"), # noqa: ERA001
        # ("forcefield_dict.v_dict.a_c", "bond_data.Ag_Ba"), # noqa: ERA001
        # ("forcefield_dict.v_dict.b_a", "bond_data.Ba_Pb"), # noqa: ERA001
        # ("forcefield_dict.v_dict.g_b", "bond_data.Ga_Pb"), # noqa: ERA001
        # ("forcefield_dict.v_dict.z_z", "bond_data.Zn_Zn"), # noqa: ERA001
        # ("forcefield_dict.v_dict.z_r", "bond_data.Rh_Zn"), # noqa: ERA001
        # ("forcefield_dict.v_dict.r_f", "bond_data.Mn_Rh"), # noqa: ERA001
        # ("forcefield_dict.v_dict.f_b", "bond_data.Mn_Pb"), # noqa: ERA001
        ("forcefield_dict.v_dict.m_b_a", "angle_data.Pd_Pb_Ba"),
        ("forcefield_dict.v_dict.m_b_g", "angle_data.Pd_Pb_Ga"),
        ("forcefield_dict.v_dict.b_m_b", "angle_data.Pb_Pd_Pb"),
        # ("forcefield_dict.v_dict.b_a_c",
        #  "angle_data.Pb_Ba_Ag"),
        ("forcefield_dict.v_dict.e_g_b", "angle_data.Pb_Ga_Fe"),
        ("forcefield_dict.v_dict.d_e_g", "angle_data.Ga_Fe_Ni"),
        ("forcefield_dict.v_dict.d_d_e", "angle_data.Fe_Ni_Ni"),
        # ("forcefield_dict.v_dict.a_c_a",
        #  "angle_data.Ba_Ag_Ba"),
        ("forcefield_dict.v_dict.m_b_f", "angle_data.Pd_Pb_Mn"),
        ("forcefield_dict.v_dict.z_z_r", "angle_data.Rh_Zn_Zn"),
        ("forcefield_dict.v_dict.r_f_b", "angle_data.Pb_Mn_Rh"),
        ("forcefield_dict.v_dict.z_r_f", "angle_data.Mn_Rh_Zn"),
        # ("forcefield_dict.v_dict.b_a_c_a_b",
        #  "dihedral_data.Pb_Ba_Ag_Ba_Pb"),
        # ("forcefield_dict.v_dict.m_b_g_e",
        #  "dihedral_data.Fe_Ni_Ni_Fe"),
        # ("forcefield_dict.v_dict.e_d_d_e",
        #  "dihedral_data.Pd_Pb_Ga_Fe"),
        # ("forcefield_dict.v_dict.r_z_z_r",
        #  "dihedral_data.Rh_Zn_Zn_Rh"),
        # ("forcefield_dict.v_dict.m_b_f_r",
        #  "dihedral_data.Pd_Pb_Mn_Rh"),
    ]
    xpos = {i: j for j, i in enumerate(paths)}

    tsyst = (
        # "e10_e12",
        "e13_e11",
        "e13_e12",
    )
    tsyst_pass = (
        # "e10_e16",
        # "e17_e16",
        # "e10_e11",
        # "e14_e16",
        # "e14_e18",
        # "e10_e18",
        "e14_e11",
        "e14_e12",
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        pair = entry.properties["pair"]
        if pair in tsyst:
            xp = -0.2
            cp = "tab:blue"
        elif pair in tsyst_pass:
            xp = +0.2
            cp = "tab:orange"
        else:
            continue

        for pathstrings in paths:
            x = xpos[pathstrings]
            path1 = pathstrings[0].split(".")
            path2 = pathstrings[1].split(".")
            try:
                target = entry.properties[path1[0]][path1[1]][path1[2]]
                actual = entry.properties[path2[0]][path2[1]]
            except KeyError:
                continue

            for act in actual:
                if len(path2[1].split("_")) > 3:  # Torsion.  # noqa: PLR2004
                    rel = (0 - act) / 180
                elif "b_m_b" in pathstrings[0]:
                    target = 90 if act < 135 else 180  # noqa: PLR2004
                    rel = (target - act) / target
                else:
                    rel = (target - act) / target

                ax.scatter(
                    x + xp,
                    rel,
                    alpha=1,
                    marker="o",
                    c=cp,
                    ec="k",
                    s=160,
                    zorder=2,
                )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axhline(y=0, c="k", zorder=-2)
    ax.set_ylabel("rel. error from target", fontsize=16)
    ax.set_xticks(list(range(len(xpos))))
    ax.set_xticklabels([i[0].split(".")[-1] for i in xpos], rotation=90)

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


def case_study_1(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run case study 1 studying Pd(II) heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs1_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs1_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs1_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs1_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgencs1"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs1.db"

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        # From prep.
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
        "e12-1": {
            "egb": 120,
            "deg": 180,
            "dd": 7.0,
            "de": 1.5,
            "dde": 171,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e12-2": {
            "egb": 120,
            "deg": 180,
            "dd": 7.0,
            "de": 1.5,
            "dde": 163,
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
        "e13-1": {
            "egb": 120,
            "deg": 180,
            "dd": 10.8,
            "de": 1.4,
            "dde": 151,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e13-2": {
            "egb": 120,
            "deg": 180,
            "dd": 10.2,
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
        "l2": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 0.0},
        "ls2": {"ba": 2.8, "aa": 4.7, "bac": 144, "s": 0.0},
        "ls3": {"ba": 2.8, "aa": 5.0, "bac": 153, "s": 0.5},
        "l3": {"ba": 2.8, "aa": 5.3, "bac": 164, "s": 0.0},
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
            "dd": 2.3,
            "de": 8.3,
            "dde": 148,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lc": {
            "egb": 90,
            "deg": 150,
            "dd": 2.3,
            "de": 8.3,
            "dde": 150,
            "eg": 2.4,
            "gb": 2.8,
        },
        "ld": {
            "egb": 90,
            "deg": 150,
            "dd": 2.3,
            "de": 8.3,
            "dde": 165,
            "eg": 2.4,
            "gb": 2.8,
        },
    }

    ligand_types = {
        "e10": "sixbead",
        "e11": "sixbead",
        "e12": "sixbead",
        "e13": "sixbead",
        "e12-1": "sixbead",
        "e12-2": "sixbead",
        "e13-1": "sixbead",
        "e13-2": "sixbead",
        "e14": "sixbead",
        "e17": "sixbead",
        "la": "sixbead",
        "lb": "sixbead",
        "lc": "sixbead",
        "ld": "sixbead",
        "e16": "twoarm",
        "e18": "twoarm",
        "l1": "twoarm",
        "l2": "twoarm",
        "l3": "twoarm",
    }

    pairs_to_predict = [
        # large, small.
        (("la", "l1"), (2,)),
        (("lb", "l1"), (2,)),
        (("lc", "l1"), (2,)),
        (("ld", "l1"), (2,)),
        (("la", "l2"), (2,)),
        (("lb", "l2"), (2,)),
        (("lc", "l2"), (2,)),
        (("ld", "l2"), (2,)),
        (("la", "l3"), (2,)),
        (("lb", "l3"), (2,)),
        (("lc", "l3"), (2,)),
        (("ld", "l3"), (2,)),
        (("e10", "e16"), (2,)),
        (("e17", "e16"), (2,)),
        (("e17", "e10"), (2,)),
        (("e10", "e11"), (2,)),
        (("e14", "e16"), (2,)),
        (("e14", "e18"), (2,)),
        (("e10", "e18"), (2,)),
        (("e10", "e12"), (2,)),
        (("e14", "e11"), (2,)),
        (("e14", "e12"), (2,)),
        (("e13", "e11"), (2,)),
        (("e13-1", "e11"), (2,)),
        (("e13-2", "e11"), (2,)),
        (("e13", "e12"), (2,)),
        (("e13", "e12-1"), (2,)),
        (("e13", "e12-2"), (2,)),
        (("e13-1", "e12"), (2,)),
        (("e13-2", "e12"), (2,)),
        (("e14", "e13"), (2,)),
        (("e12", "e11"), (2,)),
    ]
    pairs = define_pairs(pairs_to_predict, ligand_types)

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
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_l_m[2] * multiplier,
                        large_bb: stoichiometry_l_l_m[0] * multiplier,
                        small_bb: stoichiometry_l_l_m[1] * multiplier,
                    },
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
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
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
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
        filename="mgen_3.png",
        pairs=pairs_to_predict,
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
        pairs=pairs_to_predict,
        structure_dir=structure_dir,
    )
    parity_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_8.png",
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
    deviation_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_9.png",
    )


def main() -> None:
    """Run script."""
    args = _parse_args()

    case_study_1(args.run)


if __name__ == "__main__":
    main()
