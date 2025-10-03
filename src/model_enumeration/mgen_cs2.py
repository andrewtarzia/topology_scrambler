"""Script to generate and optimise CG models."""

import argparse
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
    abead_d,
    analyse_cage,
    binder_bead,
    cbead_d,
    ff_opt_plot,
    get_regraphed_molecule,
    get_vertexset_molecule,
    optimise_cage,
    target_optimisation,
    trigonal_bead,
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
attempts = (
    None,
    "regraphed-spring-10",
    "regraphed-kamada-10",
    "set-kamada-10",
    "set-spring-10",
    "set-spectral-10",
)


def make_topt_plot_2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: dict[str, dict[str, tuple | int]],
    ffopt_targets: dict[str, str],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for pair in pairs:
        if "b" in pair:
            continue
        try:
            mix_target = ffopt_targets[pair]

        except KeyError:
            continue

        try:
            entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
                mix_target
            )
        except RuntimeError:
            continue

        if "optimisation_energy_per_bb" not in entry.properties:
            continue
        d_energy = (
            entry.properties["optimisation_energy_per_bb"]
            - entry.properties["energy_per_bb"]
        )
        if d_energy > 0:
            logging.info("%s has energy change > 0", entry.key)

        term_dict = {
            term: entry.properties["optimisation_x"][int(i)]
            for i, term in entry.properties["optimisation_map"].items()
        }

        ffdict = entry.properties["forcefield_dict"]["v_dict"]
        init_term_dict = {
            term: ffdict["_".join(list(term))] for term in term_dict
        }

        orig = [val for i, val in init_term_dict.items()]
        new = [val for i, val in term_dict.items()]

        c = "tab:blue" if pair[1] in ("1", "2") else "tab:orange"

        ax.scatter(
            sum(
                [
                    abs((j - i) / i) * 100
                    for i, j in zip(orig, new, strict=True)
                ]
            ),
            d_energy,
            c=c,
            alpha=1,
            ec="k",
            s=120,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("sum relative change in FF terms", fontsize=16)
    ax.set_ylabel(rf"$\Delta$ {eb_str()}", fontsize=16)

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


def make_topt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    mixtures: dict[str, dict[str, tuple | int]],
) -> dict:
    """Visualise energies."""
    modifiable = ["nb", "bnb", "bac", "ba", "ac"]
    units = [
        r"$\mathrm{\AA}$",
        "$^\\circ$",
        "$^\\circ$",
        r"$\mathrm{\AA}$",
        r"$\mathrm{\AA}$",
    ]
    fig, axs = plt.subplots(
        ncols=len(modifiable),
        sharey=True,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()
    ds = [[] for i in modifiable]
    for mix in mixtures:
        if "cc" in mix:
            continue

        try:
            mix_target = mixtures[mix]["target"]

        except KeyError:
            continue

        entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
            mix_target
        )

        if entry.properties["multiplier"] != 1:
            continue

        if "optimisation_energy_per_bb" not in entry.properties:
            raise RuntimeError

        term_dict = {
            term: entry.properties["optimisation_x"][int(i)]
            for i, term in entry.properties["optimisation_map"].items()
        }

        ffdict = entry.properties["forcefield_dict"]["v_dict"]
        init_term_dict = {
            term: ffdict["_".join(list(term))] for term in term_dict
        }

        orig = [val for i, val in init_term_dict.items()]
        new = [val for i, val in term_dict.items()]
        for i, ax in enumerate(flat_axs):
            ax.scatter(
                new[i],
                entry.properties["optimisation_energy_per_bb"],
                c="tab:blue",
                alpha=1,
                ec="k",
                s=80,
            )
            ax.plot(
                (orig[i], new[i]),
                (
                    entry.properties["optimisation_energy_per_bb"],
                    entry.properties["optimisation_energy_per_bb"],
                ),
                c="k",
                alpha=1,
                lw=1,
                zorder=-2,
                marker="s",
                markersize=3,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel(f"${modifiable[i]}$ [{units[i]}]", fontsize=16)
            ds[i].append(new[i] - orig[i])

            ax.set_yscale("log")
            if i == 0:
                ax.set_ylabel(f"opt. {eb_str()}", fontsize=16)

    for i, ax in enumerate(flat_axs):
        ax.set_title(
            rf"avg. $|\Delta|$={round(np.mean(ds[i]), 2)} {units[i]}",
            fontsize=16,
        )

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


def study_2_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(16, 5))

    multis = {
        1: (multi_cmap["1"], -0.2),
        2: (multi_cmap["2"], 0.0),
        3: (multi_cmap["3"], 0.2),
    }

    xs = {}
    lbls = set()
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        if "cc" in entry.key:
            continue

        if entry.properties["mix"] not in xs:
            xs[entry.properties["mix"]] = len(xs)

        x = xs[entry.properties["mix"]]
        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]

        lbl = multi
        ax.scatter(
            x + multis[multi][1],
            y,
            c=multis[multi][0],
            alpha=1,
            ec="k",
            s=80,
            label=lbl if lbl not in lbls else None,
        )
        lbls.add(lbl)
        logging.info(
            "E for %s is %s",
            entry.key,
            round(entry.properties["energy_per_bb"], 2),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_yscale("log")
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.legend(fontsize=16)
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=90)

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


def study_2_cc_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(16, 5))

    multis = {
        1: multi_cmap["1"],
        2: multi_cmap["2"],
        3: multi_cmap["3"],
        4: multi_cmap["4"],
    }

    xs = {}
    lines = [[], []]
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        if "cc" not in entry.key:
            continue

        xstr = (entry.key.split("_")[1], entry.key.split("_")[2])
        if xstr not in xs:
            xs[xstr] = len(xs)

        x = xs[xstr]
        m = "o" if "cc3" in entry.key else "s"

        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]
        logging.info("key %s has E %s", entry.key, y)
        if "cc3" in entry.key:
            lines[0].append((x, y))
        elif "cc20" in entry.key:
            lines[1].append((x, y))
        ax.scatter(
            x,
            y,
            c=multis[multi],
            alpha=1,
            ec="k",
            s=80,
            marker=m,
            label=None if xstr != ("1", "1") else entry.key.split("_")[0],
        )

    ax.plot(
        [i[0] for i in lines[0]],
        [i[1] for i in lines[0]],
        color="k",
        zorder=-2,
    )
    ax.plot(
        [i[0] for i in lines[1]],
        [i[1] for i in lines[1]],
        color="k",
        zorder=-2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_yscale("log")
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(
        [f"{i[0]},{i[1]}" for i in xs],
        fontsize=16,
        rotation=90,
    )
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


def study_2_plot_2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(16, 5))

    multis = {1: (multi_cmap["1"], -0.2), 2: (multi_cmap["2"], 0.2)}

    xs = {}
    lbls = set()
    mix_mins = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        if "cc" in entry.key:
            continue

        if entry.properties["mix"] not in xs:
            xs[entry.properties["mix"]] = len(xs)
            mix_mins[entry.properties["mix"]] = {
                i: (float("inf"), None) for i in multis
            }

        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]

        if y < mix_mins[entry.properties["mix"]][multi][0]:
            mix_mins[entry.properties["mix"]][multi] = (y, entry.key)

    for mix, mdict in mix_mins.items():
        for multi, (y, key) in mdict.items():
            lbl = multi
            if key is None:
                continue
            p = ax.bar(
                xs[mix] + multis[multi][1],
                y,
                fc=multis[multi][0],
                width=0.3,
                ec="k",
                label=lbl if lbl not in lbls else None,
            )
            padding = 0
            ltype = "center"
            string = key.split("_")
            string = "t: " + string[2] + f" ({string[3]})"
            ax.bar_label(
                p,
                labels=[string],
                rotation=90,
                label_type=ltype,
                padding=padding,
                fontsize=12,
            )
            lbls.add(lbl)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.legend(fontsize=16)
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=90)
    ax.set_yscale("log")

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


def study_2_plot_3(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 2))

    l_positions = {
        "cs6l1": 3.8,
        "cs6l1b": 3.8,
        "cs6l2": 5.7,
        "cs6l2b": 5.7,
        "cs6l5": 8.0,
        "cs6l5b": 8.0,
        "cs6l6": 9.9,
        "cs6l6b": 9.9,
        "cs6l9": 14.2,
        "cs6l9b": 14.2,
    }
    x_positions = {
        "cs6l1": -0.2,
        "cs6l1b": 0.2,
        "cs6l2": 1.8,
        "cs6l2b": 2.2,
        "cs6l5": 3.8,
        "cs6l5b": 4.2,
        "cs6l6": 5.8,
        "cs6l6b": 6.2,
        "cs6l9": 7.8,
        "cs6l9b": 8.2,
    }
    y_positions = {"cs6zr1": 0.1, "cs6zr2": 0.0}

    # create the new map
    cmap = plt.cm.Blues_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        mix, multi, idx, midx = entry.key.split("_")
        if multi == "1":
            yp = 0
            if idx != "0":
                continue
        elif multi == "2":
            yp = 0.3
            if idx != "4":
                continue
        if entry.properties["di"] not in x_positions:
            continue
        x = x_positions[entry.properties["di"]]
        y = y_positions[entry.properties["tri"]] + yp
        c = entry.properties["energy_per_bb"]

        ax.scatter(
            x,
            y,
            c=c,
            marker="s",
            s=200,
            edgecolor="k",
            cmap=cmap,
            norm=norm,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([x_positions[i] + 0.2 for i in x_positions if "b" not in i])
    ax.set_xticklabels(
        [l_positions[i] for i in x_positions if "b" not in i], fontsize=16
    )
    ax.set_yticks(
        [y_positions[i] for i in y_positions]
        + [y_positions[i] + 0.3 for i in y_positions]
    )
    ax.set_yticklabels(list(y_positions) * 2, fontsize=16)
    ax.set_ylim(-0.1, 0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(eb_str(), fontsize=16)

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


def study_2_plot_4(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if "cc" in entry.key:
            continue

        e = entry.properties["energy_per_bb"]
        opte = entry.properties.get("optimisation_energy_per_bb", None)
        m = entry.properties["multiplier"]

        if opte is not None:
            ax.scatter(
                e,
                opte,
                c=multi_cmap[str(m)],
                ec="k",
                s=120,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"input {eb_str()}", fontsize=16)
    ax.set_ylabel(f"opt-ff {eb_str()}", fontsize=16)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.plot((0, 20), (0, 20), c="k", ls="--", zorder=-2)
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


def make_summary_plot2(  # noqa: C901, PLR0912
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
    parser.add_argument(
        "--opt_ff",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def case_study_2(run: bool, opt_ff: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run case study 2 studying Tri + Di homoleptic systems."""
    wd = pathlib.Path("/home/atarzia/onbear/tarziaa-cgx1/model_enum_data/")
    calculation_dir = wd / "mgencs2_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs2_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs2_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs2_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs2"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs2.db"

    present_beads = (cbead_d, abead_d, binder_bead, trigonal_bead)
    stoichiometry_t_d = (2, 3)
    vdw_cutoff = 2
    multipliers = (1, 2, 3, 4)
    # Very approximate.
    ligand_measures = {
        "cs6l1": {"ba": 3.8 / 3, "aa": 3.8 / 3, "bac": 180},
        "cs6l1b": {"ba": 3.8 / 3, "aa": 3.8 / 3, "bac": 160},
        "cs6l2": {"ba": 5.7 / 3, "aa": 5.7 / 3, "bac": 180},
        "cs6l2b": {"ba": 5.7 / 3, "aa": 5.7 / 3, "bac": 160},
        "cs6l5": {"ba": 8.0 / 3, "aa": 8.0 / 3, "bac": 180},
        "cs6l5b": {"ba": 8.0 / 3, "aa": 8.0 / 3, "bac": 160},
        "cs6l6": {"ba": 9.9 / 3, "aa": 9.9 / 3, "bac": 180},
        "cs6l6b": {"ba": 9.9 / 3, "aa": 9.9 / 3, "bac": 160},
        "cs6l9": {"ba": 14.2 / 3, "aa": 14.2 / 3, "bac": 180},
        "cs6l9b": {"ba": 14.2 / 3, "aa": 14.2 / 3, "bac": 160},
        "cs6zr1": {"bnb": 60, "nb": 3.5},
        "cs6zr2": {"bnb": 70, "nb": 3.5},
        "cs6cc31": {"bnb": 120, "nb": 2.9},
        "cs6cc32": {"ba": 1.5, "aa": 1.5, "bac": 115},
        "cs6cc201": {"bnb": 120, "nb": 2.9},
        "cs6cc202": {"ba": 1.5, "aa": 2.5, "bac": 145},
    }

    mixtures = {
        "l1zr1": {
            "linear": (
                "cs6l1",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1zr1_1_0_3",
        },
        "l1zr2": {
            "linear": (
                "cs6l1",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1zr2_1_0_2",
        },
        "l2zr1": {
            "linear": (
                "cs6l2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2zr1_1_0_3",
        },
        "l2zr2": {
            "linear": (
                "cs6l2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2zr2_1_0_3",
        },
        "l5zr1": {
            "linear": (
                "cs6l5",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5zr1_1_0_2",
        },
        "l5zr2": {
            "linear": (
                "cs6l5",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5zr2_1_0_2",
        },
        "l6zr1": {
            "linear": (
                "cs6l6",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6zr1_1_0_2",
        },
        "l6zr2": {
            "linear": (
                "cs6l6",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6zr2_1_0_2",
        },
        "l9zr1": {
            "linear": (
                "cs6l9",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9zr1_1_0_1",
        },
        "l9zr2": {
            "linear": (
                "cs6l9",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9zr2_1_0_3",
        },
        "cc3": {
            "linear": (
                "cs6cc32",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6cc31",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "cc3_2_4_3",
        },
        "cc20": {
            "linear": (
                "cs6cc202",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6cc201",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "cc20_4_25_2",
        },
        "l1bzr1": {
            "linear": (
                "cs6l1b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1bzr1_1_0_4",
        },
        "l1bzr2": {
            "linear": (
                "cs6l1b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1bzr2_1_0_2",
        },
        "l2bzr1": {
            "linear": (
                "cs6l2b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2bzr1_1_0_2",
        },
        "l2bzr2": {
            "linear": (
                "cs6l2b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2bzr2_1_0_2",
        },
        "l5bzr1": {
            "linear": (
                "cs6l5b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5bzr1_1_0_0",
        },
        "l5bzr2": {
            "linear": (
                "cs6l5b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5bzr2_1_0_2",
        },
        "l6bzr1": {
            "linear": (
                "cs6l6b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6bzr1_1_0_2",
        },
        "l6bzr2": {
            "linear": (
                "cs6l6b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6bzr2_1_0_2",
        },
        "l9bzr1": {
            "linear": (
                "cs6l9b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9bzr1_1_0_5",
        },
        "l9bzr2": {
            "linear": (
                "cs6l9b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9bzr2_1_0_2",
        },
    }

    cg_scale = 2
    for mix, mdict in mixtures.items():
        cgx.molecular.BeadLibrary(present_beads)
        linear_name, linear = mdict["linear"]
        trigonal_name, trigonal = mdict["trigonal"]

        cs6_definer_dict = {
            # Trigonal.
            "nb": (
                "bond",
                ligand_measures[trigonal_name]["nb"] / cg_scale,
                1e5,
            ),
            "bnb": ("angle", ligand_measures[trigonal_name]["bnb"], 1e2),
            # Linear.
            "ba": (
                "bond",
                ligand_measures[linear_name]["ba"] / cg_scale,
                1e5,
            ),
            "ac": (
                "bond",
                ligand_measures[linear_name]["aa"] / 2 / cg_scale,
                1e5,
            ),
            "bac": (
                "angle",
                ligand_measures[linear_name]["bac"],
                1e2,
            ),
            "aca": ("angle", 180, 1e2),
            # Constant.
            "nba": ("angle", 180, 1e2),
            # Nonbondeds.
            "n": ("nb", 10.0, 1.0),
            "a": ("nb", 10.0, 1.0),
            "b": ("nb", 10.0, 1.0),
            "c": ("nb", 10.0, 1.0),
        }
        if "b" in linear_name:
            cs6_definer_dict["bacab"] = ("tors", "0134", 180, 50, 1)

        forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
            identifier=f"{mix}ff",
            prefix=f"{mix}ff",
            vdw_bond_cutoff=vdw_cutoff,
            present_beads=present_beads,
            definer_dict=cs6_definer_dict,
        )
        if run:
            linear_bb = cgx.utilities.optimise_ligand(
                molecule=linear.get_building_block(),
                name=f"{mix}_{linear.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            linear_bb.write(
                str(ligand_dir / f"{mix}_{linear.get_name()}_optl.mol")
            )

            trigonal_bb = cgx.utilities.optimise_ligand(
                molecule=trigonal.get_building_block(),
                name=trigonal.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            trigonal_bb.write(
                str(ligand_dir / f"{mix}_{trigonal.get_name()}_optl.mol")
            )

            for multiplier in multipliers:
                if multiplier > 2 and "cc" not in mix:  # noqa: PLR2004
                    continue
                logging.info("doing: mix %s, multi %s", mix, multiplier)

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        trigonal_bb: stoichiometry_t_d[0] * multiplier,
                        linear_bb: stoichiometry_t_d[1] * multiplier,
                    },
                    graph_type=f"{stoichiometry_t_d[0] * multiplier}"
                    f"P{stoichiometry_t_d[1] * multiplier}",
                    graph_set="rx",
                )
                logging.info(
                    "graph iteration has %s graphs", iterator.count_graphs()
                )

                for idx, topology_code in enumerate(iterator.yield_graphs()):
                    generated_conformers = []
                    for midx, scale in enumerate(attempts):
                        name = f"{mix}_{multiplier}_{idx}_{midx}"

                        try:
                            if isinstance(scale, str) and "regraphed" in scale:
                                constructed_molecule = get_regraphed_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=None,
                                )

                            else:
                                constructed_molecule = get_vertexset_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=None,
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
                                f"{mix}_{multiplier}_{idx}_{nmash_idx}"
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
                                    "tri": trigonal_name,
                                    "di": linear_name,
                                    "mix": mix,
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

        if opt_ff:
            ffoptcalculation_dir = calculation_dir / "ff_opt"
            ffoptcalculation_dir.mkdir(exist_ok=True)
            try:
                mix_target = mdict["target"]
            except KeyError:
                continue

            if "cc" in mix:
                # Do not include the tritopic BB in optimisation, otherwise
                # another minimum is found by changing those angles.
                modifiable = ["bac", "ba", "ac"]
            else:
                modifiable = ["nb", "bnb", "bac", "ba", "ac"]
            logging.info(
                "running optimisation of %s molecules over %s",
                mix_target,
                modifiable,
            )
            target_optimisation(
                database_path=database_path,
                target_key=mix_target,
                calculation_dir=ffoptcalculation_dir,
                definer_dict=cs6_definer_dict,
                modifiable_terms=modifiable,
                forcefield=forcefield,
            )

    study_2_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )
    study_2_cc_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_6.png",
    )
    study_2_plot_2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
    )
    study_2_plot_3(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
    )
    study_2_plot_4(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
    )
    for mix, mdict in mixtures.items():
        try:
            mix_target = mdict["target"]
            ff_opt_plot(
                database_path=database_path,
                target=mix,
                figure_dir=figure_dir,
                filename=f"mgen_5_{mix}.png",
                key_target=mix_target,
            )
        except KeyError:
            continue

    make_topt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_10.png",
        mixtures={i: mixtures[i] for i in mixtures if "cc" not in i},
    )
    make_topt_plot_2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_11.png",
        pairs={i: mixtures[i] for i in mixtures if "cc" not in i},
        ffopt_targets={
            i: mixtures[i]["target"] for i in mixtures if "cc" not in i
        },
    )


def main() -> None:
    """Run script."""
    args = _parse_args()
    case_study_2(args.run, args.opt_ff)


if __name__ == "__main__":
    main()
