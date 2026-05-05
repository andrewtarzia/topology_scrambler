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
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    analyse_cage,
    define_pairs,
    get_regraphed_molecule,
    get_vertexset_molecule,
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


def study_4_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    cmap = {
        "cs3l1_cs3l1p": "tab:blue",
        "cs3l1_cs3l6p": "tab:orange",
        "cs3l2_cs3l1p": "tab:green",
        "cs3l2_cs3l6p": "tab:red",
    }

    xs = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        pair = entry.properties["pair"]
        string = entry.key.split("_")
        string = (
            string[0].split("cs3")[1]
            + "-"
            + string[1].split("cs3")[1]
            + "-t"
            + string[3]
            + "-"
            + string[5]
        )
        if ey < isomer_energy():
            xs[string] = len(xs)

            p = ax.bar(
                xs[string],
                ey,
                fc=cmap[pair],
                alpha=1.0,
                ec="k",
            )
            ax.bar_label(
                p,
                labels=[round(ey, 2)],
                rotation=90,
                label_type="edge",
                padding=8,
                fontsize=12,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylim(0, None)

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


def study_4_plot_5(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    cmap = {
        "cs3l1_cs3l1p": "tab:blue",
        "cs3l1_cs3l6p": "tab:orange",
        "cs3l2_cs3l1p": "tab:green",
        "cs3l2_cs3l6p": "tab:red",
    }
    xmap = {
        "cs3l1_cs3l1p": -0.3,
        "cs3l1_cs3l6p": -0.1,
        "cs3l2_cs3l1p": 0.1,
        "cs3l2_cs3l6p": 0.3,
    }

    xs = {"isomer A": 0, "isomer B": 1}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        pair = entry.properties["pair"]
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        if (tidx, bidx) == (2, 89):
            x = xs["isomer A"] + xmap[pair]
        elif (tidx, bidx) == (2, 112):
            x = xs["isomer B"] + xmap[pair]
        else:
            continue
        string = pair.split("_")
        string = string[0].split("cs3")[1] + "-" + string[1].split("cs3")[1]

        p = ax.bar(
            x,
            ey,
            width=0.1,
            fc=cmap[pair],
            alpha=1.0,
            ec="k",
        )
        ax.bar_label(
            p,
            labels=[f"{string}: {round(ey, 2)}"],
            rotation=90,
            label_type="edge",
            padding=8,
            fontsize=12,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylim(0, 1.2)

    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=0)
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


def study_4_plot_2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(16, 5))

    counts = {}
    counts_low = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        m = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if pair not in counts:
            counts[pair] = 0
            counts_low[pair] = 0

        if m == 6:  # noqa: PLR2004
            if ey < 1:
                counts[pair] += 1
            if ey < isomer_energy():
                counts_low[pair] += 1

        if pair != "cs3l1_cs3l1p":
            continue

        ax.scatter(
            m,
            ey,
            c="tab:blue",
            s=120,
            ec="k",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks([2, 3, 4, 5, 6])

    bar = ax2.bar(
        range(len(counts_low)),
        list(counts_low.values()),
        align="center",
        fc="tab:orange",
        label=f"{eb_str(no_unit=True)}<0.3",
    )
    ax2.bar_label(bar, fmt="%.f", fontsize=16)
    ax2.bar(
        range(len(counts)),
        list(counts.values()),
        align="center",
        fc="none",
        ec="k",
        label=f"{eb_str(no_unit=True)}<1.0",
    )

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_ylabel("count", fontsize=16)
    ax2.set_xticks(range(len(counts)), list(counts.keys()))
    ax2.legend(fontsize=16)

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
    """Run starship case study studying Rh heteroleptic systems."""
    run = _parse_args().run

    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")
    run_prefix = "rh_isomer"
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

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        "cs3l1": {"ba": 1.5, "aa": 7.9, "bac": 120},
        "cs3l2": {"ba": 1.5, "aa": 7.7, "bac": 120},
        "cs3l1p": {"ba": 1.5, "aa": 2.4, "bac": 150},
        "cs3l6p": {"ba": 1.5, "aa": 4.8, "bac": 150},
    }
    ligand_types = {
        "cs3l1": "twoarm",
        "cs3l2": "twoarm",
        "cs3l1p": "twoarm",
        "cs3l6p": "twoarm",
    }
    pairs_to_predict = [
        # large, small.
        (("cs3l1", "cs3l1p"), (2, 3, 4, 5, 6)),
        (("cs3l1", "cs3l6p"), (6,)),
        (("cs3l2", "cs3l1p"), (6,)),
        (("cs3l2", "cs3l6p"), (6,)),
    ]

    pairs = define_pairs(pairs_to_predict, ligand_types)

    cs3_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbe": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "ede": ("angle", 180, 1e2),
        # Torsions.
        "bacab": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }

    if run:
        for pair in pairs:
            forcefield = precursors_to_forcefield(
                pair=pair,
                large=pairs[pair]["large"],
                small=pairs[pair]["small"],
                large_meas=ligand_measures[pairs[pair]["large_name"]],
                small_meas=ligand_measures[pairs[pair]["small_name"]],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
                constant_definer_dict=cs3_definer_dict,
            )

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

                run_topology_codes = []
                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Filter graphs for 1-loops.
                    if contains_parallels(topology_code):
                        continue

                    # Testing bb-config aware graph check.
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

    study_4_plot_5(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rh_isomer_1.png",
    )
    study_4_plot_2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rh_isomer_2.png",
    )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rh_isomer_3.png",
        pairs=pairs_to_predict,
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rh_isomer_4.png",
        pairs=pairs_to_predict,
        structure_dir=structure_dir,
    )
    study_4_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rh_isomer_5.png",
    )


if __name__ == "__main__":
    main()
