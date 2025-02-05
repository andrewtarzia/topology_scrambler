"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stko
from openmm import OpenMMException
from rdkit import RDLogger
from ufo_utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)
from utilities import eb_str, isomer_energy, pore_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: PLR0913
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: cgx.scram.Scrambler,
    topology_code: cgx.scram.TopologyCode,
    bb_config: cgx.scram.BuildingBlockConfiguration,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "opt_pore_data" not in properties:
        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        (l1, l2, multiplier, topology_idx, mash_idx, bbconfig_name) = (
            name.split("_")
        )
        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield.get_forcefield_dictionary(),
                "energy_per_bb": cgx.utilities.get_energy_per_bb(
                    energy_decomposition=properties["energy_decomposition"],
                    number_building_blocks=iterator.get_num_building_blocks(),
                ),
                "l1": l1,
                "l2": l2,
                "num_components": num_components,
                "multiplier": multiplier,
                "topology_idx": topology_idx,
                "mash_idx": mash_idx,
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
                "bb_config_idx": bb_config.idx,
                "min_distance": (
                    cgx.analysis.GeomMeasure().calculate_min_distance(
                        database.get_molecule(key=name)
                    )["min_distance"]
                ),
            },
        )


def make_plot(
    target_pair: str,
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"
        if pair != target_pair:
            continue

        energy = entry.properties["energy_per_bb"]
        min_distance = entry.properties["min_distance"]

        if multi not in energies:
            energies[multi] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[multi].append((round(energy, 4), entry.key, min_distance))

    with (figure_dir / f"min_{pair}.txt").open("w") as f:
        for multi in energies:
            if len(energies[multi]) == 0:
                continue

            sorted_energies = sorted(energies[multi], key=lambda p: p[0])
            min_energy = sorted_energies[0]

            sorted_pores = sorted(
                energies[multi], key=lambda p: p[2], reverse=True
            )
            max_pore = sorted_pores[0]

            offset = 20 * int(multi)
            bbox = {"boxstyle": "round", "fc": "1.0"}
            arrowprops = {
                "arrowstyle": "->",
                "connectionstyle": "angle,angleA=0,angleB=90,rad=10",
            }
            ax.annotate(
                text=f"E: {round(min_energy[0], 3)} @ {min_energy[1]}",
                xy=(min_energy[2], min_energy[0]),
                xycoords="data",
                xytext=(-0.5 * offset, -offset),
                textcoords="offset points",
                bbox=bbox,
                arrowprops=arrowprops,
                color=cmap[multi],
                fontsize=8,
            )
            offset = -20 * int(multi)
            ax.annotate(
                text=f"P: {round(max_pore[2], 3)} @ {max_pore[1]}",
                xy=(max_pore[2], max_pore[0]),
                xycoords="data",
                xytext=(0.5 * offset, -offset),
                textcoords="offset points",
                bbox=bbox,
                arrowprops=arrowprops,
                color=cmap[multi],
                fontsize=8,
            )

            ax.scatter(
                [i[2] for i in energies[multi]],
                [i[0] for i in energies[multi]],
                marker="o",
                c=cmap[multi],
                s=20,
                ec="none",
                alpha=0.3,
                label=f"M={multi}",
            )
            ax.scatter(
                min_energy[2],
                min_energy[0],
                marker="o",
                c=cmap[multi],
                s=20,
                ec="k",
            )
            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(pore_str(), fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.axhline(y=isomer_energy(), c="k", ls="--")
    ax.set_ylim(None, 1000)
    ax.legend(ncols=1, fontsize=16)
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
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}

    xs = ["1", "2", "3", "4"]
    ys = ["lf_ls1"]

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), tidx, bidx))

    vmin = 0
    vmax = 1
    for pair, multi in energies:
        sorted_energies = sorted(energies[(pair, multi)], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = ys.index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )
        ax.text(
            x=x,
            y=y,
            s=min_energy[1],
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="w" if min_energy[0] < 0.5 else "k",  # noqa: PLR2004
            fontsize=16,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels(ys)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"4:2:3 {eb_str()}", fontsize=16)

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
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    systems = {
        ("lf_ls1", "1"): {"name": "lf-ls1-1", "data": []},
        ("lf_ls1", "2"): {"name": "lf-ls1-2", "data": []},
        ("lf_ls1", "3"): {"name": "lf-ls1-3", "data": []},
        ("lf_ls1", "4"): {"name": "lf-ls1-4", "data": []},
    }

    count = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"

        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in systems:
            continue
        energy = entry.properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        systems[(pair, multi)]["data"].append(energy)
        count += 1

    logging.info("structures built, %s", count)
    rng = np.random.default_rng(seed=2)

    for i, (pair, multi) in enumerate(systems):
        if len(systems[(pair, multi)]["data"]) == 0:
            continue
        min_energy = min(systems[(pair, multi)]["data"])

        ax.scatter(
            [
                i + (2 * rng.random() - 1) * 0.3
                for j in range(len(systems[(pair, multi)]["data"]))
            ],
            systems[(pair, multi)]["data"],
            c="tab:blue",
            alpha=0.1,
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

    ax.axvline(x=3 + 0.5, c="gray")
    ax.axvline(x=6 + 0.5, c="gray")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
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


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

    stages = (
        "opt1",
        "nx00",
        "nx10",
        "nx20",
        "nx30",
        "shifted",
        "smd",
        "nx01",
        "nx11",
        "nx21",
        "nx31",
        "nx02",
        "nx12",
        "nx22",
        "nx32",
    )
    mash_ids = ("0", "1", "2", "3")
    sources = {i: 0 for i in stages}
    mashes = {i: 0 for i in mash_ids}
    lowe_sources = {i: 0 for i in stages}  # Produces low energy structures.
    lowe_mashes = {i: 0 for i in mash_ids}  # Produces low energy structures.
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        sources[entry.properties["source"]] += 1
        mashes[entry.properties["mash_idx"]] += 1
        energy = entry.properties["energy_per_bb"]
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1
            lowe_mashes[entry.properties["mash_idx"]] += 1

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
        [int(i) for i in mash_ids],
        [lowe_mashes[i] for i in mash_ids],
        color="#086788",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 1.0",
    )
    ax1.bar(
        [int(i) for i in mash_ids],
        [mashes[i] for i in mash_ids],
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
    ax.set_yscale("log")

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylabel("count", fontsize=16)  # , color=color)
    ax1.set_xticks(range(len(mash_ids)))
    ax1.set_xticklabels(mash_ids)
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


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Run script."""
    args = _parse_args()
    raise SystemExit("Change paths")
    raise SystemExit("rerun")
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "ufo_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "ufo_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "ufo_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "ufo_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "ufo.db"

    ligand_measures = {
        # From prep.
        "lf": {"dd": 8.0, "de": 4.3, "dde": 133, "eg": 1.4, "gb": 1.4},
        # From optl.
        "ls1": {"ba": 2.8, "aa": 4.9, "bac": 150, "bacab": 180},
        "ls2": {"ba": 2.8, "aa": 4.7, "bac": 155, "bacab": 180},
        "ls3": {"ba": 2.8, "aa": 5.0, "bac": 145, "bacab": 180},
        "ls4": {"ba": 2.8, "aa": 5.0, "bac": 150, "bacab": 180},
        "ls5": {"ba": 2.8, "aa": 5.0, "bac": 150, "bacab": 180},
        "ls7": {"ba": 2.8, "aa": 5.0, "bac": 150, "bacab": 180},
        "ls8": {"ba": 2.8, "aa": 5.0, "bac": 150, "bacab": 180},
        "ls9": {"ba": 2.8, "aa": 5.3, "bac": 165, "bacab": 180},
        "ls10": {"ba": 2.8, "aa": 5.4, "bac": 167, "bacab": 180},
    }

    l2s = ("ls1", "ls2", "ls3", "ls4", "ls5", "ls7", "ls8", "ls9", "ls10")
    pairs = {}
    for l2 in l2s:
        name = f"lf_{l2}"
        pairs[name] = {
            "converging_name": "lf",
            "diverging_name": l2,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        }

    if args.run:
        for pair in pairs:
            converging_name = pairs[pair]["converging_name"]
            diverging_name = pairs[pair]["diverging_name"]
            if diverging_name in ("ls3", "ls4", "ls5", "ls7", "ls8"):
                logging.info("add sterics, or remove")
                continue
            converging = pairs[pair]["converging"]
            diverging = pairs[pair]["diverging"]
            tetra = pairs[pair]["tetra"]

            forcefield = precursors_to_forcefield(
                pair=pair,
                diverging=diverging,
                converging=converging,
                conv_meas=ligand_measures[converging_name],
                dive_meas=ligand_measures[diverging_name],
            )

            converging_name = (
                f"{converging.get_name()}_f{forcefield.get_identifier()}"
            )
            converging_bb = cgx.utilities.optimise_ligand(
                molecule=converging.get_building_block(),
                name=converging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            converging_bb.write(
                str(ligand_dir / f"{converging_name}_optl.mol")
            )
            converging_bb = converging_bb.clone()

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

            diverging_name = (
                f"{diverging.get_name()}_f{forcefield.get_identifier()}"
            )
            diverging_bb = cgx.utilities.optimise_ligand(
                molecule=diverging.get_building_block(),
                name=diverging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            diverging_bb.write(str(ligand_dir / f"{diverging_name}_optl.mol"))
            diverging_bb = diverging_bb.clone()

            for multiplier in pairs[pair]["multipliers"]:
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: pairs[pair]["stoichiometry_L_L_M"][2]
                        * multiplier,
                        diverging_bb: pairs[pair]["stoichiometry_L_L_M"][0]
                        * multiplier,
                        converging_bb: pairs[pair]["stoichiometry_L_L_M"][1]
                        * multiplier,
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

                logging.info(
                    "producing between %s and %s structures",
                    len(possible_bbdicts) * iterator.count_graphs() * 1,
                    len(possible_bbdicts) * iterator.count_graphs() * 4,
                )

                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts, enumerate(iterator.yield_graphs())
                ):
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
                        constructed_molecule = (
                            cgx.scram.try_except_construction(
                                iterator=iterator,
                                topology_code=topology_code,
                                building_block_configuration=bb_config,
                                vertex_positions=vertex_positions,
                            )
                        )
                        name = (
                            f"{pair}_{multiplier}_{idx}_{mash_idx}"
                            f"_b{bb_config.idx}"
                        )

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
                                conformer.molecule.with_centroid(
                                    (0, 0, 0)
                                ).write(
                                    str(structure_dir / f"{name}_optc.mol")
                                )

                            analyse_cage(
                                database_path=database_path,
                                name=name,
                                forcefield=forcefield,
                                iterator=iterator,
                                topology_code=topology_code,
                                bb_config=bb_config,
                            )

                        except OpenMMException:
                            logging.info("failed optimisation of %s", name)

            make_plot(
                database_path=database_path,
                target_pair=pair,
                structure_dir=structure_dir,
                figure_dir=figure_dir,
                filename=f"ufo_1_{pair}.png",
            )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufo_3.png",
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufo_4.png",
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufo_5.png",
    )

    for pair in pairs:
        make_plot(
            database_path=database_path,
            target_pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"ufo_1_{pair}.png",
        )


if __name__ == "__main__":
    main()
