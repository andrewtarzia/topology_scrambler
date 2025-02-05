"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
import time
import warnings
from collections import defaultdict

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from openmm import OpenMMException
from rdkit import RDLogger
from utilities import eb_str, isomer_energy, multi_cmap
from validation_utilities import (
    abead_d,
    analyse_cage,
    binder_bead,
    cbead_d,
    get_validation_forcefield,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    parser.add_argument(
        "--nodoubles",
        action="store_true",
        help="set to only study no double-walleds",
    )

    return parser.parse_args()


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = (
        "opt1",
        "smd",
        "shifted",
        "nx0",
        "nx1",
        "nx2",
        "nx3",
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
    )
    sources = {i: 0 for i in stages}
    # Produces low energy structures.
    lowe_sources = {i: 0 for i in stages}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        sources[entry.properties["source"]] += 1
        energy = entry.properties["energy_per_bb"]
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1

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

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)  # , color=color)

    ax.legend(fontsize=16)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    ax.set_xlabel("stage", fontsize=16)
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


def make_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    energies = defaultdict(list)
    bacs = defaultdict(list)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]

        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((bac_angle, energy, entry.key))
        bacs[bac_angle].append((multi, energy, entry.key))

    fig, (axx, ax) = plt.subplots(
        nrows=2,
        figsize=(8, 6),
        height_ratios=[1, 4],
        sharex=True,
    )

    countsx = {}
    for i, multi in enumerate(sorted([int(i) for i in energies])):
        instable = False
        idx = str(multi)
        min_energy_tuple = min(energies[idx], key=lambda p: p[1])

        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

            stable = [
                i
                for i in energies[idx]
                if i[0] == bac_angle and i[1] < isomer_energy()
            ]
            if len(stable) > 0:
                logging.info("stable cages: %s", stable)
                instable = True
            if bac_angle not in countsx:
                countsx[bac_angle] = 0
            countsx[bac_angle] += len(stable)

        ax.plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=multi_cmap[str(multi)],
            ls="-",
            marker="o",
            markersize=4,
            alpha=1.0,
            zorder=2,
            label=(
                f"M{idx}: {min_energy_tuple[0]}, {min_energy_tuple[2]}"
                if instable
                else f"M{idx}"
            ),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_yscale("log")
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.set_ylabel(eb_str(), fontsize=16)

    axx.plot(
        list(countsx),
        [countsx[i] for i in countsx],
        c="k",
        marker="o",
        mec="k",
        zorder=2,
        markersize=6.0,
    )
    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_ylabel("stable", fontsize=16)
    axx.set_ylim(0, None)

    leg = ax.legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_xlabel("target $bac$ angle [$^\\circ$]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_timings_plot(
    figure_dir: pathlib.Path,
    timing_file: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    timings = pl.read_csv(
        timing_file,
        has_header=False,
        new_columns=[
            "lig",
            "multi",
            "num_vertices",
            "idx",
            "mash_idx",
            "opt-time/s",
            "ana-time/s",
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        timings["num_vertices"],
        timings["opt-time/s"],
        c="tab:blue",
        s=100,
        alpha=1,
        label="optimisation time",
        ec="k",
    )
    ax.scatter(
        timings["num_vertices"],
        timings["ana-time/s"],
        c="tab:orange",
        s=100,
        alpha=1,
        label="analysis time",
        ec="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. vertices", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_summary_plot(
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = defaultdict(list)

    tops = figure_dir / "ivalidation_tops.txt"
    if tops.exists():
        tops.unlink()

    to_save = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        bite_angle = (bac_angle - 90) * 2

        vstr = entry.key.split("_")[2]

        if entry.properties["num_components"] > 1:
            continue
        energies[(multi, bite_angle)].append((round(energy, 4), vstr))
        if energy < isomer_energy():
            to_save.append(entry.key)

    with tops.open("w") as f:
        for ts in sorted(to_save):
            file_ = structure_dir / f"{ts}_optc.mol"
            if file_.exists():
                f.write(f"{file_} ")

    vmin = 0
    vmax = 1
    for multi, bite_angle in energies:
        sorted_energies = sorted(
            energies[(multi, bite_angle)], key=lambda p: p[0]
        )
        min_energy = sorted_energies[0]

        x = int(bite_angle)
        y = int(multi)

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
            fontsize=6,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("target bite angle [deg]", fontsize=16)
    ax.set_ylabel("multiplier", fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
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


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    args = _parse_args()
    raise SystemExit("Change paths")
    raise SystemExit("rerun")
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures"
    if not args.nodoubles:
        calculation_dir = wd / "ivalidation_calculations"
        structure_dir = wd / "ivalidation_structures"
        ligand_dir = wd / "ivalidation_ligands"
        data_dir = wd / "ivalidation_data"
        database_path = data_dir / "ivalidation_run.db"
        timing_file = data_dir / "ivalidation_times.csv"
        max_num = 1000

    else:
        calculation_dir = wd / "dvalidation_calculations"
        structure_dir = wd / "dvalidation_structures"
        ligand_dir = wd / "dvalidation_ligands"
        data_dir = wd / "dvalidation_data"
        database_path = data_dir / "dvalidation_run.db"
        timing_file = data_dir / "dvalidation_times.csv"
        max_num = 1000

    calculation_dir.mkdir(exist_ok=True)
    structure_dir.mkdir(exist_ok=True)
    ligand_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    ligands = {
        str(bac_angle): {
            "forcefield": get_validation_forcefield(
                bac_angle=bac_angle,
                identifier=str(i),
            ),
            "stoichiometry_L_M": (2, 1),
            "ditopic": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        }
        for i, bac_angle in enumerate(range(90, 181, 5))
    }

    if args.run:
        for lig in ligands:
            forcefield = ligands[lig]["forcefield"]
            ditopic = ligands[lig]["ditopic"]
            tetra = ligands[lig]["tetra"]

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

            for multiplier in ligands[lig]["multipliers"]:
                if int(lig) < 90 and multiplier > 8:  # noqa: PLR2004
                    continue
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: ligands[lig]["stoichiometry_L_M"][1]
                        * multiplier,
                        ditopic_bb: ligands[lig]["stoichiometry_L_M"][0]
                        * multiplier,
                    },
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
                    graph_set="rx_nodoubles" if args.nodoubles else "rx",
                )

                logging.info("doing: ligand %s, multi %s", lig, multiplier)
                for idx, topology_code in enumerate(iterator.yield_graphs()):
                    if max_num is not None and idx > max_num:
                        break

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
                                vertex_positions=vertex_positions,
                            )
                        )

                        name = f"{lig}_{multiplier}_{idx}_{mash_idx}"
                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )
                        num_vertices = iterator.get_num_building_blocks()

                        # Optimise and save.
                        logging.info("building %s", name)

                        try:
                            st1 = time.time()
                            conformer = opt_function(
                                molecule=constructed_molecule,
                                name=name,
                                output_dir=calculation_dir,
                                forcefield=forcefield,
                                platform=None,
                                database_path=database_path,
                            )
                            et1 = time.time()
                            if conformer is not None:
                                conformer.molecule.with_centroid(
                                    (0, 0, 0)
                                ).write(
                                    str(structure_dir / f"{name}_optc.mol")
                                )

                            st2 = time.time()
                            analyse_cage(
                                database_path=database_path,
                                name=name,
                                forcefield=forcefield,
                                iterator=iterator,
                                topology_code=topology_code,
                            )
                            et2 = time.time()

                            with timing_file.open("a") as f:
                                f.write(
                                    f"{lig},{multiplier},{num_vertices},{idx},"
                                    f"{mash_idx},{et1 - st1},{et2 - st2}\n"
                                )

                        except OpenMMException:
                            logging.info("failed optimisation of %s", name)

                make_plot(
                    database_path=database_path,
                    figure_dir=figure_dir,
                    filename="validationd_1.png"
                    if args.nodoubles
                    else "validationi_1.png",
                )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="validationd_1.png"
        if args.nodoubles
        else "validationi_1.png",
    )

    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="validationd_5.png"
        if args.nodoubles
        else "validationi_5.png",
    )

    make_summary_plot(
        database_path=database_path,
        structure_dir=structure_dir,
        figure_dir=figure_dir,
        filename="validationd_2.png"
        if args.nodoubles
        else "validationi_2.png",
    )

    make_timings_plot(
        timing_file=timing_file,
        figure_dir=figure_dir,
        filename="validationd_times.png"
        if args.nodoubles
        else "validation_times.png",
    )


if __name__ == "__main__":
    main()
