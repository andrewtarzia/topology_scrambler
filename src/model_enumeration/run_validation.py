"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
import time
import warnings
from collections import defaultdict

import cgexplore
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from openmm import OpenMMException
from rdkit import RDLogger
from utilities import eb_str, multi_cmap
from validation_utilities import (
    abead_d,
    analyse_cage,
    binder_bead,
    cbead_d,
    get_validation_forcefield,
    tetra_bead,
)

import scram

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


def make_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    energies = defaultdict(list)
    bacs = defaultdict(list)
    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]

        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((bac_angle, energy, entry.key))
        bacs[bac_angle].append((multi, energy, entry.key))

    fig, ax = plt.subplots(figsize=(8, 5))
    axx = ax.twinx()
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
                if i[0] == bac_angle and i[1] < 0.3  # noqa: PLR2004
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
        ax.axhline(y=0.3, c="k", ls="--")

    axx.plot(
        list(countsx),
        [countsx[i] for i in countsx],
        c="gray",
        marker="D",
        mec="k",
        zorder=2,
        markersize=3.0,
    )
    axx.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        labelcolor="gray",
    )
    ax.set_ylabel(eb_str(), fontsize=16)
    axx.set_ylabel("num. stable structures", fontsize=16, color="gray")
    axx.set_ylim(0, None)

    leg = ax.legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_xlabel("target $bac$ angle [deg]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_parity_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    full_database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    db = cgexplore.utilities.AtomliteDatabase(database_path)
    full_db = cgexplore.utilities.AtomliteDatabase(full_database_path)

    energies = defaultdict(list)
    for entry in db.get_entries():
        if not full_db.has_molecule(entry.key):
            continue

        multi = entry.properties["multiplier"]
        energy1 = entry.properties["energy_per_bb"]
        energy2 = full_db.get_entry(entry.key).properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((energy1, energy2))

    fig, ax = plt.subplots(figsize=(5, 5))

    for multi in energies:
        ax.scatter(
            [i[0] for i in energies[multi]],
            [i[1] for i in energies[multi]],
            c=multi_cmap[str(multi)],
            ec="k",
            alpha=1.0,
            zorder=2,
            s=20,
            label=f"M{multi}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(y=0.3, c="k", ls="--")
    ax.axvline(x=0.3, c="k", ls="--")
    ax.plot((0.001, 100), (0.001, 100), c="gray", ls="-", zorder=-1)

    ax.set_xlabel(f"1st {eb_str()}", fontsize=16)
    ax.set_ylabel(f"2nd {eb_str()}", fontsize=16)

    leg = ax.legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

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
    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        bite_angle = (bac_angle - 90) * 2

        vstr = entry.key.split("_")[2]

        if entry.properties["num_components"] > 1:
            continue
        energies[(multi, bite_angle)].append((round(energy, 4), vstr))
        if energy < 0.3:  # noqa: PLR2004
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
        max_num = None

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
            "ditopic": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
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

            ditopic_bb = scram.toy.prepare_building_block(
                precursor=ditopic,
                forcefield=forcefield,
                calculation_dir=calculation_dir,
                ligand_dir=ligand_dir,
            )
            tetra_bb = scram.toy.prepare_building_block(
                precursor=tetra,
                forcefield=forcefield,
                calculation_dir=calculation_dir,
                ligand_dir=ligand_dir,
            )

            for multiplier in ligands[lig]["multipliers"]:
                if int(lig) < 90 and multiplier > 8:  # noqa: PLR2004
                    continue
                iterator = scram.topologies.IHomolepticTopologyIterator(
                    building_block_counts={
                        tetra_bb: ligands[lig]["stoichiometry_L_M"][1]
                        * multiplier,
                        ditopic_bb: ligands[lig]["stoichiometry_L_M"][0]
                        * multiplier,
                    },
                    graph_type=scram.topologies.get_graph_type(
                        stoichiometry=ligands[lig]["stoichiometry_L_M"],
                        multiplier=multiplier,
                    ),
                )

                logging.info("doing: ligand %s, multi %s", lig, multiplier)
                for idx, topology_code in enumerate(iterator.get_graphs()):
                    if args.nodoubles and topology_code.contains_doubles():
                        continue

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
                            opt_function = scram.toy.optimise_cage
                        else:
                            vertex_positions = None
                            opt_function = scram.toy.graph_optimise_cage

                        # Do the construction.
                        constructed_molecule = (
                            scram.toy.try_except_construction(
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
                                    f"{mash_idx},{et1-st1},{et2-st2}\n"
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

    make_summary_plot(
        database_path=database_path,
        structure_dir=structure_dir,
        figure_dir=figure_dir,
        filename="validationd_2.png"
        if args.nodoubles
        else "validationi_2.png",
    )
    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="validationd_1.png"
        if args.nodoubles
        else "validationi_1.png",
    )
    make_timings_plot(
        timing_file=timing_file,
        figure_dir=figure_dir,
        filename="validationd_times.png"
        if args.nodoubles
        else "validation_times.png",
    )
    if args.nodoubles:
        make_parity_plot(
            database_path=database_path,
            full_database_path=wd / "ivalidation_data" / "ivalidation_run.db",
            figure_dir=figure_dir,
            filename="validationd_3.png",
        )


if __name__ == "__main__":
    main()
