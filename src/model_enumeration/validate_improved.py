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
import polars as pl
import stko
from openmm import OpenMMException
from rdkit import RDLogger
from utilities import abead_d, binder_bead, cbead_d, eb_str, tetra_bead
from validate_cg_model import (
    analyse_cage,
    get_validation_forcefield,
    make_plot,
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
        "--atomise",
        action="store_true",
        help="set to build atomistic structures",
    )
    return parser.parse_args()


def make_timings_plot(
    figure_dir: pathlib.Path,
    timing_file: pathlib.Path,
    graph_timing_file: pathlib.Path,
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

    graph_timings = pl.read_csv(
        graph_timing_file,
        has_header=False,
        new_columns=[
            "num_vertices",
            "algtype",
            "num_found",
            "gen-time/s",
        ],
    )

    nx_graph_timings = graph_timings.filter(pl.col("algtype") == "nx")
    rx_graph_timings = graph_timings.filter(pl.col("algtype") == "rx")
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(16, 5))
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
    ax.plot(
        nx_graph_timings["num_vertices"],
        nx_graph_timings["gen-time/s"],
        c="tab:green",
        marker="s",
        markersize=12,
        alpha=1,
        label="nx-graph time",
        mec="k",
    )
    ax.plot(
        rx_graph_timings["num_vertices"],
        rx_graph_timings["gen-time/s"],
        c="tab:purple",
        marker="o",
        markersize=10,
        alpha=1,
        label="rx-graph time",
        mec="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. vertices", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=16)

    ax1.plot(
        nx_graph_timings["num_vertices"],
        nx_graph_timings["num_found"],
        c="tab:green",
        marker="s",
        markersize=12,
        alpha=1,
        label="nx-graph time",
        mec="k",
    )
    ax1.plot(
        rx_graph_timings["num_vertices"],
        rx_graph_timings["num_found"],
        c="tab:purple",
        marker="o",
        markersize=10,
        alpha=1,
        label="rx-graph time",
        mec="k",
    )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("num. vertices", fontsize=16)
    ax1.set_ylabel("num. graphs found", fontsize=16)
    ax1.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
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
    energies = defaultdict(list)

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


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "ivalidation_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "ivalidation_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "ivalidation_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "ivalidation_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "ivalidation_run.db"
    timing_file = data_dir / "ivalidation_times.csv"
    graph_timing_file = data_dir / "graph_times.csv"

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
        for i, bac_angle in enumerate(range(40, 181, 5))
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
                    building_blocks={
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
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    mash_idx = constructed.mash_idx
                    acage = constructed.constructed_molecule
                    name = f"{lig}_{multiplier}_{idx}_{mash_idx}"
                    acage.write(structure_dir / f"{name}_unopt.mol")
                    num_vertices = iterator.get_num_building_blocks()

                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )
                    if num_components != 1:
                        continue

                    # Optimise and save.
                    logging.info("building %s", name)
                    try:
                        st1 = time.time()
                        conformer = scram.toy.graph_optimise_cage(
                            molecule=acage,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        et1 = time.time()
                        if conformer is not None:
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        st2 = time.time()
                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=constructed.topology_code,
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
                    structure_dir=structure_dir,
                    figure_dir=figure_dir,
                    filename="validationi_1.png",
                )

    make_plot(
        database_path=database_path,
        structure_dir=structure_dir,
        figure_dir=figure_dir,
        filename="validationi_1.png",
    )

    make_timings_plot(
        timing_file=timing_file,
        graph_timing_file=graph_timing_file,
        figure_dir=figure_dir,
        filename="validation_times.png",
    )
    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="validationi_2.png",
    )


if __name__ == "__main__":
    main()
