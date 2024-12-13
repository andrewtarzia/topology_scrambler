"""Script to analyse generated graphs."""

import logging
import pathlib
import warnings
from collections import Counter

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import rustworkx as rx
from rdkit import RDLogger
from utilities import abead_d, binder_bead, cbead_d, tetra_bead
from validation_utilities import multi_cmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def analyse_graphs(figure_dir: pathlib.Path) -> None:  # noqa: PLR0915, C901
    """Analyse the rustworkx graphs."""
    multipliers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    properties = {i: {} for i in multipliers}

    fake_ditopic_bb = cgx.molecular.TwoC1Arm(
        bead=cbead_d,
        abead1=abead_d,
    ).get_building_block()
    fake_tetra_bb = cgx.molecular.FourC1Arm(
        bead=tetra_bead,
        abead1=binder_bead,
    ).get_building_block()

    stable_graphs = {
        2: (1,),
        3: (2,),
        4: (12,),
        5: (44,),
        6: (2, 55, 129, 14),
        8: (395, 591),
    }

    targets = []
    for multi in multipliers:
        iterator = cgx.scram.IHomolepticTopologyIterator(
            building_block_counts={
                fake_tetra_bb: 1 * multi,
                fake_ditopic_bb: 2 * multi,
            },
            graph_type=f"{1*multi}P{2*multi}",
        )

        for idx, topology_code in enumerate(iterator.get_graphs()):
            nx_graph = topology_code.get_nx_graph()
            if len(list(nx.connected_components(nx_graph))) != 1:
                continue

            avg_eccentricity = np.mean(
                list(nx.eccentricity(nx_graph).values())
            )
            diameter = nx.diameter(nx_graph)

            weighted_graph = topology_code.get_weighted_graph()
            num_parallel_edges = len(
                [
                    i
                    for i in weighted_graph.edges()
                    if i == 2  # noqa: PLR2004
                ]
            )

            filtered_paths = set()
            for node in weighted_graph.nodes():
                paths = list(
                    rx.graph_all_simple_paths(
                        weighted_graph,
                        origin=node,
                        to=node,
                        cutoff=12,
                        min_depth=4,
                    )
                )

                for path in paths:
                    if (
                        tuple(path) not in filtered_paths
                        and tuple(path[::-1]) not in filtered_paths
                    ):
                        filtered_paths.add(tuple(path))

            path_lengths = [len(i) - 1 for i in filtered_paths]
            counter = Counter(path_lengths)

            properties[multi][idx] = {
                "avg_eccentricity": avg_eccentricity,
                "diameter": diameter,
                "avg_eccentricity_ratio": avg_eccentricity / diameter,
                "num_parallel_edges": num_parallel_edges,
                "counter": counter,
                "stable": multi in stable_graphs
                and idx in stable_graphs[multi],
            }

            if num_parallel_edges == 0 and counter[4] == 0:
                targets.append((multi, idx))

    print(
        "want to filter the above for types -- "
        "so I think make it in the topology code section"
        "- also want to try and remove the non-primitive loops"
        "-- needs to be moved into `contains_doubles`"
    )

    pairs = (
        ("avg_eccentricity_ratio", "num_parallel_edges"),
        ("multi", 4),
        ("multi", 6),
        ("multi", 8),
        ("multi", 10),
        ("multi", 12),
    )

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 10))
    flat_axs = axs.flatten()
    for multi in properties:
        datas = properties[multi]

        for ax, pair in zip(flat_axs, pairs, strict=False):
            if pair[0] == "multi":
                xs = [multi for i in datas]
                xs1 = [multi for i in datas if datas[i]["stable"]]
                ys = [datas[i]["counter"][int(pair[1])] for i in datas]
                ys1 = [
                    datas[i]["counter"][int(pair[1])]
                    for i in datas
                    if datas[i]["stable"]
                ]

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel("multiplier", fontsize=16)
                ax.set_ylabel(f"count({pair[1]})", fontsize=16)

            else:
                xs = [datas[i][pair[0]] for i in datas]
                xs1 = [datas[i][pair[0]] for i in datas if datas[i]["stable"]]
                ys = [datas[i][pair[1]] for i in datas]
                ys1 = [datas[i][pair[1]] for i in datas if datas[i]["stable"]]
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel(pair[0], fontsize=16)
                ax.set_ylabel(pair[1], fontsize=16)

            ax.scatter(
                xs,
                ys,
                c="tab:gray",
                s=80,
                alpha=0.1,
                ec="none",
                zorder=1,
            )
            ax.scatter(
                xs1,
                ys1,
                c=multi_cmap[str(multi)],
                s=120,
                alpha=1,
                ec="k",
                zorder=2,
            )

    filename = "validationi_graphanalysis"
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
    ax.scatter(
        nx_graph_timings["num_vertices"],
        nx_graph_timings["gen-time/s"],
        c="tab:green",
        marker="s",
        s=100,
        alpha=1,
        label="nx-graph time",
        ec="k",
    )
    ax.scatter(
        rx_graph_timings["num_vertices"],
        rx_graph_timings["gen-time/s"],
        c="tab:purple",
        marker="o",
        s=100,
        alpha=1,
        label="rx-graph time",
        ec="k",
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


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    raise SystemExit("rerun graph times")
    raise SystemExit("rerun, save timing files")
    raise SystemExit("Search for m12 stk combo in jaon")
    graph_timing_file = data_dir / "graph_times.csv"
    analyse_graphs(figure_dir)

    make_timings_plot(
        timing_file=timing_file,
        figure_dir=figure_dir,
        filename="validationd_times.png"
        if args.nodoubles
        else "validation_times.png",
    )


if __name__ == "__main__":
    main()
