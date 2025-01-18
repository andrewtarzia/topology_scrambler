"""Script to analyse generated graphs."""

import logging
import pathlib
import warnings
from collections import Counter

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
from rdkit import RDLogger
from ufo_utilities import abead_d, binder_bead, cbead_d, tetra_bead
from utilities import multi_cmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def analyse_graphs(figure_dir: pathlib.Path) -> None:  # noqa: PLR0915, C901
    """Analyse the rustworkx graphs."""
    multipliers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    fake_ditopic_bb = cgx.molecular.TwoC1Arm(
        bead=cbead_d,
        abead1=abead_d,
    ).get_building_block()
    fake_tetra_bb = cgx.molecular.FourC1Arm(
        bead=tetra_bead,
        abead1=binder_bead,
    ).get_building_block()

    stable_graphs = {
        "rx": {
            2: (1,),
            3: (2,),
            4: (9,),
            5: (27,),
            6: (2, 52, 94, 14),
            7: (82, 106),
            8: (555, 104, 378),
            9: (873, 39, 694, 696, 920),
            10: (664, 37, 728),
        },
        "rx_nodoubles": {
            6: (0,),
            8: (3,),
            9: (9,),
            11: (209,),
        },
    }

    for gset in ("rx", "rx_nodoubles"):
        properties = {i: {} for i in multipliers}
        targets = []
        for multi in multipliers:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    fake_tetra_bb: 1 * multi,
                    fake_ditopic_bb: 2 * multi,
                },
                graph_type=f"{1*multi}P{2*multi}",
                graph_set=gset,
            )

            for idx, topology_code in enumerate(iterator.yield_graphs()):
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
                    "stable": multi in stable_graphs[gset]
                    and idx in stable_graphs[gset][multi],
                }

                if num_parallel_edges == 0 and counter[4] == 0:
                    targets.append((multi, idx))

        pairs = (("multi", 4), ("multi", 6), ("multi", 8), ("multi", 10))

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
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
                    xs1 = [
                        datas[i][pair[0]] for i in datas if datas[i]["stable"]
                    ]
                    ys = [datas[i][pair[1]] for i in datas]
                    ys1 = [
                        datas[i][pair[1]] for i in datas if datas[i]["stable"]
                    ]
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

        filename = f"graph_analysis_{gset}_1"
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


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    analyse_graphs(figure_dir)
    msg = (
        "I want to be able to filter the above for types -- "
        "so I think make it in the topology code section"
        "- also want to try and remove the non-primitive loops"
        "-- needs to be moved into `contains_doubles`"
    )
    raise SystemExit(msg)


if __name__ == "__main__":
    main()
