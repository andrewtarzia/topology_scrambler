"""Script to analyse generated graphs."""

import json
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

from model_enumeration.mgen_utilities import (
    abead_d,
    binder_bead,
    cbead_d,
    tetra_bead,
)
from model_enumeration.utilities import multi_cmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def analyse_graphs(figure_dir: pathlib.Path) -> None:  # noqa: C901, PLR0912, PLR0915
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
        output = figure_dir / f"ganalysis_{gset}.json"
        if output.exists():
            with output.open("r") as f:
                properties = json.load(f)

        else:
            properties = {i: {} for i in multipliers}
            for multi in multipliers:
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        fake_tetra_bb: 1 * multi,
                        fake_ditopic_bb: 2 * multi,
                    },
                    graph_type=f"{1 * multi}P{2 * multi}",
                    graph_set=gset,
                )

                for idx, topology_code in enumerate(iterator.yield_graphs()):
                    nx_graph = topology_code.get_nx_graph()

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

                    for i in range(20):
                        if i not in counter:
                            counter[i] = 0

                    properties[multi][idx] = {
                        "avg_eccentricity": avg_eccentricity,
                        "diameter": diameter,
                        "avg_eccentricity_ratio": avg_eccentricity / diameter,
                        "num_parallel_edges": num_parallel_edges,
                        "counter": counter,
                        "stable": multi in stable_graphs[gset]
                        and idx in stable_graphs[gset][multi],
                    }

            with output.open("w") as f:
                json.dump(properties, f)

        pairs = (
            ("avg_eccentricity", "diameter"),
            ("multi", 6),
            ("multi", 8),
            ("multi", 10),
        )

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
        flat_axs = axs.flatten()
        for multi, datas in properties.items():
            for ax, pair in zip(flat_axs, pairs, strict=False):
                if pair[0] == "multi":
                    xs = [int(multi) for i in datas]
                    xs1 = [int(multi) for i in datas if datas[i]["stable"]]
                    ys = [
                        int(datas[i]["counter"][str(pair[1])]) for i in datas
                    ]
                    ys1 = [
                        int(datas[i]["counter"][str(pair[1])])
                        for i in datas
                        if datas[i]["stable"]
                    ]

                    ax.tick_params(axis="both", which="major", labelsize=16)
                    ax.set_xlabel("multiplier", fontsize=16)
                    ax.set_ylabel(f"count({pair[1]})", fontsize=16)

                else:
                    xs = [float(datas[i][pair[0]]) for i in datas]
                    xs1 = [
                        float(datas[i][pair[0]])
                        for i in datas
                        if datas[i]["stable"]
                    ]
                    if pair[1] == "diameter":
                        ylbl = f"{pair[1]}/multi"
                        scale = int(multi)
                        ax.set_xlim(1, 13)
                        ax.set_ylim(0, 2)
                    else:
                        scale = 1
                        ylbl = pair[1]
                    ys = [float(datas[i][pair[1]]) / scale for i in datas]
                    ys1 = [
                        float(datas[i][pair[1]]) / scale
                        for i in datas
                        if datas[i]["stable"]
                    ]
                    ax.tick_params(axis="both", which="major", labelsize=16)
                    ax.set_xlabel(pair[0], fontsize=16)
                    ax.set_ylabel(ylbl, fontsize=16)

                ax.scatter(
                    xs,
                    ys,
                    c="tab:gray",
                    s=40,
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


def count_graphs(figure_dir: pathlib.Path) -> None:  # noqa: C901, PLR0912, PLR0915
    """Analyse the rustworkx graphs."""
    multipliers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    output = figure_dir / "count.json"
    gset = "rx"

    if output.exists():
        with output.open("r") as f:
            properties = json.load(f)

    else:
        fake_ditopic_bb = cgx.molecular.TwoC1Arm(
            bead=cbead_d,
            abead1=abead_d,
        ).get_building_block()
        fake_tetra_bb = cgx.molecular.FourC1Arm(
            bead=tetra_bead,
            abead1=binder_bead,
        ).get_building_block()

        attempts = ("kamada", "spring", "spectral")

        properties = {i: {} for i in multipliers}
        for multi in multipliers:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    fake_tetra_bb: 1 * multi,
                    fake_ditopic_bb: 2 * multi,
                },
                graph_type=f"{1 * multi}P{2 * multi}",
                graph_set=gset,
            )

            for idx, topology_code in enumerate(iterator.yield_graphs()):
                constructions = {}
                for string in attempts:
                    try:
                        nx_graph = topology_code.get_nx_graph()
                        if string == "kamada":
                            nxpos = nx.kamada_kawai_layout(nx_graph, dim=3)
                        elif string == "spring":
                            nxpos = nx.spring_layout(nx_graph, dim=3)
                        elif string == "spectral":
                            nxpos = nx.spectral_layout(nx_graph, dim=3)
                        vertex_positions = {
                            nidx: np.array(nxpos[nidx]) * 10
                            for nidx in topology_code.get_nx_graph().nodes
                        }
                        _ = cgx.scram.try_except_construction(
                            iterator=iterator,
                            topology_code=topology_code,
                            building_block_configuration=None,
                            vertex_positions=vertex_positions,
                        )
                        constructions[string] = True
                    except ValueError:
                        constructions[string] = False

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

                for i in range(20):
                    if i not in counter:
                        counter[i] = 0

                properties[multi][idx] = {
                    "num_parallel_edges": num_parallel_edges,
                    "counter": counter,
                    "constructions": constructions,
                }
        with output.open("w") as f:
            json.dump(properties, f)

    fig, ax = plt.subplots(
        ncols=1, nrows=1, sharey=True, sharex=True, figsize=(8, 5)
    )
    titles = {
        0: ("all", "white", "o", 12),
        2: ("- 1-loops", "tab:orange", "o", 12),
        1: ("- doubles", "tab:green", "o", 12),
        3: ("passes spring", "tab:red", "P", 8),
        4: ("passes kamada", "tab:purple", "X", 6),
        5: ("passes spectral", "tab:cyan", "D", 4),
    }
    toplots = {i: [] for i in titles}
    for multi, datas in properties.items():
        toplots[0].append((multi, len(datas)))
        toplots[1].append(
            (
                multi,
                len(
                    [
                        i
                        for i in datas
                        if not (
                            datas[i]["num_parallel_edges"] != 0
                            or datas[i]["counter"]["4"] != 0
                        )
                    ]
                ),
            )
        )
        toplots[2].append(
            (
                multi,
                len([i for i in datas if datas[i]["num_parallel_edges"] == 0]),
            )
        )
        toplots[3].append(
            (
                multi,
                len([i for i in datas if datas[i]["constructions"]["spring"]]),
            )
        )
        toplots[4].append(
            (
                multi,
                len([i for i in datas if datas[i]["constructions"]["kamada"]]),
            )
        )
        toplots[5].append(
            (
                multi,
                len(
                    [i for i in datas if datas[i]["constructions"]["spectral"]]
                ),
            )
        )

    for idx, (title, c, m, s) in titles.items():
        ec = "none" if s < 12 else "k"  # noqa: PLR2004
        ax.plot(
            [i[0] for i in toplots[idx] if i[1] != 0],
            [i[1] for i in toplots[idx] if i[1] != 0],
            c="k",
            marker=m,
            markerfacecolor=c,
            markersize=s,
            alpha=1,
            mec=ec,
            zorder=1,
            label=title,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("multi", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.axhline(y=int(1e4), c="k", zorder=-1)
    ax.axhline(y=0, c="k", zorder=-1)
    ax.set_yscale("log")
    ax.legend(fontsize=16)

    filename = f"graph_analysis_{gset}_2"
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

    count_graphs(figure_dir)
    analyse_graphs(figure_dir)
    raise SystemExit(
        "for nx gen positions - can you mess around with clustering the geometries?"
    )


if __name__ == "__main__":
    main()
