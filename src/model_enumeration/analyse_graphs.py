"""Script to analyse generated graphs."""

import json
import logging
import pathlib
import warnings
from collections import Counter, abc

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import stk
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

mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)


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

    fake_m12l24_path = (
        pathlib.Path(__file__).parent.absolute() / "rxnd_12P24.json"
    )

    stable_graphs = {
        "rx": {
            2: (1,),
            3: (2,),
            4: (0, 9),
            5: (27,),
            6: (2, 14),
            8: (555,),
        },
        "rx_nodoubles": {6: (0,), 8: (3,), 10: (50,)},
        "rx_fake": {},
    }

    for gset in ("rx", "rx_nodoubles", "rx_fake"):
        output = figure_dir / f"ganalysis_{gset}.json"
        if output.exists():
            with output.open("r") as f:
                properties = json.load(f)

        else:
            properties = {i: {} for i in multipliers}
            for multi in multipliers:
                if gset == "rx_fake":
                    with fake_m12l24_path.open("r") as f:
                        all_graphs = json.load(f)
                    graphs = [
                        cgx.scram.TopologyCode(
                            vertex_map=combination,
                            as_string=cgx.scram.vmap_to_str(combination),
                        )
                        for combination in all_graphs
                    ]

                else:
                    iterator = cgx.scram.TopologyIterator(
                        building_block_counts={
                            fake_tetra_bb: 1 * multi,
                            fake_ditopic_bb: 2 * multi,
                        },
                        graph_type=f"{1 * multi}P{2 * multi}",
                        graph_set=gset,
                    )
                    graphs = list(iterator.yield_graphs())

                for idx, topology_code in enumerate(graphs):
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

        pair = ("avg_eccentricity", "diameter")

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))

        for multi, datas in properties.items():
            xs = [float(datas[i][pair[0]]) for i in datas]
            xs1 = [
                float(datas[i][pair[0]]) for i in datas if datas[i]["stable"]
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
            if len(xs1) > 0:
                ax.scatter(
                    xs1,
                    ys1,
                    c=multi_cmap[str(multi)],
                    s=120,
                    alpha=1,
                    ec="k",
                    zorder=2,
                    label=f"$m$={multi}",
                )

        ax.legend(fontsize=16)
        filename = f"graph_analysis_{gset}_1.png"
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
    # output = figure_dir / "count_nodoubles.json"  # noqa: ERA001
    # gset = "rx_nodoubles"  # noqa: ERA001

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
        logging.info(
            r"%s percentage change from all: %s percent",
            title,
            round(
                (
                    sum([i[1] for i in toplots[idx]])
                    / sum([i[1] for i in toplots[0]])
                )
                * 100,
                2,
            ),
        )

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

    filename = f"graph_analysis_{gset}_2.png"
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


def get_stk_topology_code(
    tfunction: abc.Callable,
) -> tuple[cgx.scram.TopologyCode, np.ndarray]:
    """Get the default stk graph."""
    vps = tfunction._vertex_prototypes  # noqa: SLF001
    eps = tfunction._edge_prototypes  # noqa: SLF001

    combination = [(i.get_vertex1_id(), i.get_vertex2_id()) for i in eps]
    tc = cgx.scram.TopologyCode(
        vertex_map=combination,
        as_string=cgx.scram.vmap_to_str(combination),
    )

    positions = [i.get_position() for i in vps]

    return tc, positions


def find_stk_graphs() -> None:  # noqa: C901
    """Analyse the rustworkx graphs matching to stk graphs."""
    stk_graphs = {
        2: (("2P4", stk.cage.M2L4Lantern),),
        3: (("3P6", stk.cage.M3L6),),
        4: (("4P8", cgx.topologies.CGM4L8), ("4P82", cgx.topologies.M4L82)),
        5: (("5P10", stk.cage.FivePlusTen),),
        6: (("6P12", stk.cage.M6L12Cube), ("6P122", cgx.topologies.M6L122)),
        8: (
            ("8P16", stk.cage.EightPlusSixteen),
            ("8P162", cgx.topologies.M8L162),
        ),
        10: (("10P20", stk.cage.TenPlusTwenty),),
        12: (("12P24", cgx.topologies.CGM12L24),),
    }

    fake_m12l24_path = (
        pathlib.Path(__file__).parent.absolute() / "rxnd_12P24.json"
    )

    fake_ditopic_bb = cgx.molecular.TwoC1Arm(
        bead=cbead_d,
        abead1=abead_d,
    ).get_building_block()
    fake_tetra_bb = cgx.molecular.FourC1Arm(
        bead=tetra_bead,
        abead1=binder_bead,
    ).get_building_block()

    for multi, options in stk_graphs.items():
        iterator1 = cgx.scram.TopologyIterator(
            building_block_counts={
                fake_tetra_bb: 1 * multi,
                fake_ditopic_bb: 2 * multi,
            },
            graph_type=f"{1 * multi}P{2 * multi}",
            graph_set="rx",
        )
        graphs1 = list(iterator1.yield_graphs())
        logging.info(
            "for m=%s from rx, there are %s graphs", multi, len(graphs1)
        )
        iterator2 = cgx.scram.TopologyIterator(
            building_block_counts={
                fake_tetra_bb: 1 * multi,
                fake_ditopic_bb: 2 * multi,
            },
            graph_type=f"{1 * multi}P{2 * multi}",
            graph_set="rx_nodoubles",
        )
        graphs2 = list(iterator2.yield_graphs())
        logging.info(
            "for m=%s from rx_nodoubles, there are %s graphs",
            multi,
            len(graphs2),
        )

        stk_tcs = {i[0]: get_stk_topology_code(i[1]) for i in options}

        for idx, topology_code in enumerate(graphs1):
            for name, (tc, _) in stk_tcs.items():
                test_graph = tc.get_graph()
                if rx.is_isomorphic(topology_code.get_graph(), test_graph):
                    logging.info(
                        "m=%s, from rx, graph %s is isomorphic to %s",
                        multi,
                        idx,
                        name,
                    )

        for idx, topology_code in enumerate(graphs2):
            for name, (tc, _) in stk_tcs.items():
                test_graph = tc.get_graph()
                if rx.is_isomorphic(topology_code.get_graph(), test_graph):
                    logging.info(
                        "m=%s, from rx_nodoubles, graph %s is "
                        "isomorphic to %s",
                        multi,
                        idx,
                        name,
                    )

        if multi == 12:  # noqa: PLR2004
            with fake_m12l24_path.open("r") as f:
                all_graphs = json.load(f)

            logging.info(
                "Loaded %d graphs from %s", len(all_graphs), fake_m12l24_path
            )
            for combination in all_graphs:
                topology_code = cgx.scram.TopologyCode(
                    vertex_map=combination,
                    as_string=cgx.scram.vmap_to_str(combination),
                )
                for name, (tc, _) in stk_tcs.items():
                    test_graph = tc.get_graph()
                    if rx.is_isomorphic(topology_code.get_graph(), test_graph):
                        logging.info(
                            "m=%s, from rx_nodoubles with 1e6, graph %s is "
                            "isomorphic to %s",
                            multi,
                            idx,
                            name,
                        )


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    find_stk_graphs()
    count_graphs(figure_dir)
    analyse_graphs(figure_dir)


if __name__ == "__main__":
    main()
