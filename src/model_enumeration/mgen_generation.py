"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import shutil
from collections import defaultdict

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    StericTwoC1Arm,
    a2bead_d,
    abead_c,
    abead_d,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)
from model_enumeration.utilities import (
    eb_str,
    isomer_energy,
    multi_cmap,
    pore_str,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def contains_parallels(topology_code: cgx.scram.TopologyCode) -> bool:
    """True if the graph contains "1-loops"."""
    weighted_graph = topology_code.get_weighted_graph()
    num_parallel_edges = len(
        [
            i
            for i in weighted_graph.edges()
            if i == 2  # noqa: PLR2004
        ]
    )

    return num_parallel_edges != 0


def get_bb_topology_code_graph(
    topology_code: cgx.scram.TopologyCode,
    bb_config: cgx.scram.BuildingBlockConfiguration,
) -> rx.PyGraph:
    """Convert TopologyCode and BBConfig to rx graph."""
    graph: rx.PyGraph = rx.PyGraph(multigraph=True)

    vertices = {}
    for vi in sorted({i for j in topology_code.vertex_map for i in j}):
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if vi in vert_ids
        )

        vertices[f"{vi}-{bb_id}"] = graph.add_node(f"{vi}-{bb_id}")

    for vert in topology_code.vertex_map:
        v1 = vert[0]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v1 in vert_ids
        )
        v1str = f"{v1}-{bb_id}"
        v2 = vert[1]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v2 in vert_ids
        )
        v2str = f"{v2}-{bb_id}"
        nodeaidx = vertices[v1str]
        nodebidx = vertices[v2str]
        graph.add_edge(nodeaidx, nodebidx, None)

    return graph


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    database.add_properties(key=name, property_dict={"lowest_e_of_mash": True})

    if "large_binder_binder_angles" not in properties:
        ss_dists = (
            stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
                molecule=final_molecule,
                metal_atom_nos=(16,),
            )
        )
        if len(ss_dists) != 0:
            min_ss_value = min(ss_dists.values())
            max_ss_value = max(ss_dists.values())

            database.add_properties(
                key=name,
                property_dict={
                    "min_ss_dist": min_ss_value,
                    "max_ss_dist": max_ss_value,
                },
            )

        # Get the bg angles.
        ligands = stko.molecule_analysis.DecomposeMOC().decompose(
            molecule=final_molecule,
            metal_atom_nos=(46,),
        )

        small_binder_binder_angles = []
        large_binder_binder_angles = []
        potential_smiles = [
            "[Pb]~[Ga]",  # Large.
            "[Pb]~[Ba]",  # Small.
            "[Pb]~[Mn]",  # Small.
        ]
        for lig in ligands:
            for smiles in potential_smiles:
                as_building_block = stk.BuildingBlock.init_from_molecule(
                    lig,
                    stk.SmartsFunctionalGroupFactory(
                        smarts=smiles, bonders=(0,), deleters=(1,)
                    ),
                )
                if as_building_block.get_num_functional_groups() == 2:  # noqa: PLR2004
                    large = smiles in ("[Pb]~[Ga]",)
                    break

            vectors = [
                as_building_block.get_centroid(atom_ids=fg.get_bonder_ids())
                - as_building_block.get_centroid(atom_ids=fg.get_deleter_ids())
                for fg in as_building_block.get_functional_groups()
            ]
            normed = [i / np.linalg.norm(i) for i in vectors]
            angle = np.degrees(
                stko.vector_angle(vector1=normed[0], vector2=normed[1])
            )
            if large:
                large_binder_binder_angles.append(angle)
            else:
                small_binder_binder_angles.append(angle)

        database.add_properties(
            key=name,
            property_dict={
                "large_binder_binder_angles": large_binder_binder_angles,
                "small_binder_binder_angles": small_binder_binder_angles,
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

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
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
        for multi, evalues in energies.items():
            if len(evalues) == 0:
                continue

            sorted_energies = sorted(evalues, key=lambda p: p[0])
            min_energy = sorted_energies[0]

            sorted_pores = sorted(evalues, key=lambda p: p[2], reverse=True)
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
                color=multi_cmap[multi],
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
                color=multi_cmap[multi],
                fontsize=8,
            )

            ax.scatter(
                [i[2] for i in evalues],
                [i[0] for i in evalues],
                marker="o",
                c=multi_cmap[multi],
                s=20,
                ec="none",
                alpha=0.3,
                label=f"M={multi}",
            )
            ax.scatter(
                min_energy[2],
                min_energy[0],
                marker="o",
                c=multi_cmap[multi],
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

    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)

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
    pairs: list[tuple[str, str]],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 10))
    energies = {}

    xs = ["1", "2", "3", "4"]

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        pair = tuple(entry.key.split("_")[:2])
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
        y = pairs.index(pair)

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
    ax.set_yticklabels(["_".join(i) for i in pairs])
    ax.set_xlim(None, 4)

    ax.axvline(0.8, c="k", alpha=0.5)
    ax.axvline(1.8, c="k", alpha=0.5)
    ax.axvline(2.8, c="k", alpha=0.5)

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


def make_summary_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
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
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        x = pairs.index((l1, l2))
        x_count[multi][x] += 1

        if "lowest_e_of_mash" not in entry.properties:
            continue
        energy = entry.properties["energy_per_bb"]

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
    ax.set_xticklabels(["_".join(i) for i in pairs], rotation=90)
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


def make_opt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, (ax, ax1) = plt.subplots(
        ncols=2,
        figsize=(10, 5),
        width_ratios=[3, 1],
        sharey=True,
    )

    stages = (
        "opt1",
        "shifted",
        "smd",
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
        "ns",
    )

    sources = {i: 0 for i in stages}
    mashes = {}
    lowe_sources = {i: 0 for i in stages}  # Produces low energy structures.
    lowe_mashes = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        midx = entry.properties["mash_idx"]
        if midx not in mashes:
            mashes[midx] = 0
            lowe_mashes[midx] = 0

        sources[entry.properties["source"]] += 1
        mashes[midx] += 1
        energy = entry.properties["energy_per_bb"]
        if energy < 1:
            lowe_sources[entry.properties["source"]] += 1
            lowe_mashes[midx] += 1

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
        [int(i) for i in mashes],
        [lowe_mashes[i] for i in mashes],
        color="#086788",
        edgecolor="none",
        lw=2,
        label=f"{eb_str()} < 1.0",
    )
    ax1.bar(
        [int(i) for i in mashes],
        [mashes[i] for i in mashes],
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

    ax1.tick_params(axis="both", which="major", labelsize=16)
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


def sterics_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(tuple)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]

        x = entry.properties["forcefield_dict"]["v_dict"]["s"]

        if "min_ss_dist" not in entry.properties:
            continue
        y = entry.properties["min_ss_dist"]

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas[(multi, l1, l2)][x][1]
            ):
                datas[(multi, l1, l2)][x] = (
                    y,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas[(multi, l1, l2)][x] = (y, entry.properties["energy_per_bb"])

    xlbl = r"min. $r_{s-s}$ [$\AA$]"

    for (multi, l1, l2), xdict in datas.items():
        ax.scatter(
            list(xdict),
            [xdict[i][0] for i in xdict],
            alpha=1.0,
            c=multi_cmap[multi],
            ec="k",
            s=60,
            label=f"M{multi}" if (l1, l2) == ("lf", "ls3") else None,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel(xlbl, fontsize=16)
    ax.legend(ncol=1, fontsize=16)
    ax.set_ylim(0, None)

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


def binder_vector_angles_plot(  # noqa: C901, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    datas_lge: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    datas_sma: dict[str, dict[str, list[float]]] = defaultdict(tuple)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]

        if "large_binder_binder_angles" not in entry.properties:
            continue
        ylge = entry.properties["large_binder_binder_angles"]
        ysma = entry.properties["small_binder_binder_angles"]

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas_lge[(multi, l1, l2)][1]
            ):
                datas_lge[(multi, l1, l2)] = (
                    ylge,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas_lge[(multi, l1, l2)] = (
                ylge,
                entry.properties["energy_per_bb"],
            )

        try:
            if (
                entry.properties["energy_per_bb"]
                < datas_sma[(multi, l1, l2)][1]
            ):
                datas_sma[(multi, l1, l2)] = (
                    ysma,
                    entry.properties["energy_per_bb"],
                )
        except IndexError:
            datas_sma[(multi, l1, l2)] = (
                ysma,
                entry.properties["energy_per_bb"],
            )

    lsdone = set()
    for (multi, l1, l2), xdict in datas_sma.items():
        ydict = datas_lge[(multi, l1, l2)]

        if xdict[1] > 1.0:
            alpha = 0.3
            zorder = -1
            c = "gray"
            ec = "none"
            s = 30
            label = None

        elif xdict[1] > 0.3:  # noqa: PLR2004
            alpha = 1
            zorder = 0
            c = multi_cmap[multi]
            ec = "none"
            s = 30
            label = f"M{multi}"
            if label in lsdone:
                label = None
            lsdone.add(label)

        else:
            alpha = 1
            zorder = 1
            c = multi_cmap[multi]
            ec = "k"
            s = 60
            label = None

        ax.scatter(
            xdict[0],
            ydict[0],
            alpha=alpha,
            marker="o",
            c=c,
            ec=ec,
            s=s,
            label=label,
            zorder=zorder,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("mean small binder angle [$^\\circ$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylabel("mean large binder angle [$^\\circ$]", fontsize=16)
    ax.plot((0, 180), (0, 180), c="k", zorder=-1)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
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


def get_regraphed_molecule(
    scale: str,
    topology_code: cgx.scram.TopologyCode,
    iterator: cgx.scram.TopologyIterator,
    bb_config: cgx.scram.BuildingBlockConfiguration,
) -> stk.ConstructedMolecule:
    """Take a graph that considers all atoms, and get atom positions."""
    constructed_molecule = cgx.scram.try_except_construction(
        iterator=iterator,
        topology_code=topology_code,
        building_block_configuration=bb_config,
        vertex_positions=None,
    )
    stko_graph = stko.Network.init_from_molecule(constructed_molecule)
    _, stype, scale_value = scale.split("-")
    if stype == "spring":
        nx_positions = nx.spring_layout(stko_graph.get_graph(), dim=3)
    elif stype == "kamada":
        nx_positions = nx.kamada_kawai_layout(stko_graph.get_graph(), dim=3)
    else:
        raise NotImplementedError
    pos_mat = np.array([nx_positions[i] for i in nx_positions])
    return constructed_molecule.with_position_matrix(
        pos_mat * float(scale_value)
    ).with_centroid(np.array((0.0, 0.0, 0.0)))


def get_vertexset_molecule(
    scale: str,
    topology_code: cgx.scram.TopologyCode,
    iterator: cgx.scram.TopologyIterator,
    bb_config: cgx.scram.BuildingBlockConfiguration,
) -> stk.ConstructedMolecule:
    """Take a graph and genereate from graph vertex positions.."""
    if scale is None:
        return cgx.scram.try_except_construction(
            iterator=iterator,
            topology_code=topology_code,
            building_block_configuration=bb_config,
            vertex_positions=None,
        )

    nx_graph = topology_code.get_nx_graph()
    _, stype, scale_value = scale.split("-")
    if stype == "kamada":
        nxpos = nx.kamada_kawai_layout(nx_graph, dim=3)
    elif stype == "spring":
        nxpos = nx.spring_layout(nx_graph, dim=3)
    elif stype == "spectral":
        nxpos = nx.spectral_layout(nx_graph, dim=3)
    else:
        raise NotImplementedError

    vertex_positions = {
        nidx: np.array(nxpos[nidx]) * float(scale_value)
        for nidx in topology_code.get_nx_graph().nodes
    }
    return cgx.scram.try_except_construction(
        iterator=iterator,
        topology_code=topology_code,
        building_block_configuration=bb_config,
        vertex_positions=vertex_positions,
    )


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgen_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgen_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgen_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgen_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cg"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgen.db"

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        # From prep.
        "lf": {
            "egb": 120,
            "deg": 180,
            "dd": 8.0,
            "de": 4.3,
            "dde": 133,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e10": {
            "egb": 120,
            "deg": 180,
            "dd": 5.9,
            "de": 4.1,
            "dde": 139,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e11": {
            "egb": 120,
            "deg": 180,
            "dd": 6.9,
            "de": 1.4,
            "dde": 170,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e12": {
            "egb": 120,
            "deg": 180,
            "dd": 7.0,
            "de": 1.5,
            "dde": 167,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e13": {
            "egb": 120,
            "deg": 180,
            "dd": 10.5,
            "de": 1.4,
            "dde": 151,
            "eg": 1.4,
            "gb": 1.4,
        },
        "e14": {
            "egb": 120,
            "deg": 180,
            "dd": 5.9,
            "de": 4.1,
            "dde": 143,
            "eg": 1.4,
            "gb": 1.4,
        },
        # From optl.
        "l2": {"ba": 2.8, "aa": 4.9, "bac": 150, "s": 0.0},
        "ls2": {"ba": 2.8, "aa": 4.7, "bac": 155, "s": 0.0},
        "ls3": {"ba": 2.8, "aa": 5.0, "bac": 145, "s": 0.5},
        "ls4": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 1.0},
        "ls5": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 0.5},
        "ls7": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 1.5},
        "ls8": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 2.0},
        "l3": {"ba": 2.8, "aa": 5.3, "bac": 165, "s": 0.0},
        "ls10": {"ba": 2.8, "aa": 5.4, "bac": 167, "s": 0.0},
        "l1": {"ba": 2.8, "aa": 8.2, "bac": 136, "s": 0.0},
        "e16": {"ba": 2.8, "aa": 7.3, "bac": 121, "s": 0.0},
        "e18": {"ba": 2.8, "aa": 10.0, "bac": 121, "s": 0.0},
        "e17": {
            "egb": 90,
            "deg": 150,
            "dd": 7.3,
            "de": 4.0,
            "dde": 151,
            "eg": 2.4,
            "gb": 2.8,
        },
        "la": {
            "egb": 90,
            "deg": 150,
            "dd": 6.1,
            "de": 8.3,
            "dde": 136,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lb": {
            "egb": 90,
            "deg": 150,
            "dd": 6.7,
            "de": 5.7,
            "dde": 148,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lc": {
            "egb": 90,
            "deg": 150,
            "dd": 7.0,
            "de": 5.7,
            "dde": 150,
            "eg": 2.4,
            "gb": 2.8,
        },
        "ld": {
            "egb": 90,
            "deg": 150,
            "dd": 7.5,
            "de": 5.7,
            "dde": 165,
            "eg": 2.4,
            "gb": 2.8,
        },
    }

    ligand_types = {
        # From prep.
        "lf": "sixbead",
        "e10": "sixbead",
        "e11": "sixbead",
        "e12": "sixbead",
        "e13": "sixbead",
        "e14": "sixbead",
        "e17": "sixbead",
        "la": "sixbead",
        "lb": "sixbead",
        "lc": "sixbead",
        "ld": "sixbead",
        # From optl.
        "e16": "twoarm",
        "e18": "twoarm",
        "l2": "twoarm",
        "ls2": "twoarm",
        "ls3": "stwoarm",
        "ls4": "stwoarm",
        "ls5": "stwoarm",
        "ls7": "stwoarm",
        "ls8": "stwoarm",
        "ls9": "twoarm",
        "ls10": "twoarm",
        "l1": "twoarm",
        "l3": "twoarm",
    }

    pairs_to_predict = [
        # large, small.
        ("lf", "l2"),
        ("lf", "ls2"),
        ("lf", "ls3"),
        ("lf", "ls4"),
        ("lf", "ls5"),
        ("lf", "ls7"),
        ("lf", "ls8"),
        ("lf", "l3"),
        ("lf", "ls10"),
        ("la", "l1"),
        ("lb", "l1"),
        ("lc", "l1"),
        ("ld", "l1"),
        ("la", "l2"),
        ("lb", "l2"),
        ("lc", "l2"),
        ("ld", "l2"),
        ("la", "l3"),
        ("lb", "l3"),
        ("lc", "l3"),
        ("ld", "l3"),
        ("e10", "e16"),
        ("e17", "e16"),
        ("e17", "e10"),
        ("e10", "e11"),
        ("e14", "e16"),
        ("e14", "e18"),
        ("e10", "e18"),
        ("e10", "e12"),
        ("e14", "e11"),
        ("e14", "e12"),
        ("e13", "e11"),
        ("e13", "e12"),
        ("e14", "e13"),
        ("e12", "e11"),
    ]

    pairs = {}
    for large, small in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        else:
            msg = large
            raise NotImplementedError(msg)

        if ligand_types[small] == "twoarm":
            small_prec = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
        elif ligand_types[small] == "stwoarm":
            small_prec = StericTwoC1Arm(
                bead=cbead_d, abead1=abead_d, steric_bead=steric_bead
            )
        elif ligand_types[small] == "sixbead":
            small_prec = cgx.molecular.SixBead(
                bead=c2bead_d,
                abead1=a2bead_d,
                abead2=e2bead_d,
            )
        else:
            msg = small
            raise NotImplementedError(msg)

        multi = (1, 2, 3, 4) if large == "lf" else (1, 2, 3)
        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "multipliers": multi,
            "vdw_cutoff": 2,
        }

    if args.run:
        for pair in pairs:
            large_name = pairs[pair]["large_name"]
            small_name = pairs[pair]["small_name"]

            large = pairs[pair]["large"]
            small = pairs[pair]["small"]
            tetra = pairs[pair]["tetra"]

            forcefield = precursors_to_forcefield(
                pair=pair,
                large=large,
                small=small,
                large_meas=ligand_measures[large_name],
                small_meas=ligand_measures[small_name],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
            )

            small_bb = cgx.utilities.optimise_ligand(
                molecule=small.get_building_block(),
                name=f"{pair}_{small.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            small_bb.write(
                str(ligand_dir / f"{pair}_{small.get_name()}_optl.mol")
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(str(ligand_dir / f"{tetra.get_name()}_optl.mol"))

            large_bb = cgx.utilities.optimise_ligand(
                molecule=large.get_building_block(),
                name=f"{pair}_{large.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            large_bb.write(
                str(ligand_dir / f"{pair}_{large.get_name()}_optl.mol")
            )

            pair_lowest_energy = float("inf")

            for multiplier in pairs[pair]["multipliers"]:
                if pair_lowest_energy < isomer_energy():
                    continue

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
                attempts = (
                    None,
                    "regraphed-spring-10",
                    "regraphed-kamada-10",
                    "regraphed-spring-5",
                    "regraphed-kamada-5",
                    "regraphed-spring-3",
                    "regraphed-kamada-3",
                    "set-kamada-15",
                    "set-kamada-10",
                    "set-kamada-5",
                    "set-kamada-3",
                    "set-spring-15",
                    "set-spring-10",
                    "set-spring-5",
                    "set-spring-3",
                    "set-spectral-15",
                    "set-spectral-10",
                    "set-spectral-5",
                    "set-spectral-3",
                )

                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Filter graphs for 1-loops.
                    if contains_parallels(topology_code):
                        continue

                    if pair_lowest_energy < isomer_energy():
                        logging.info(
                            "energy < 0.1 for tidx: %s and bidx: %s, stopping",
                            idx,
                            bb_config.idx,
                        )
                        continue

                    # Testing bb-config aware graph check.
                    # Convert TopologyCode to a graph.
                    current_graph = get_bb_topology_code_graph(
                        topology_code=topology_code,
                        bb_config=bb_config,
                    )

                    # Check that graph for isomorphism with others graphs.
                    passed_iso = True
                    for tc, bc in run_topology_codes:
                        test_graph = get_bb_topology_code_graph(
                            topology_code=tc, bb_config=bc
                        )

                        if rx.is_isomorphic(
                            current_graph,
                            test_graph,
                            node_matcher=lambda x, y: x.split("-")[1]
                            == y.split("-")[1],
                        ):
                            passed_iso = False
                            break

                    if not passed_iso:
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
                                    "l1": large_name,
                                    "l2": small_name,
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
                        try:
                            current_energy = (
                                cgx.utilities.AtomliteDatabase(database_path)
                                .get_entry(name)
                                .properties["energy_per_bb"]
                            )
                            pair_lowest_energy = min(
                                (pair_lowest_energy, current_energy)
                            )

                        except RuntimeError:
                            pass

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
                    )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
        pairs=pairs_to_predict,
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
        pairs=pairs_to_predict,
    )
    sterics_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_6.png",
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_5.png",
    )

    binder_vector_angles_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_7.png",
    )
    for pair in pairs:
        make_plot(
            database_path=database_path,
            target_pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"mgen_1_{pair}.png",
        )
    raise SystemExit("a plot that shows the three/2? distinct case studies")
    raise SystemExit("rethink binders, because it is not handling minus")


if __name__ == "__main__":
    main()
