"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import shutil
from collections import abc, defaultdict

import atomlite
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openmm
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
    constant_definer_dict,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
    trigonal_bead,
)
from model_enumeration.utilities import (
    contains_parallels,
    eb_str,
    isomer_energy,
    multi_cmap,
    percent_change,
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


def generate_nearby_forcefields(  # noqa: C901, PLR0912
    forcefield: cgx.forcefields.ForceField,
    actual_present_bead_elements: abc.Sequence[str],
) -> cgx.forcefields.ForceFieldLibrary:
    """Generate nearby forcefields."""
    modifiable = (
        # "dd",
        # "de",
        "deg",
        "egb",
        "dde",
        # "ac",
        "bac",
        "bed",
        # "zz",
        # "zr",
        "zzr",
        "zrf",
        "rfb",
    )

    ff_targets = forcefield.get_targets()

    fflib = cgx.forcefields.ForceFieldLibrary(
        present_beads=forcefield.get_present_beads(),
        vdw_bond_cutoff=forcefield.get_vdw_bond_cutoff(),
        prefix=f"{forcefield.get_prefix()}_lib",
    )
    for fftype, currtargets in ff_targets.items():
        for target in currtargets:
            if fftype == "bonds":
                cp = f"{target.type1}{target.type2}"

                bond_r = target.bond_r.value_in_unit(openmm.unit.angstrom)
                # Do not change bonds not in molecule!
                if cp in modifiable and (
                    target.element1
                    and target.element2 in actual_present_bead_elements
                ):
                    bond_rs = [
                        openmm.unit.Quantity(
                            value=i,
                            unit=openmm.unit.angstrom,
                        )
                        for i in (
                            bond_r,
                            bond_r - percent_change(bond_r, 2),
                            bond_r + percent_change(bond_r, 2),
                        )
                    ]
                else:
                    bond_rs = [target.bond_r]

                target_range = cgx.terms.TargetBondRange(
                    type1=target.type1,
                    type2=target.type2,
                    element1=target.element1,
                    element2=target.element2,
                    bond_rs=bond_rs,
                    bond_ks=[target.bond_k],
                )
                fflib.add_bond_range(target_range)

            elif fftype == "angles":
                if isinstance(
                    target,
                    cgx.terms.TargetPyramidAngle | cgx.terms.TargetAngle,
                ):
                    cp = f"{target.type1}{target.type2}{target.type3}"

                    angle_v = target.angle.value_in_unit(openmm.unit.degrees)
                    # Do not change bonds not in molecule!
                    if cp in modifiable and (
                        target.element1
                        and target.element2
                        and target.element3 in actual_present_bead_elements
                    ):
                        angles = [
                            openmm.unit.Quantity(
                                value=i, unit=openmm.unit.degrees
                            )
                            for i in [
                                angle_v,
                                angle_v - percent_change(angle_v, 2),
                                angle_v + percent_change(angle_v, 2),
                            ]
                            if i <= 180  # noqa: PLR2004
                        ]

                    else:
                        angles = [target.angle]

                    if isinstance(target, cgx.terms.TargetPyramidAngle):
                        func = cgx.terms.PyramidAngleRange
                    elif isinstance(target, cgx.terms.TargetAngle):
                        func = cgx.terms.TargetAngleRange

                    target_range = func(
                        type1=target.type1,
                        type2=target.type2,
                        type3=target.type3,
                        element1=target.element1,
                        element2=target.element2,
                        element3=target.element3,
                        angles=angles,
                        angle_ks=[target.angle_k],
                    )

                if isinstance(target, cgx.terms.TargetCosineAngle):
                    raise NotImplementedError

                fflib.add_angle_range(target_range)

            elif fftype == "torsions":
                target_range = cgx.terms.TargetTorsionRange(
                    search_string=target.search_string,
                    search_estring=target.search_estring,
                    measured_atom_ids=target.measured_atom_ids,
                    phi0s=[target.phi0],
                    torsion_ks=[target.torsion_k],
                    torsion_ns=[target.torsion_n],
                )
                fflib.add_torsion_range(target_range)

            elif fftype == "nonbondeds":
                target_range = cgx.terms.TargetNonbondedRange(
                    bead_class=target.bead_class,
                    bead_element=target.bead_element,
                    sigmas=[target.sigma],
                    epsilons=[target.epsilon],
                    force=target.force,
                )
                fflib.add_nonbonded_range(target_range)

            else:
                raise NotImplementedError

    return fflib


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


def analyse_cage(database_path: pathlib.Path, name: str) -> None:
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
            metal_atom_nos=(46, 6),
        )

        small_binder_binder_angles = []
        large_binder_binder_angles = []
        potential_smiles = [
            "[Pb]~[Ga]",  # Large.
            "[Pb]~[Fe]",  # Large.
            "[Pb]~[Ba]",  # Small.
            "[Pb]~[Mn]",  # Small.
            "[Pb]~[Mn]",  # Tritopic.
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
                    large = smiles in ("[Pb]~[Ga]", "[Pb]~[Fe]")
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


def make_plot(  # noqa: PLR0915
    target_pair: str,
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax0, ax, ax1) = plt.subplots(
        ncols=3,
        figsize=(16, 5),
        width_ratios=[1, 1, 1],
    )

    entries = list(cgx.utilities.AtomliteDatabase(database_path).get_entries())
    ff_entries = [i for i in entries if "_f-" in i.key]

    systems = {}
    for entry in entries:
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]

        if entry.properties["pair"] != target_pair:
            continue
        energy = entry.properties["energy_per_bb"]

        ff_options = [i for i in ff_entries if i.key.startswith(entry.key)]
        ff_energies = [
            ff_entry.properties["energy_per_bb"] for ff_entry in ff_options
        ]
        ffidxs = [int(i.key.split("-")[-1]) for i in ff_options]

        systems[entry.key] = (energy, ff_energies, multi, tidx, bidx)

    rankings = {}
    energies = [energy for energy, _, _, _, _ in systems.values()]
    rankings[-1] = [energies.index(i) for i in sorted(energies)]

    systems_counts_lt_1 = {
        i: [-1] if values[0] < 1 else [] for i, values in systems.items()
    }

    systems_counts_as_best = {i: [] for i in systems}
    systems_counts_as_best[list(systems.keys())[rankings[-1][0]]].append(-1)

    for idx in ffidxs:
        energies = [
            ffenergies[idx] for _, ffenergies, _, _, _ in systems.values()
        ]
        for system, values in systems.items():
            if values[1][idx] < 1.0:
                systems_counts_lt_1[system].append(idx)

        rankings[idx] = [energies.index(i) for i in sorted(energies)]
        lowest_energy = rankings[idx][0]

        systems_counts_as_best[list(systems.keys())[lowest_energy]].append(idx)

    labels = set()
    for ix, (system, when_ranked) in enumerate(systems_counts_as_best.items()):
        energies = [systems[system][0], *list(systems[system][1])]

        count = len(when_ranked)
        count_lt_1 = len(systems_counts_lt_1[system])
        ax0.barh(
            ix,
            count_lt_1 / (len(ffidxs) + 1),
            color=multi_cmap[systems[system][2]],
            alpha=1,
            height=0.8,
            label=f"M={systems[system][2]}"
            if systems[system][2] not in labels
            else None,
        )
        ax.barh(
            ix,
            count / (len(ffidxs) + 1),
            color=multi_cmap[systems[system][2]],
            alpha=1,
            height=0.8,
            label=f"M={systems[system][2]}"
            if systems[system][2] not in labels
            else None,
        )
        labels.add(systems[system][2])

        l1, l2, multi, tidx, midx, bidx = system.split("_")

        if (multi, tidx, bidx) in (("3", "2", "b1"), ("4", "9", "b3")):
            ax1.scatter(
                [energies[0] for i, energy in enumerate(energies) if i != 0],
                [energy for i, energy in enumerate(energies) if i != 0],
                alpha=1,
                ec="k",
                s=120,
                label=f"T: $m$: {systems[system][2]}, "
                f"$t$: {systems[system][3]}"
                f", $b$: {systems[system][4]}",
            )

        if count / (len(ffidxs) + 1) > 0:
            ax1.scatter(
                [
                    energies[0]
                    for i, energy in enumerate(energies)
                    if i - 1 in when_ranked and i != 0
                ],
                [
                    energy
                    for i, energy in enumerate(energies)
                    if i - 1 in when_ranked and i != 0
                ],
                alpha=1,
                ec="k",
                s=120,
                label=f"$m$: {systems[system][2]}, $t$: {systems[system][3]}"
                f", $b$: {systems[system][4]}",
            )
            ax.text(
                count / (len(ffidxs) + 1) + 0.01,
                ix,
                f"$t$: {systems[system][3]}, $b$: {systems[system][4]}",
                ha="left",
                va="center",
                fontsize=16,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_yticks([])
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(f"prop. rank 1 (of {len(ffidxs) + 1})", fontsize=16)
    ax.legend(fontsize=16)

    ax0.tick_params(axis="both", which="major", labelsize=16)
    ax0.set_yticks([])
    ax0.set_xlim(0, 1.2)
    ax0.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax0.set_xlabel(f"prop. {eb_str()} < 1.0", fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel(f"original {eb_str()}", fontsize=16)
    ax1.set_ylabel(f"top ranked ffx {eb_str()}", fontsize=16)
    ax1.plot((0, 10), (0, 10), c="k", zorder=-1)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 10)
    ax1.legend(fontsize=16)

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
    fig, ax = plt.subplots(figsize=(7, 10))
    energies = {}

    xs = []

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        if multi not in xs:
            xs.append(multi)
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


def parity_plot(  # noqa: C901, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

    tarzia_result = {
        # large, small.
        ("la", "l1"): (0.54, "tab:orange"),
        ("lb", "l1"): (0.35, "tab:blue"),
        ("lc", "l1"): (0.34, "tab:blue"),
        ("ld", "l1"): (0.32, "tab:orange"),
        ("la", "l2"): (0.96, "tab:orange"),
        ("lb", "l2"): (0.73, "tab:orange"),
        ("lc", "l2"): (0.7, "tab:orange"),
        ("ld", "l2"): (0.74, "tab:orange"),
        ("la", "l3"): (1.19, "tab:orange"),
        ("lb", "l3"): (0.94, "tab:orange"),
        ("lc", "l3"): (0.92, "tab:orange"),
        ("ld", "l3"): (0.96, "tab:orange"),
        ("e10", "e16"): (0.47, "tab:blue"),
        ("e17", "e16"): (0.25, "tab:blue"),
        ("e17", "e10"): (0.35, "tab:orange"),
        ("e10", "e11"): (0.61, "tab:blue"),
        ("e14", "e16"): (0.44, "tab:blue"),
        ("e14", "e18"): (0.55, "tab:blue"),
        ("e10", "e18"): (0.59, "tab:blue"),
        ("e10", "e12"): (0.61, "tab:blue"),
        ("e14", "e11"): (0.57, "tab:blue"),
        ("e14", "e12"): (0.57, "tab:blue"),
        ("e13", "e11"): (0.5, "tab:blue"),
        ("e13", "e12"): (0.5, "tab:blue"),
        ("e14", "e13"): (0.65, "tab:orange"),
        ("e12", "e11"): (0.66, "tab:orange"),
    }

    ys = {i: float("inf") for i in tarzia_result}
    max_blue_g = 0
    max_blue_e = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = str(entry.properties["l1"])
        l2 = str(entry.properties["l2"])
        if multi != "2" or (l1, l2) not in tarzia_result:
            continue

        x, c = tarzia_result[(l1, l2)]
        y = entry.properties["energy_per_bb"]

        ys[(l1, l2)] = min((y, ys[(l1, l2)]))

    for (l1, l2), (x, c) in tarzia_result.items():
        if c == "tab:blue":
            max_blue_g = max((x, max_blue_g))
            if ys[(l1, l2)] != float("inf"):
                max_blue_e = max((ys[(l1, l2)], max_blue_e))

    rng = np.random.default_rng(12345)
    g_fps = 0
    e_fps = 0
    for (l1, l2), (x, c) in tarzia_result.items():
        if c == "tab:blue":
            xval = 0

        elif c == "tab:orange":
            xval = 1

            if x < max_blue_g:
                logging.info("g FP: %s", (l1, l2))
                g_fps += 1
            if ys[(l1, l2)] < max_blue_e:
                logging.info("e FP: %s", (l1, l2))
                e_fps += 1

        ax.scatter(
            (rng.random() - 0.5) * 0.6 + xval,
            x,
            c=c,
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
        )
        ax1.scatter(
            (rng.random() - 0.5) * 0.6 + xval,
            ys[(l1, l2)],
            c=c,
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"$g_{\mathrm{avg}}$", fontsize=16)
    ax1.set_ylabel(f"$m=2$ {eb_str()}", fontsize=16)
    ax.set_ylim(0, None)
    ax1.set_ylim(0, None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["forms $cis$ cage", "not"])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["forms $cis$ cage", "not"])
    ax.axvline(x=0.5, c="k")
    ax1.axvline(x=0.5, c="k")
    ax.axhline(y=max_blue_g, c="k")
    ax1.axhline(y=max_blue_e, c="k")
    ax.set_xlim(-0.5, 1.5)
    ax1.set_xlim(-0.5, 1.5)
    ax.set_title(f"FP: {g_fps}", fontsize=16)
    ax1.set_title(f"FP: {e_fps}", fontsize=16)

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


def study_2_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(5, 5))

    targets = {
        ("lf", "l2"): {
            "2": float("inf"),
            "3": float("inf"),
            "4": float("inf"),
        },
        ("lf", "ls2"): {
            "2": float("inf"),
            "3": float("inf"),
            "4": float("inf"),
        },
        ("lf", "ls3"): {
            "2": float("inf"),
            "3": float("inf"),
            "4": float("inf"),
        },
        ("lf", "l3"): {
            "2": float("inf"),
            "3": float("inf"),
            "4": float("inf"),
        },
        ("lf", "ls10"): {
            "2": float("inf"),
            "3": float("inf"),
            "4": float("inf"),
        },
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        l1 = str(entry.properties["l1"])
        l2 = str(entry.properties["l2"])
        if (l1, l2) not in targets:
            continue

        targets[(l1, l2)][str(entry.properties["multiplier"])] = min(
            (
                entry.properties["energy_per_bb"],
                targets[(l1, l2)][str(entry.properties["multiplier"])],
            )
        )

    for pair, edict in targets.items():
        min_e = min(edict.values())
        ax.plot(
            [int(i) for i, ed in edict.items() if ed != float("inf")],
            [ed - min_e for ed in edict.values() if ed != float("inf")],
            alpha=1.0,
            mec="k",
            markersize=8,
            marker="o",
            label=pair,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"rel. {eb_str()}", fontsize=16)
    ax.set_ylim(0, None)
    ax.legend(fontsize=16)
    ax.set_xticks([2, 3, 4])
    ax.set_xticklabels([2, 3, 4], fontsize=16)
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


def study_5_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(5, 5))

    tmap = {"mix1": "tab:blue", "mix2": "tab:orange", "mix3": "tab:red"}
    targets = {
        "mix1": {
            "0": float("inf"),
            "1": float("inf"),
            "2": float("inf"),
        },
        "mix2": {
            "0": float("inf"),
            "1": float("inf"),
            "2": float("inf"),
        },
        "mix3": {
            "0": float("inf"),
            "1": float("inf"),
            "2": float("inf"),
        },
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        targets[entry.properties["mix"]][str(entry.properties["sidx"])] = min(
            (
                entry.properties["energy_per_bb"],
                targets[entry.properties["mix"]][
                    str(entry.properties["sidx"])
                ],
            )
        )
        ax.scatter(
            entry.properties["sidx"],
            entry.properties["energy_per_bb"],
            c=tmap[entry.properties["mix"]],
            alpha=0.5,
            ec="none",
            s=60,
            marker="o",
        )
        logging.info(
            "E for %s, %s is %s (%s)",
            entry.properties["mix"],
            entry.properties["sidx"],
            round(entry.properties["energy_per_bb"], 2),
            entry.key,
        )

    for pair, edict in targets.items():
        ax.plot(
            [int(i) for i, ed in edict.items() if ed != float("inf")],
            [ed for ed in edict.values() if ed != float("inf")],
            alpha=1.0,
            c=tmap[pair],
            mec="k",
            markersize=8,
            marker="o",
            label=pair,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_yscale("log")
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.legend(fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2:2:1:1", "2:2:0:2", "2:2:2:0"], fontsize=16)

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


def study_6_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(8, 5))

    multis = {
        1: ("tab:blue", -0.2),
        2: ("tab:orange", 0.0),
        3: ("tab:green", 0.2),
    }

    xs = {}
    lbls = set()
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if entry.properties["mix"] not in xs:
            xs[entry.properties["mix"]] = len(xs)

        x = xs[entry.properties["mix"]]
        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]

        lbl = multi
        ax.scatter(
            x + multis[multi][1],
            y,
            c=multis[multi][0],
            alpha=1,
            ec="k",
            s=80,
            label=lbl if lbl not in lbls else None,
        )
        lbls.add(lbl)
        logging.info(
            "E for %s is %s",
            entry.key,
            round(entry.properties["energy_per_bb"], 2),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_yscale("log")
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.legend(fontsize=16)
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16)

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
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        x = [i[0] for i in pairs].index((l1, l2))
        x_count[multi][x] += 1
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
    ax.set_xticklabels(
        ["_".join(i) for i in [i[0] for i in pairs]], rotation=90
    )
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
    parser.add_argument(
        "--study1",
        action="store_true",
        help="set to run and or visualise case study 1 (w/o --run, only viz)",
    )
    parser.add_argument(
        "--study3",
        action="store_true",
        help="set to run and or visualise case study 3 (w/o --run, only viz)",
    )
    parser.add_argument(
        "--study4",
        action="store_true",
        help="set to run and or visualise case study 4 (w/o --run, only viz)",
    )
    parser.add_argument(
        "--study5",
        action="store_true",
        help="set to run and or visualise case study 5 (w/o --run, only viz)",
    )
    parser.add_argument(
        "--study6",
        action="store_true",
        help="set to run and or visualise case study 6 (w/o --run, only viz)",
    )
    parser.add_argument(
        "--study7",
        action="store_true",
        help="set to run and or visualise case study 7 (w/o --run, only viz)",
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
    bb_config: cgx.scram.BuildingBlockConfiguration | None,
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


def define_pairs(
    pairs_to_predict: abc.Sequence[
        tuple[abc.Sequence[str]], abc.Sequence[int]
    ],
    ligand_types: dict[str, str],
) -> dict[str, dict[str, cgx.molecular.Precursor | tuple | int]]:
    """Define pairs from provided information."""
    pairs = {}
    for (large, small), multis in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        elif ligand_types[large] == "twoarm":
            large_prec = cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c)

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

        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "multipliers": multis,
            "vdw_cutoff": 2,
        }
    return pairs


def case_study_1_2(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run case study 1/2 studying Pd(II) heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgen_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
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
            "dd": 6.0,
            "de": 5.7,
            "dde": 134,
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
        "l2": {"ba": 2.8, "aa": 5.0, "bac": 150, "s": 0.0},
        "ls2": {"ba": 2.8, "aa": 4.7, "bac": 144, "s": 0.0},
        "ls3": {"ba": 2.8, "aa": 5.0, "bac": 153, "s": 0.5},
        "l3": {"ba": 2.8, "aa": 5.3, "bac": 164, "s": 0.0},
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
            "dd": 2.3,
            "de": 8.3,
            "dde": 148,
            "eg": 2.4,
            "gb": 2.8,
        },
        "lc": {
            "egb": 90,
            "deg": 150,
            "dd": 2.3,
            "de": 8.3,
            "dde": 150,
            "eg": 2.4,
            "gb": 2.8,
        },
        "ld": {
            "egb": 90,
            "deg": 150,
            "dd": 2.3,
            "de": 8.3,
            "dde": 165,
            "eg": 2.4,
            "gb": 2.8,
        },
    }

    ligand_types = {
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
        "e16": "twoarm",
        "e18": "twoarm",
        "l2": "twoarm",
        "ls2": "twoarm",
        "ls3": "stwoarm",
        "ls10": "twoarm",
        "l1": "twoarm",
        "l3": "twoarm",
    }

    pairs_to_predict = [
        # large, small.
        (("la", "l1"), (2,)),
        (("lb", "l1"), (2,)),
        (("lc", "l1"), (2,)),
        (("ld", "l1"), (2,)),
        (("la", "l2"), (2,)),
        (("lb", "l2"), (2,)),
        (("lc", "l2"), (2,)),
        (("ld", "l2"), (2,)),
        (("la", "l3"), (2,)),
        (("lb", "l3"), (2,)),
        (("lc", "l3"), (2,)),
        (("ld", "l3"), (2,)),
        (("e10", "e16"), (2,)),
        (("e17", "e16"), (2,)),
        (("e17", "e10"), (2,)),
        (("e10", "e11"), (2,)),
        (("e14", "e16"), (2,)),
        (("e14", "e18"), (2,)),
        (("e10", "e18"), (2,)),
        (("e10", "e12"), (2,)),
        (("e14", "e11"), (2,)),
        (("e14", "e12"), (2,)),
        (("e13", "e11"), (2,)),
        (("e13", "e12"), (2,)),
        (("e14", "e13"), (2,)),
        (("e12", "e11"), (2,)),
        (("lf", "l2"), (2, 3, 4)),
        (("lf", "ls2"), (2, 3, 4)),
        (("lf", "ls3"), (2, 3, 4)),
        (("lf", "l3"), (2, 3, 4)),
        (("lf", "ls10"), (2, 3, 4)),
    ]
    pairs = define_pairs(pairs_to_predict, ligand_types)

    if run:
        for pair in pairs:
            forcefield = precursors_to_forcefield(
                pair=pair,
                large=pairs[pair]["large"],
                small=pairs[pair]["small"],
                large_meas=ligand_measures[pairs[pair]["large_name"]],
                small_meas=ligand_measures[pairs[pair]["small_name"]],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
                constant_definer_dict=constant_definer_dict,
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
                    )

                collated_entries = [
                    i
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if "_f-" not in i.key
                    and i.properties["pair"] == pair
                    and i.properties["multiplier"] == multiplier
                    and "lowest_e_of_mash" in i.properties
                ]

                # Generate a series of new ffs.
                forcefield_lib = tuple(
                    generate_nearby_forcefields(
                        forcefield=forcefield,
                        actual_present_bead_elements={
                            i.__class__.__name__
                            for i in stk.BuildingBlock.init_from_rdkit_mol(
                                atomlite.json_to_rdkit(
                                    collated_entries[0].molecule
                                )
                            ).get_atoms()
                        },
                    ).yield_forcefields(),
                )
                logging.info(
                    "exploring %s molecules with %s ffs",
                    len(collated_entries),
                    len(forcefield_lib),
                )

                for (ffidx, temp_forcefield), entry in it.product(
                    enumerate(forcefield_lib), collated_entries
                ):
                    name = entry.key + f"_f-{ffidx}"

                    fina_mol_file = ffcalculation_dir / f"{name}_ff.mol"
                    if not fina_mol_file.exists():
                        logging.info("optimising %s with ff %s", name, ffidx)
                        current_cage = stk.BuildingBlock.init_from_rdkit_mol(
                            atomlite.json_to_rdkit(entry.molecule)
                        )
                        conformer = cgx.utilities.run_optimisation(
                            assigned_system=temp_forcefield.assign_terms(
                                current_cage, name, ffcalculation_dir
                            ),
                            name=name,
                            file_suffix=f"ff{ffidx}",
                            output_dir=ffcalculation_dir,
                            platform=None,
                        )
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            fina_mol_file
                        )
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_molecule(molecule=conformer.molecule, key=name)
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_properties(
                            key=name,
                            property_dict={
                                "energy_decomposition": (
                                    conformer.energy_decomposition,  # type:ignore[dict-item]
                                ),
                                "source": conformer.source,
                                "optimised": True,
                                "energy_per_bb": (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                ),
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        conformer.molecule
                                    )["min_distance"]
                                ),
                            },
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

    study_2_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_9.png",
    )
    parity_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_8.png",
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_5.png",
    )
    for pair in pairs:
        make_plot(
            database_path=database_path,
            target_pair=pair,
            figure_dir=figure_dir,
            filename=f"mgen_1_{pair}.png",
        )

    sterics_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_6.png",
    )

    binder_vector_angles_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_7.png",
    )
    raise SystemExit(
        "main fig 1: because we do not specify stericd exactly, we do screen, so here is just change in binder angle -- show propotion"
    )
    raise SystemExit(
        "main fig 2: show sterics for ufos again -- redo to fit the new model"
    )
    raise SystemExit("add another grid to sterics to show we can do all that")
    raise SystemExit("recompute xrds based on new model")
    raise SystemExit(
        "big si fig shows the cis-het structures - need to recheck their calcs"
    )
    raise SystemExit(
        "one bar chart - top - min E of original. then bar of prop in "
        "target 4, target 3, 3, 4"
    )
    raise SystemExit("show the known systems in the make plot")
    raise SystemExit(
        "a plot that shows the three/2? distinct case studies - for main"
    )
    raise SystemExit("rethink binders, because it is not handling minus")


def case_study_3(run: bool) -> None:
    """Run case study 3 studying Rh heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs3_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs3_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs3_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs3_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs3"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs3.db"

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
                    )

                collated_entries = [
                    i
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if "_f-" not in i.key
                    and i.properties["pair"] == pair
                    and i.properties["multiplier"] == multiplier
                    and "lowest_e_of_mash" in i.properties
                ]

                # Generate a series of new ffs.
                forcefield_lib = tuple(
                    generate_nearby_forcefields(
                        forcefield=forcefield,
                        actual_present_bead_elements={
                            i.__class__.__name__
                            for i in stk.BuildingBlock.init_from_rdkit_mol(
                                atomlite.json_to_rdkit(
                                    collated_entries[0].molecule
                                )
                            ).get_atoms()
                        },
                    ).yield_forcefields(),
                )
                logging.info(
                    "exploring %s molecules with %s ffs",
                    len(collated_entries),
                    len(forcefield_lib),
                )

                for (ffidx, temp_forcefield), entry in it.product(
                    enumerate(forcefield_lib), collated_entries
                ):
                    name = entry.key + f"_f-{ffidx}"

                    fina_mol_file = ffcalculation_dir / f"{name}_ff.mol"
                    if not fina_mol_file.exists():
                        logging.info("optimising %s with ff %s", name, ffidx)
                        current_cage = stk.BuildingBlock.init_from_rdkit_mol(
                            atomlite.json_to_rdkit(entry.molecule)
                        )
                        conformer = cgx.utilities.run_optimisation(
                            assigned_system=temp_forcefield.assign_terms(
                                current_cage, name, ffcalculation_dir
                            ),
                            name=name,
                            file_suffix=f"ff{ffidx}",
                            output_dir=ffcalculation_dir,
                            platform=None,
                        )
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            fina_mol_file
                        )
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_molecule(molecule=conformer.molecule, key=name)
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_properties(
                            key=name,
                            property_dict={
                                "energy_decomposition": (
                                    conformer.energy_decomposition,  # type:ignore[dict-item]
                                ),
                                "source": conformer.source,
                                "optimised": True,
                                "energy_per_bb": (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                ),
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        conformer.molecule
                                    )["min_distance"]
                                ),
                            },
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


def case_study_4(run: bool) -> None:
    """Run case study 4 studying PW heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs4_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs4_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs4_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs4_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs4"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs4.db"

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        "cs41a": {"ba": 1.5, "aa": 9.5, "bac": 145},
        "cs41b": {"ba": 5.7, "aa": 2.4, "bac": 145},
        "cs41c": {"ba": 1.5, "aa": 9.5, "bac": 150},
        "cs41d": {"ba": 5.7, "aa": 2.4, "bac": 150},
        "cs490": {"ba": 1.5, "aa": 5.9, "bac": 135},
    }
    ligand_types = {
        "cs41a": "twoarm",
        "cs41b": "twoarm",
        "cs41c": "twoarm",
        "cs41d": "twoarm",
        "cs490": "twoarm",
    }
    pairs_to_predict = [
        # large, small.
        (("cs490", "cs41c"), (2, 3, 4, 5, 6, 7, 8, 9)),
        (("cs490", "cs41d"), (2, 3, 4, 5, 6, 7, 8, 9)),
        (("cs490", "cs41a"), (9,)),
        (("cs490", "cs41b"), (9,)),
    ]

    pairs = define_pairs(pairs_to_predict, ligand_types)

    cs4_definer_dict = {
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
                constant_definer_dict=cs4_definer_dict,
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
                    )

                collated_entries = [
                    i
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if "_f-" not in i.key
                    and i.properties["pair"] == pair
                    and i.properties["multiplier"] == multiplier
                    and "lowest_e_of_mash" in i.properties
                ]

                # Generate a series of new ffs.
                forcefield_lib = tuple(
                    generate_nearby_forcefields(
                        forcefield=forcefield,
                        actual_present_bead_elements={
                            i.__class__.__name__
                            for i in stk.BuildingBlock.init_from_rdkit_mol(
                                atomlite.json_to_rdkit(
                                    collated_entries[0].molecule
                                )
                            ).get_atoms()
                        },
                    ).yield_forcefields(),
                )
                logging.info(
                    "exploring %s molecules with %s ffs",
                    len(collated_entries),
                    len(forcefield_lib),
                )

                for (ffidx, temp_forcefield), entry in it.product(
                    enumerate(forcefield_lib), collated_entries
                ):
                    name = entry.key + f"_f-{ffidx}"

                    fina_mol_file = ffcalculation_dir / f"{name}_ff.mol"
                    if not fina_mol_file.exists():
                        logging.info("optimising %s with ff %s", name, ffidx)
                        current_cage = stk.BuildingBlock.init_from_rdkit_mol(
                            atomlite.json_to_rdkit(entry.molecule)
                        )
                        conformer = cgx.utilities.run_optimisation(
                            assigned_system=temp_forcefield.assign_terms(
                                current_cage, name, ffcalculation_dir
                            ),
                            name=name,
                            file_suffix=f"ff{ffidx}",
                            output_dir=ffcalculation_dir,
                            platform=None,
                        )
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            fina_mol_file
                        )
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_molecule(molecule=conformer.molecule, key=name)
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_properties(
                            key=name,
                            property_dict={
                                "energy_decomposition": (
                                    conformer.energy_decomposition,  # type:ignore[dict-item]
                                ),
                                "source": conformer.source,
                                "optimised": True,
                                "energy_per_bb": (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                ),
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        conformer.molecule
                                    )["min_distance"]
                                ),
                            },
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


def case_study_5(run: bool) -> None:
    """Run case study 5 studying Pd(II) heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs5_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs5_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs5_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs5_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs5"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs5.db"

    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        c2bead_d,
        a2bead_d,
        e2bead_d,
        binder_bead,
        tetra_bead,
        steric_bead,
    )

    ligand_measures = {
        "lin": {"ba": 2.8, "aa": 1.5, "bac": 180},
        "mxy": {"be": 7.6, "ee": 5.0, "bed": 90},
        "pxy": {"be": 7.7, "ee": 5.8, "bed": 110},
        "fxy": {"be": 7.7, "ee": 5.8, "bed": 120},
        "tetra": {"mb": 2.0, "bmb": 90},
        "sqp": {"bf": 2.0, "bfb": 90, "fbm": 90},
    }
    stoichs_mcll = ((2, 2, 1, 1), (2, 2, 0, 2), (2, 2, 2, 0))
    mixtures = {
        "mix1": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "mxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
            "stoichiometries": stoichs_mcll,
            "vdw_cutoff": 2,
        },
        "mix2": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "pxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
            "stoichiometries": stoichs_mcll,
            "vdw_cutoff": 2,
        },
        "mix3": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "fxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
            "stoichiometries": stoichs_mcll,
            "vdw_cutoff": 2,
        },
    }

    cg_scale = 2
    if run:
        for mix, mdict in mixtures.items():
            cgx.molecular.BeadLibrary(present_beads)
            bent_name, bent = mdict["bent"]
            linear_name, linear = mdict["linear"]
            corner_name, corner = mdict["corner"]
            tetra_name, tetra = mdict["tetra"]
            cs5_definer_dict = {
                # Bent.
                "be": (
                    "bond",
                    ligand_measures[bent_name]["be"] / cg_scale,
                    1e5,
                ),
                "ed": (
                    "bond",
                    ligand_measures[bent_name]["ee"] / 2 / cg_scale,
                    1e5,
                ),
                "bed": (
                    "angle",
                    ligand_measures[bent_name]["bed"],
                    1e2,
                ),
                "ede": ("angle", 180, 1e2),
                # Corner.
                "mb": (
                    "bond",
                    ligand_measures[tetra_name]["mb"] / cg_scale,
                    1e5,
                ),
                "bf": (
                    "bond",
                    ligand_measures[corner_name]["bf"] / cg_scale,
                    1e5,
                ),
                "bmb": ("pyramid", ligand_measures[tetra_name]["bmb"], 1e2),
                "bfb": ("angle", ligand_measures[corner_name]["bfb"], 1e2),
                "fbm": ("angle", ligand_measures[corner_name]["fbm"], 1e2),
                # Linear.
                "ba": (
                    "bond",
                    ligand_measures[linear_name]["ba"] / cg_scale,
                    1e5,
                ),
                "ac": (
                    "bond",
                    ligand_measures[linear_name]["aa"] / 2 / cg_scale,
                    1e5,
                ),
                "bac": (
                    "angle",
                    ligand_measures[linear_name]["bac"],
                    1e2,
                ),
                "aca": ("angle", 180, 1e2),
                # Bonds.
                # Constant.
                "mba": ("angle", 180, 1e2),
                "mbe": ("angle", 180, 1e2),
                # Nonbondeds.
                "m": ("nb", 10.0, 1.0),
                "d": ("nb", 10.0, 1.0),
                "e": ("nb", 10.0, 1.0),
                "a": ("nb", 10.0, 1.0),
                "b": ("nb", 10.0, 1.0),
                "c": ("nb", 10.0, 1.0),
                "f": ("nb", 10.0, 1.0),
            }

            forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=f"{mix}ff",
                prefix=f"{mix}ff",
                vdw_bond_cutoff=mixtures[mix]["vdw_cutoff"],
                present_beads=present_beads,
                definer_dict=cs5_definer_dict,
            )

            bent_bb = cgx.utilities.optimise_ligand(
                molecule=bent.get_building_block(),
                name=f"{mix}_{bent.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            bent_bb.write(
                str(ligand_dir / f"{mix}_{bent.get_name()}_optl.mol")
            )

            linear_bb = cgx.utilities.optimise_ligand(
                molecule=linear.get_building_block(),
                name=f"{mix}_{linear.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            linear_bb.write(
                str(ligand_dir / f"{mix}_{linear.get_name()}_optl.mol")
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(
                str(ligand_dir / f"{mix}_{tetra.get_name()}_optl.mol")
            )

            corner_bb = corner.get_building_block()
            corner_bb.write(
                str(ligand_dir / f"{mix}_{corner.get_name()}_optl.mol")
            )

            for sidx, stoichiometry in enumerate(mdict["stoichiometries"]):
                logging.info("doing: %s, stoich %s", mix, stoichiometry)
                gtype = f"{stoichiometry[0]}P{stoichiometry[0] * 2}"

                if sum(stoichiometry[1:]) != stoichiometry[0] * 2:
                    raise RuntimeError

                if stoichiometry[2] == 0:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        corner_bb: stoichiometry[1],
                        bent_bb: stoichiometry[3],
                    }
                elif stoichiometry[3] == 0:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        linear_bb: stoichiometry[2],
                        corner_bb: stoichiometry[1],
                    }
                else:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        linear_bb: stoichiometry[2],
                        corner_bb: stoichiometry[1],
                        bent_bb: stoichiometry[3],
                    }

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts=b_counts,
                    graph_type=gtype,
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
                        name = f"{mix}_s{sidx}_{idx}_{midx}_b{bb_config.idx}"

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

                        except (ValueError, KeyError):
                            continue

                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )
                        # Optimise and save.
                        logging.info("building %s", name)

                        try:
                            potential_names = [
                                f"{mix}_s{sidx}_{idx}_"
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
                                    "linear": linear_name,
                                    "tetra": tetra_name,
                                    "corner": corner_name,
                                    "bent": bent_name,
                                    "mix": mix,
                                    "num_components": num_components,
                                    "num_bbs": (
                                        iterator.get_num_building_blocks()
                                    ),
                                    "sidx": sidx,
                                    "stoichiometry": stoichiometry,
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

                    if len(generated_conformers) == 0:
                        continue
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

                collated_entries = [
                    i
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if "_f-" not in i.key
                    and i.properties["mix"] == mix
                    and i.properties["sidx"] == sidx
                    and "lowest_e_of_mash" in i.properties
                ]

                # Generate a series of new ffs.
                forcefield_lib = tuple(
                    generate_nearby_forcefields(
                        forcefield=forcefield,
                        actual_present_bead_elements={
                            i.__class__.__name__
                            for i in stk.BuildingBlock.init_from_rdkit_mol(
                                atomlite.json_to_rdkit(
                                    collated_entries[0].molecule
                                )
                            ).get_atoms()
                        },
                    ).yield_forcefields(),
                )
                logging.info(
                    "exploring %s molecules with %s ffs",
                    len(collated_entries),
                    len(forcefield_lib),
                )

                for (ffidx, temp_forcefield), entry in it.product(
                    enumerate(forcefield_lib), collated_entries
                ):
                    name = entry.key + f"_f-{ffidx}"

                    fina_mol_file = ffcalculation_dir / f"{name}_ff.mol"
                    if not fina_mol_file.exists():
                        logging.info("optimising %s with ff %s", name, ffidx)
                        current_cage = stk.BuildingBlock.init_from_rdkit_mol(
                            atomlite.json_to_rdkit(entry.molecule)
                        )
                        conformer = cgx.utilities.run_optimisation(
                            assigned_system=temp_forcefield.assign_terms(
                                current_cage, name, ffcalculation_dir
                            ),
                            name=name,
                            file_suffix=f"ff{ffidx}",
                            output_dir=ffcalculation_dir,
                            platform=None,
                        )
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            fina_mol_file
                        )
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_molecule(molecule=conformer.molecule, key=name)
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_properties(
                            key=name,
                            property_dict={
                                "energy_decomposition": (
                                    conformer.energy_decomposition,  # type:ignore[dict-item]
                                ),
                                "source": conformer.source,
                                "optimised": True,
                                "energy_per_bb": (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                ),
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        conformer.molecule
                                    )["min_distance"]
                                ),
                            },
                        )

    study_5_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )


def case_study_6(run: bool) -> None:
    """Run case study 6 studying Tri + Di homoleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs6_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs6_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs6_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs6_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs6"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs6.db"

    present_beads = (cbead_d, abead_d, binder_bead, trigonal_bead)
    stoichiometry_t_d = (2, 3)
    vdw_cutoff = 2
    multipliers = (1, 2, 3)
    # Very approximate.
    ligand_measures = {
        "cs6l1": {"ba": 3.8 / 3, "aa": 3.8 / 3, "bac": 180},
        "cs6l2": {"ba": 5.7 / 3, "aa": 5.7 / 3, "bac": 180},
        "cs6l5": {"ba": 8.0 / 3, "aa": 8.0 / 3, "bac": 180},
        "cs6l6": {"ba": 9.9 / 3, "aa": 9.9 / 3, "bac": 180},
        "cs6l9": {"ba": 14.2 / 3, "aa": 14.2 / 3, "bac": 180},
        "cs6zr1": {"bnb": 60, "nb": 3.5},
        "cs6zr2": {"bnb": 70, "nb": 3.5},
        "cs6cc31": {"bnb": 120, "nb": 2.9},
        "cs6cc32": {"ba": 1.5, "aa": 1.5, "bac": 105},
    }

    mixtures = {
        "l1zr1": {
            "linear": (
                "cs6l1",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l1zr2": {
            "linear": (
                "cs6l1",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l2zr1": {
            "linear": (
                "cs6l2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l2zr2": {
            "linear": (
                "cs6l2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l5zr1": {
            "linear": (
                "cs6l5",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l5zr2": {
            "linear": (
                "cs6l5",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l6zr1": {
            "linear": (
                "cs6l6",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l6zr2": {
            "linear": (
                "cs6l6",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l9zr1": {
            "linear": (
                "cs6l9",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "l9zr2": {
            "linear": (
                "cs6l9",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
        "cc3": {
            "linear": (
                "cs6cc32",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6cc31",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
        },
    }

    cg_scale = 2
    if run:
        for mix, mdict in mixtures.items():
            cgx.molecular.BeadLibrary(present_beads)
            linear_name, linear = mdict["linear"]
            trigonal_name, trigonal = mdict["trigonal"]

            cs6_definer_dict = {
                # Trigonal.
                "nb": (
                    "bond",
                    ligand_measures[trigonal_name]["nb"] / cg_scale,
                    1e5,
                ),
                "bnb": ("angle", ligand_measures[trigonal_name]["bnb"], 1e2),
                # Linear.
                "ba": (
                    "bond",
                    ligand_measures[linear_name]["ba"] / cg_scale,
                    1e5,
                ),
                "ac": (
                    "bond",
                    ligand_measures[linear_name]["aa"] / 2 / cg_scale,
                    1e5,
                ),
                "bac": (
                    "angle",
                    ligand_measures[linear_name]["bac"],
                    1e2,
                ),
                "aca": ("angle", 180, 1e2),
                # Constant.
                "nba": ("angle", 180, 1e2),
                # Nonbondeds.
                "n": ("nb", 10.0, 1.0),
                "a": ("nb", 10.0, 1.0),
                "b": ("nb", 10.0, 1.0),
                "c": ("nb", 10.0, 1.0),
            }

            forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=f"{mix}ff",
                prefix=f"{mix}ff",
                vdw_bond_cutoff=vdw_cutoff,
                present_beads=present_beads,
                definer_dict=cs6_definer_dict,
            )
            linear_bb = cgx.utilities.optimise_ligand(
                molecule=linear.get_building_block(),
                name=f"{mix}_{linear.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            linear_bb.write(
                str(ligand_dir / f"{mix}_{linear.get_name()}_optl.mol")
            )

            trigonal_bb = cgx.utilities.optimise_ligand(
                molecule=trigonal.get_building_block(),
                name=trigonal.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            trigonal_bb.write(
                str(ligand_dir / f"{mix}_{trigonal.get_name()}_optl.mol")
            )

            for multiplier in multipliers:
                logging.info("doing: mix %s, multi %s", mix, multiplier)
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        trigonal_bb: stoichiometry_t_d[0] * multiplier,
                        linear_bb: stoichiometry_t_d[1] * multiplier,
                    },
                    graph_type=f"{stoichiometry_t_d[0] * multiplier}"
                    f"P{stoichiometry_t_d[1] * multiplier}",
                    graph_set="rx",
                )
                logging.info(
                    "graph iteration has %s graphs", iterator.count_graphs()
                )

                for idx, topology_code in enumerate(iterator.yield_graphs()):
                    generated_conformers = []
                    for midx, scale in enumerate(attempts):
                        name = f"{mix}_{multiplier}_{idx}_{midx}"

                        try:
                            if isinstance(scale, str) and "regraphed" in scale:
                                constructed_molecule = get_regraphed_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=None,
                                )

                            else:
                                constructed_molecule = get_vertexset_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=None,
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
                                f"{mix}_{multiplier}_{idx}_{nmash_idx}"
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
                                    "tri": trigonal_name,
                                    "di": linear_name,
                                    "mix": mix,
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
                    )

                collated_entries = [
                    i
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if "_f-" not in i.key
                    and i.properties["mix"] == mix
                    and i.properties["multiplier"] == multiplier
                    and "lowest_e_of_mash" in i.properties
                ]

                # Generate a series of new ffs.
                forcefield_lib = tuple(
                    generate_nearby_forcefields(
                        forcefield=forcefield,
                        actual_present_bead_elements={
                            i.__class__.__name__
                            for i in stk.BuildingBlock.init_from_rdkit_mol(
                                atomlite.json_to_rdkit(
                                    collated_entries[0].molecule
                                )
                            ).get_atoms()
                        },
                    ).yield_forcefields(),
                )
                logging.info(
                    "exploring %s molecules with %s ffs",
                    len(collated_entries),
                    len(forcefield_lib),
                )

                for (ffidx, temp_forcefield), entry in it.product(
                    enumerate(forcefield_lib), collated_entries
                ):
                    name = entry.key + f"_f-{ffidx}"

                    fina_mol_file = ffcalculation_dir / f"{name}_ff.mol"
                    if not fina_mol_file.exists():
                        logging.info("optimising %s with ff %s", name, ffidx)
                        current_cage = stk.BuildingBlock.init_from_rdkit_mol(
                            atomlite.json_to_rdkit(entry.molecule)
                        )
                        conformer = cgx.utilities.run_optimisation(
                            assigned_system=temp_forcefield.assign_terms(
                                current_cage, name, ffcalculation_dir
                            ),
                            name=name,
                            file_suffix=f"ff{ffidx}",
                            output_dir=ffcalculation_dir,
                            platform=None,
                        )
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            fina_mol_file
                        )
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_molecule(molecule=conformer.molecule, key=name)
                        cgx.utilities.AtomliteDatabase(
                            database_path
                        ).add_properties(
                            key=name,
                            property_dict={
                                "energy_decomposition": (
                                    conformer.energy_decomposition,  # type:ignore[dict-item]
                                ),
                                "source": conformer.source,
                                "optimised": True,
                                "energy_per_bb": (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                ),
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        conformer.molecule
                                    )["min_distance"]
                                ),
                            },
                        )

    study_6_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )


def case_study_7(run: bool) -> None:
    """Run case study 7 studying Pd(II) heteroleptic systems."""


def main() -> None:
    """Run script."""
    args = _parse_args()

    if args.study1:
        case_study_1_2(args.run)
    if args.study3:
        case_study_3(args.run)
    if args.study4:
        case_study_4(args.run)
    if args.study5:
        case_study_5(args.run)
    if args.study6:
        case_study_6(args.run)
    if args.study7:
        case_study_7(args.run)


if __name__ == "__main__":
    main()
