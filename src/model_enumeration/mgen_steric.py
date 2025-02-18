"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import warnings
from collections import defaultdict

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_known import convert_coordinates
from model_enumeration.mgen_utilities import (
    StericTwoC1Arm,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)
from model_enumeration.utilities import convert_topo, eb_str, topology_cmap

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
    return parser.parse_args()


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    database.add_properties(
        key=name,
        property_dict={
            "forcefield_dict": forcefield.get_forcefield_dictionary(),
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=properties["energy_decomposition"],
                number_building_blocks=num_building_blocks,
            ),
        },
    )

    g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
    bond_data = g_measure.calculate_bonds(final_molecule)
    bond_data = {str("_".join(i)): bond_data[i] for i in bond_data}
    angle_data = g_measure.calculate_angles(final_molecule)
    angle_data = {str("_".join(i)): angle_data[i] for i in angle_data}
    dihedral_data = g_measure.calculate_torsions(
        molecule=final_molecule,
        absolute=True,
    )
    database.add_properties(
        key=name,
        property_dict={
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
        },
    )

    ligands = stko.molecule_analysis.DecomposeMOC().decompose(
        molecule=final_molecule,
        metal_atom_nos=(46,),
    )

    # Get the bg angles.
    c_binder_binder_angles = []
    d_binder_binder_angles = []
    for lig in ligands:
        if lig.get_num_atoms() == 8:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ga]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = True
        elif lig.get_num_atoms() == 6:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ba]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = False

        if as_building_block.get_num_functional_groups() != 2:  # noqa: PLR2004
            raise RuntimeError

        vectors = [
            as_building_block.get_centroid(atom_ids=fg.get_bonder_ids())
            - as_building_block.get_centroid(atom_ids=fg.get_deleter_ids())
            for fg in as_building_block.get_functional_groups()
        ]
        normed = [i / np.linalg.norm(i) for i in vectors]
        angle = np.degrees(
            stko.vector_angle(vector1=normed[0], vector2=normed[1])
        )
        if converging:
            c_binder_binder_angles.append(angle)
        else:
            d_binder_binder_angles.append(angle)

    database.add_properties(
        key=name,
        property_dict={
            "converging_binder_binder_angles": c_binder_binder_angles,
            "diverging_binder_binder_angles": d_binder_binder_angles,
        },
    )

    ss_dists = stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
        molecule=final_molecule,
        metal_atom_nos=(16,),
    )

    min_ss_value = min(ss_dists.values())
    max_ss_value = max(ss_dists.values())

    database.add_properties(
        key=name,
        property_dict={
            "min_ss_dist": min_ss_value,
            "max_ss_dist": max_ss_value,
        },
    )


def make_geom_grid(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    row_plot = {
        "ffx": "diverging_binder_binder_angles",
        "aay": "converging_binder_binder_angles",
        "xlim": (None, None),
        "ylim": (None, None),
        "xlbl": "observed rigid angle  [$^\\circ$]",
        "ylbl": "observed twistable angle  [$^\\circ$]",
    }

    cmaps = {
        "3P6-x": "tab:pink",
        "4P8-x": "tab:cyan",
    }
    lbls = set()
    for tstr in ("3P6", "4P8", "3P6-x", "4P8-x"):
        for entry in cgx.utilities.AtomliteDatabase(
            database_path
        ).get_entries():
            if tstr != entry.key.split("_")[1]:
                continue

            if (
                row_plot["ffx"] not in entry.properties
                or row_plot["aay"] not in entry.properties
            ):
                continue

            xs = entry.properties[row_plot["ffx"]]
            ys = entry.properties[row_plot["aay"]]

            try:
                colour = topology_cmap[tstr]
                label = (
                    convert_topo(tstr)
                    if convert_topo(tstr) not in lbls
                    else None
                )
            except KeyError:
                colour = cmaps[tstr]
                label = (
                    convert_topo(tstr.replace("-x", "")) + "-x"
                    if convert_topo(tstr.replace("-x", "")) + "-x" not in lbls
                    else None
                )

            zorder = 1
            alpha = 1
            ec = "none"
            lbls.add(label)

            ax.scatter(
                np.mean(xs),
                np.mean(ys),
                c=colour,
                alpha=alpha,
                edgecolor=ec,
                s=80,
                zorder=zorder,
                label=label,
                cmap="Blues_r",
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(row_plot["xlbl"], fontsize=16)
    ax.set_ylabel(row_plot["ylbl"], fontsize=16)
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


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmaps = {
        "3P6-x": "tab:pink",
        "4P8-x": "tab:cyan",
    }
    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.key.split("_")[1]
        if (
            "forcefield_dict" not in entry.properties
            or "forcefield_dict" not in entry.properties
        ):
            continue
        x = entry.properties["forcefield_dict"]["v_dict"]["s"]
        y = entry.properties["energy_per_bb"]

        datas[tstr][x].append(y)

    for tstr, tdatas in datas.items():
        try:
            colour = topology_cmap[tstr]
            label = convert_topo(tstr)
        except KeyError:
            colour = cmaps[tstr]
            label = convert_topo(tstr.replace("-x", "")) + "-x"

        ax.plot(
            list(tdatas),
            [min(tdatas[i]) for i in tdatas],
            alpha=1.0,
            marker="o",
            markerfacecolor=colour,
            mec="k",
            markersize=10,
            ls="-",
            label=label,
            c="k" if "s" in tstr else "w",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)

    ax.legend(ncol=1, fontsize=16)
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


def ss_plot(
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
        tstr = entry.key.split("_")[1]
        if (
            "forcefield_dict" not in entry.properties
            or "forcefield_dict" not in entry.properties
        ):
            continue
        x = entry.properties["forcefield_dict"]["v_dict"]["s"]
        if "min_ss_dist" not in entry.properties:
            continue
        y = entry.properties["min_ss_dist"]
        y2 = entry.properties["max_ss_dist"]

        try:
            if entry.properties["energy_per_bb"] < datas[tstr][x][1]:
                datas[tstr][x] = (y, y2, entry.properties["energy_per_bb"])
        except IndexError:
            datas[tstr][x] = (y, y2, entry.properties["energy_per_bb"])

    xlbl = r"$r_{s-s}$ [$\AA$]"

    cmaps = {
        "3P6-x": "tab:pink",
        "4P8-x": "tab:cyan",
    }
    for tstr, tdatas in datas.items():
        try:
            colour = topology_cmap[tstr]
            label = convert_topo(tstr)
        except KeyError:
            colour = cmaps[tstr]
            label = convert_topo(tstr.replace("-x", "")) + "-x"

        ax.fill_between(
            x=list(tdatas),
            y1=[tdatas[i][0] for i in tdatas],
            y2=[tdatas[i][1] for i in tdatas],
            alpha=0.2,
            facecolor=colour,
        )
        ax.plot(
            list(tdatas),
            [tdatas[i][0] for i in tdatas],
            alpha=1,
            c=colour,
            label=label,
        )
        ax.plot(
            list(tdatas),
            [tdatas[i][1] for i in tdatas],
            alpha=1,
            c=colour,
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgensteric_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgensteric_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgensteric_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgensteric_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgensteric"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgensteric.db"

    s_range = list(np.linspace(0.0, 7.0, 15))

    pair = "lf_l2"
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "lf"
    diverging = StericTwoC1Arm(
        bead=cbead_d, abead1=abead_d, steric_bead=steric_bead
    )
    diverging_name = "l2"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    combos = {"s": {"xr": s_range, "xl": "l2"}}
    topologies = ("3P6", "4P8", "3P6-x", "4P8-x")

    if args.run:
        for (cname, pair_range_dict), (textstr) in it.product(
            combos.items(), topologies
        ):
            for i, xp in enumerate(pair_range_dict["xr"]):
                # Rewrite each time.
                if "-x" in textstr:
                    ligand_measures = {
                        "lf": {
                            "egb": 120.0,
                            "deg": 180.0,
                            "dd": 7.87,
                            "de": 4.25,
                            "dde": 125.0,
                            "eg": 1.4,
                            "gb": 1.4,
                        },
                        "l2": {"ba": 2.8, "aa": 5.25, "bac": 165.0, "s": 0.0},
                    }
                    tstr = textstr.replace("-x", "")
                else:
                    ligand_measures = {
                        "lf": {
                            "egb": 120.0,
                            "deg": 180.0,
                            "dd": 8.0,
                            "de": 4.25,
                            "dde": 125.0,
                            "eg": 1.4,
                            "gb": 1.4,
                        },
                        "l2": {"ba": 2.8, "aa": 5.0, "bac": 150.0, "s": 0.0},
                    }
                    tstr = textstr

                ligand_measures[pair_range_dict["xl"]][cname] = xp

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    large=converging,
                    small=diverging,
                    large_meas=ligand_measures[converging_name],
                    small_meas=ligand_measures[diverging_name],
                    vdw_bond_cutoff=2,
                )

                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=converging.get_building_block(),
                    name=f"{converging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb = converging_bb.clone()

                tetra_bb = cgx.utilities.optimise_ligand(
                    molecule=tetra.get_building_block(),
                    name=f"{tetra.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                tetra_bb = tetra_bb.clone()

                diverging_bb = cgx.utilities.optimise_ligand(
                    molecule=diverging.get_building_block(),
                    name=f"{diverging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                diverging_bb = diverging_bb.clone()

                if i == 0:
                    generated_conformers = []
                    for attempt, scale in enumerate(
                        (
                            "from_0",
                            "spring",
                            "kamada",
                            1.1,
                            1.0,
                            0.9,
                            0.8,
                            0.7,
                            0.6,
                            0.5,
                        )
                    ):
                        actual_scale = (
                            1 if not isinstance(scale, float) else scale
                        )
                        name = f"scan_{textstr}_{cname}_{i}_{attempt}"
                        logging.info("building %s", name)

                        if tstr == "3P6":
                            constructed_molecule = stk.ConstructedMolecule(
                                stk.cage.M3L6(
                                    building_blocks={
                                        tetra_bb: (0, 1, 2),
                                        converging_bb: (3, 5, 7),
                                        diverging_bb: (4, 6, 8),
                                    },
                                    vertex_positions=None,
                                    scale_multiplier=actual_scale,
                                )
                            )
                            num_bbs = 9

                        elif tstr == "4P8":
                            constructed_molecule = stk.ConstructedMolecule(
                                cgx.topologies.CGM4L8(
                                    building_blocks={
                                        tetra_bb: (0, 1, 2, 3),
                                        converging_bb: (4, 6, 8, 10),
                                        diverging_bb: (5, 7, 9, 11),
                                    },
                                    vertex_positions=None,
                                    scale_multiplier=actual_scale,
                                )
                            )
                            num_bbs = 12

                        else:
                            raise NotImplementedError

                        if scale == "spring":
                            stko_graph = stko.Network.init_from_molecule(
                                constructed_molecule
                            )
                            nx_positions = nx.spring_layout(
                                stko_graph.get_graph(), dim=3
                            )
                            constructed_molecule = convert_coordinates(
                                constructed_molecule, nx_positions
                            )

                        if scale == "kamada":
                            stko_graph = stko.Network.init_from_molecule(
                                constructed_molecule
                            )
                            nx_positions = nx.kamada_kawai_layout(
                                stko_graph.get_graph(), dim=3
                            )
                            constructed_molecule = convert_coordinates(
                                constructed_molecule, nx_positions
                            )

                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )

                        potential_names = [
                            f"scan_{textstr}_{cname}_{i}_{num}"
                            for num in range(10)
                        ]

                        conformer = cgx.scram.optimise_cage(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                            potential_names=potential_names,
                        )
                        energy_per_bb = cgx.utilities.get_energy_per_bb(
                            energy_decomposition=(
                                conformer.energy_decomposition
                            ),
                            number_building_blocks=num_bbs,
                        )
                        generated_conformers.append(
                            (
                                name,
                                conformer.molecule.with_centroid((0, 0, 0)),
                                energy_per_bb,
                            )
                        )

                    min_energy_conformer = sorted(
                        generated_conformers, key=lambda p: p[2]
                    )[0]
                    min_energy_name, min_energy_structure, _ = (
                        min_energy_conformer
                    )
                    min_energy_structure.write(
                        str(structure_dir / f"{min_energy_name}_optc.mol")
                    )

                name = f"scan_{textstr}_{cname}_{i}"
                logging.info("building %s", name)
                if tstr == "3P6":
                    constructed_molecule = stk.ConstructedMolecule(
                        stk.cage.M3L6(
                            building_blocks={
                                tetra_bb: (0, 1, 2),
                                converging_bb: (3, 5, 7),
                                diverging_bb: (4, 6, 8),
                            },
                            vertex_positions=None,
                            scale_multiplier=1.0,
                        )
                    )
                    num_bbs = 9

                elif tstr == "4P8":
                    constructed_molecule = stk.ConstructedMolecule(
                        cgx.topologies.CGM4L8(
                            building_blocks={
                                tetra_bb: (0, 1, 2, 3),
                                converging_bb: (4, 6, 8, 10),
                                diverging_bb: (5, 7, 9, 11),
                            },
                            vertex_positions=None,
                            scale_multiplier=1.0,
                        )
                    )
                    num_bbs = 12

                else:
                    raise NotImplementedError

                constructed_molecule.write(
                    str(structure_dir / f"{name}_unopt.mol")
                )

                si = name.split("_")[3]
                potential_names = []
                for cstr in combos:
                    for diff in range(20):
                        potential_names.append(
                            f"scan_{textstr}_{cstr}_{int(si) - diff}",
                        )
                        potential_names.append(
                            f"scan_{textstr}_{cstr}_{int(si) + diff}",
                        )
                        potential_names.append(
                            f"scan_{textstr}_{cstr}_0_{int(si) - diff}",
                        )
                        potential_names.append(
                            f"scan_{textstr}_{cstr}_0_{int(si) + diff}",
                        )

                try:
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
                        conformer.molecule.with_centroid(
                            np.array((0, 0, 0))
                        ).write(str(structure_dir / f"{name}_optc.mol"))

                    analyse_cage(
                        database_path=database_path,
                        name=name,
                        forcefield=forcefield,
                        num_building_blocks=num_bbs,
                    )

                except OpenMMException:
                    pass

            # Rescan over the surface for improved energies.
            for i, xp in enumerate(pair_range_dict["xr"]):
                ligand_measures[pair_range_dict["xl"]][cname] = xp

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    large=converging,
                    small=diverging,
                    large_meas=ligand_measures[converging_name],
                    small_meas=ligand_measures[diverging_name],
                    vdw_bond_cutoff=2,
                )

                name = f"scan_{textstr}_{cname}_{i}"
                logging.info("rescanning %s", name)

                current_cage = stk.BuildingBlock.init_from_file(
                    structure_dir / f"{name}_optc.mol"
                )

                potential_names = []

                x_indices_of_interest = [
                    pair_range_dict["xr"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - xp) for i in pair_range_dict["xr"]],
                            pair_range_dict["xr"],
                            strict=False,
                        )
                    )
                ][:20]

                for cstr, xidx in it.product(combos, x_indices_of_interest):
                    potential_names.append(f"scan_{textstr}_{cstr}_{xidx}")

                conformer = cgx.scram.optimise_from_files(
                    molecule=current_cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                    potential_names=potential_names,
                )

                conformer.molecule.with_centroid(np.array((0, 0, 0))).write(
                    str(structure_dir / f"{name}_optc.mol")
                )

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    num_building_blocks=12,
                )

    make_geom_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_2.png",
    )
    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_1.png",
    )
    ss_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_3.png",
    )


if __name__ == "__main__":
    main()
