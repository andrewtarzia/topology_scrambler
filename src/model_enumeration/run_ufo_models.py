"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from ufo_utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)
from utilities import eb_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901, PLR0913
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: cgx.scram.TopologyIterator,
    topology_code: cgx.scram.TopologyCode,
    bb_config: cgx.scram.BuildingBlockConfiguration,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "num_components" not in properties:
        energy_decomp = {}
        for component in properties["energy_decomposition"]:
            component_tup = properties["energy_decomposition"][component]
            if component == "total energy":
                energy_decomp[f"{component}_{component_tup[1]}"] = float(
                    component_tup[0]
                )
            else:
                just_name = component.split("'")[1]
                key = f"{just_name}_{component_tup[1]}"
                value = float(component_tup[0])
                if key in energy_decomp:
                    energy_decomp[key] += value
                else:
                    energy_decomp[key] = value
        fin_energy = energy_decomp["total energy_kJ/mol"]
        if (
            sum(
                energy_decomp[i]
                for i in energy_decomp
                if "total energy" not in i
            )
            != fin_energy
        ):
            msg = (
                "energy decompisition does not sum to total energy for"
                f" {name}: {energy_decomp}"
            )
            raise RuntimeError(msg)

        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        k_dict = {}
        v_dict = {}

        for bt in ff_targets["bonds"]:
            cp = (bt.type1, bt.type2)
            k_dict["_".join(cp)] = bt.bond_k.value_in_unit(
                openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(
                openmm.unit.angstrom
            )

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(
                    openmm.unit.degrees
                )
            except TypeError:
                # Handle different angle types.
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
                v_dict["_".join(cp)] = (at.n, at.b)

        for at in ff_targets["torsions"]:
            cp = at.search_string
            k_dict["_".join(cp)] = at.torsion_k.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            v_dict["_".join(cp)] = at.phi0.value_in_unit(openmm.unit.degrees)
        for at in ff_targets["nonbondeds"]:
            v_dict[at.bead_class] = at.sigma.value_in_unit(
                openmm.unit.angstrom
            )
            k_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "k_dict": k_dict,
            "v_dict": v_dict,
        }

        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        (l1, l2, multiplier, topology_idx, mash_idx, bbconfig_name) = (
            name.split("_")
        )

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy
                / iterator.get_num_building_blocks(),
                "l1": l1,
                "l2": l2,
                "num_components": num_components,
                "multiplier": multiplier,
                "topology_idx": topology_idx,
                "mash_idx": mash_idx,
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
                "bb_config_idx": bb_config.idx,
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
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = {}
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"
        if pair != target_pair:
            continue

        energy = entry.properties["energy_per_bb"]

        if multi not in energies:
            energies[multi] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[multi].append((round(energy, 4), entry.key))

    print(energies)
    with (figure_dir / f"min_{pair}.txt").open("w") as f:
        for multi in energies:
            if len(energies[multi]) == 0:
                continue

            sorted_energies = sorted(energies[multi], key=lambda p: p[0])
            min_energy = sorted_energies[0]

            ax.plot(
                [i[0] for i in energies[multi]],
                marker="o",
                c=cmap[multi],
                markersize=4,
                label=f"{multi}: {round(min_energy[0],3)} @ {min_energy[1]}",
            )

            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.axhline(y=0.3, c="k", ls="--")
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

    raise SystemExit


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}

    xs = ["1", "2", "3", "4"]
    ys = ["lf_ls1"]

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), tidx, bidx))

    vmin = 0
    vmax = 1
    for pair, multi in energies:
        sorted_energies = sorted(energies[(pair, multi)], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = ys.index(pair)

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
            fontsize=16,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels(ys)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"4:2:3 {eb_str()}", fontsize=16)

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
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    systems = {
        ("lf_ls1", "1"): {"name": "lf-ls1-1", "data": []},
        ("lf_ls1", "2"): {"name": "lf-ls1-2", "data": []},
        ("lf_ls1", "3"): {"name": "lf-ls1-4", "data": []},
        ("lf_ls1", "4"): {"name": "lf-ls1-1", "data": []},
    }

    count = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        l1 = entry.properties["l1"]
        l2 = entry.properties["l2"]
        pair = f"{l1}_{l2}"

        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in systems:
            continue
        energy = entry.properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        systems[(pair, multi)]["data"].append(energy)
        count += 1

    logging.info("structures built, %s", count)
    rng = np.random.default_rng(seed=2)

    for i, (pair, multi) in enumerate(systems):
        if len(systems[(pair, multi)]["data"]) == 0:
            continue
        min_energy = min(systems[(pair, multi)]["data"])

        ax.scatter(
            [
                i + (2 * rng.random() - 1) * 0.3
                for j in range(len(systems[(pair, multi)]["data"]))
            ],
            systems[(pair, multi)]["data"],
            c="tab:blue",
            alpha=0.1,
            edgecolor="none",
            s=30,
            marker="o",
            zorder=1,
        )
        ax.scatter(
            i,
            min_energy,
            c="tab:orange",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
            zorder=2,
        )

    ax.axvline(x=3 + 0.5, c="gray")
    ax.axvline(x=6 + 0.5, c="gray")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.axhline(y=0.3, c="k", ls="--")

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


def main() -> None:  # noqa: C901, PLR0915
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "ufo_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "ufo_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "ufo_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "ufo_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "ufo.db"

    ligand_measures = {
        "lf": {"dd": 8.0, "de": 4.3, "dde": 133, "eg": 1.4, "gb": 1.4},
        "ls1": {"ba": 2.8, "aa": 4.9, "bac": 150, "bacab": 180},
        "ls9": {"ba": 2.8, "aa": 5.5, "bac": 165, "bacab": 180},
    }
    print("add new systems - steric mimics on ls1 (ls3-ls8), ls10")

    pairs = {
        "lf_ls1": {
            "converging_name": "lf",
            "diverging_name": "ls1",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        },
        # "lf_ls9": {
        #     "converging_name": "lf",
        #     "diverging_name": "ls9",
        #     "stoichiometry_L_L_M": (1, 1, 1),
        #     "converging": SixBead(
        #         bead=cbead_c,
        #         abead1=abead_c,
        #         abead2=ebead_c,
        #     ),
        #     "diverging": cgx.molecular.TwoC1Arm(
        #         bead=cbead_d,
        #         abead1=abead_d,
        #     ),
        #     "tetra": cgx.molecular.FourC1Arm(
        #         bead=tetra_bead,
        #         abead1=binder_bead,
        #     ),
        #     "multipliers": (1, 2, 3, 4),
        # },
    }

    if args.run:
        for pair in pairs:
            converging_name = pairs[pair]["converging_name"]
            diverging_name = pairs[pair]["diverging_name"]
            converging = pairs[pair]["converging"]
            diverging = pairs[pair]["diverging"]
            tetra = pairs[pair]["tetra"]

            forcefield = precursors_to_forcefield(
                pair=pair,
                diverging=diverging,
                converging=converging,
                conv_meas=ligand_measures[converging_name],
                dive_meas=ligand_measures[diverging_name],
            )

            converging_name = (
                f"{converging.get_name()}_f{forcefield.get_identifier()}"
            )
            converging_bb = cgx.utilities.optimise_ligand(
                molecule=converging.get_building_block(),
                name=converging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            converging_bb.write(
                str(ligand_dir / f"{converging_name}_optl.mol")
            )
            converging_bb = converging_bb.clone()

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

            diverging_name = (
                f"{diverging.get_name()}_f{forcefield.get_identifier()}"
            )
            diverging_bb = cgx.utilities.optimise_ligand(
                molecule=diverging.get_building_block(),
                name=diverging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            diverging_bb.write(str(ligand_dir / f"{diverging_name}_optl.mol"))
            diverging_bb = diverging_bb.clone()

            for multiplier in pairs[pair]["multipliers"]:
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.IHomolepticTopologyIterator(
                    building_block_counts={
                        tetra_bb: pairs[pair]["stoichiometry_L_L_M"][2]
                        * multiplier,
                        diverging_bb: pairs[pair]["stoichiometry_L_L_M"][0]
                        * multiplier,
                        converging_bb: pairs[pair]["stoichiometry_L_L_M"][1]
                        * multiplier,
                    },
                    graph_type=f"{1*multiplier}P{2*multiplier}",
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

                logging.info(
                    "producing: %s structures",
                    len(possible_bbdicts) * iterator.count_graphs(),
                )

                for (idx, topology_code), bb_config in it.product(
                    enumerate(iterator.yield_graphs()), possible_bbdicts
                ):
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
                                building_block_configuration=bb_config,
                                vertex_positions=vertex_positions,
                            )
                        )
                        name = (
                            f"{pair}_{multiplier}_{idx}_{mash_idx}"
                            f"_b{bb_config.idx}"
                        )

                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )

                        # Optimise and save.
                        logging.info("building %s", name)

                        try:
                            conformer = opt_function(
                                molecule=constructed_molecule,
                                name=name,
                                output_dir=calculation_dir,
                                forcefield=forcefield,
                                platform=None,
                                database_path=database_path,
                            )
                            if conformer is not None:
                                conformer.molecule.with_centroid(
                                    (0, 0, 0)
                                ).write(
                                    str(structure_dir / f"{name}_optc.mol")
                                )

                            analyse_cage(
                                database_path=database_path,
                                name=name,
                                forcefield=forcefield,
                                iterator=iterator,
                                topology_code=topology_code,
                                bb_config=bb_config,
                            )

                        except OpenMMException:
                            logging.info("failed optimisation of %s", name)

        make_plot(
            database_path=database_path,
            target_pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"ufo_1_{pair}.png",
        )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufo_3.png",
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufo_4.png",
    )

    for pair in pairs:
        make_plot(
            database_path=database_path,
            target_pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"ufo_1_{pair}.png",
        )


if __name__ == "__main__":
    main()
