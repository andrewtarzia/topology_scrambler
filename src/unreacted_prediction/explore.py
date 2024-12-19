"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk
import stko
from explore_utilities import (
    abead_c,
    binder_bead,
    capper_bead,
    cbead_c,
    eb_str,
    ebead_c,
    isomer_energy,
    pore_str,
    precursors_to_forcefield,
    tetra_bead,
)
from openmm import OpenMMException, openmm
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}
    cmap = {
        "ltp110": "tab:blue",
        "ltp130": "tab:blue",
        "ltp150": "tab:blue",
        "sltp110": "tab:red",
        "sltp130": "tab:red",
        "sltp150": "tab:red",
        "fltp130": "tab:pink",
        "cltp": "tab:orange",
        "cltpunr": "tab:green",
    }

    knowns = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        study = entry.properties["ligname"]

        energy = entry.properties["energy_per_bb"]
        min_distance = entry.properties["min_distance"]

        if study not in energies:
            energies[study] = []
        if study not in knowns:
            knowns[study] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[study].append((round(energy, 4), entry.key, min_distance))
        if int(entry.properties["topology_idx"]) in (15, 12, 29):
            knowns[study].append((round(energy, 4), entry.key, min_distance))

    for sidx, study in enumerate(energies):
        if len(energies[study]) == 0:
            continue

        sorted_energies = sorted(energies[study], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        offset = 20 * (sidx + 1)
        bbox = {"boxstyle": "round", "fc": "1.0"}
        arrowprops = {
            "arrowstyle": "->",
            "connectionstyle": "angle,angleA=0,angleB=90,rad=10",
        }
        ax.annotate(
            text=f"E: {round(min_energy[0],3)} @ {min_energy[1]}",
            xy=(min_energy[2], min_energy[0]),
            xycoords="data",
            xytext=(0.5 * offset, -offset),
            textcoords="offset points",
            bbox=bbox,
            arrowprops=arrowprops,
            color=cmap[study],
            fontsize=8,
        )

        ax.scatter(
            [i[2] for i in energies[study]],
            [i[0] for i in energies[study]],
            marker="o",
            c=cmap[study],
            s=20,
            ec="none",
            alpha=0.3,
            label=f"{study}",
        )
        ax.scatter(
            min_energy[2],
            min_energy[0],
            marker="o",
            c=cmap[study],
            s=20,
            ec="k",
            zorder=2,
        )

    for study in knowns:
        if len(knowns[study]) == 0:
            continue
        sorted_energies = sorted(knowns[study], key=lambda p: p[0])
        min_energy = sorted_energies[0]
        ax.scatter(
            [i[2] for i in knowns[study]],
            [i[0] for i in knowns[study]],
            marker="D",
            c=cmap[study],
            s=20,
            ec="k",
            zorder=-1,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(pore_str(), fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.axhline(y=isomer_energy(), c="k", ls="--")
    ax.set_ylim(None, 1000)
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


def analyse_cage(  # noqa: C901
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: cgx.scram.TopologyIterator,
    topology_code: cgx.scram.TopologyCode,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties

    if "min_distance" not in properties:
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

        ligname, topology_idx, mash_idx = name.split("_")
        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy
                / iterator.get_num_building_blocks(),
                "num_components": num_components,
                "ligname": ligname,
                "topology_idx": topology_idx,
                "mash_idx": mash_idx,
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
                "min_distance": (
                    cgx.analysis.GeomMeasure().calculate_min_distance(
                        database.get_molecule(key=name)
                    )["min_distance"]
                ),
            },
        )


class Single(cgx.molecular.Precursor):
    """A single bead Precursor."""

    def __init__(self, bead: cgx.molecular.CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"S1{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}

        self._building_block = stk.BuildingBlock(
            smiles=f"[{bead.element_string}]",
            functional_groups=(
                stk.SingleAtom(
                    stk.Atom(
                        0,
                        charge=0,
                        atomic_number=cgx.molecular.periodic_table()[
                            bead.element_string
                        ],
                    )
                ),
            ),
            position_matrix=[[0, 0, 0]],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/unreacted/")
    figure_dir = wd / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    calculation_dir = wd / "explore_calculations"
    calculation_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = wd / "explore_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir = wd / "explore_ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    data_dir = wd / "explore_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "explore.db"

    ligand_measures = {
        # From prep.
        # With flexibile backbone.
        "fltp130": {
            "dd": 5.1,
            "de": 5.0,
            "dde": 130,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180,
            "edde_k": 0,
        },
        # With and without chiral torsion on a new dde definition.
        "cltp": {
            "dd": 7.4,
            "de": 2.9,
            "dde": 125,
            "eg": 1.4,
            "gb": 1.4,
            "edde_v": 180 - 75,
            "edde_k": 50,
        },
        "cltpunr": {"dd": 7.4, "de": 2.9, "dde": 125, "eg": 1.4, "gb": 1.4},
        # Rigid, with measured values, varying angle.
        "ltp110": {"dd": 5.1, "de": 5.0, "dde": 110, "eg": 1.4, "gb": 1.4},
        "ltp130": {"dd": 5.1, "de": 5.0, "dde": 130, "eg": 1.4, "gb": 1.4},
        "ltp150": {"dd": 5.1, "de": 5.0, "dde": 150, "eg": 1.4, "gb": 1.4},
        # Small, rigid, varying angle.
        "sltp110": {"dd": 3.0, "de": 3.0, "dde": 110, "eg": 1.4, "gb": 1.4},
        "sltp130": {"dd": 3.0, "de": 3.0, "dde": 130, "eg": 1.4, "gb": 1.4},
        "sltp150": {"dd": 3.0, "de": 3.0, "dde": 150, "eg": 1.4, "gb": 1.4},
    }

    if args.run:
        for ligname, ditopic_meas in ligand_measures.items():
            # Currently, only testing the unreacted case.
            stoichiometry_l_m_c = (6, 4, 4)
            multiplier = 1

            ditopic = cgx.molecular.SixBead(
                bead=cbead_c, abead1=abead_c, abead2=ebead_c
            )
            capper = Single(bead=capper_bead)
            tetra = cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            )

            forcefield = precursors_to_forcefield(
                pair="explore",
                ditopic=ditopic,
                ditopic_meas=ditopic_meas,
            )

            capper_name = f"{capper.get_name()}_f{forcefield.get_identifier()}"
            capper_bb = capper.get_building_block()
            capper_bb.write(str(ligand_dir / f"{capper_name}_optl.mol"))

            ditopic_name = (
                f"{ditopic.get_name()}_f{forcefield.get_identifier()}"
            )
            ditopic_bb = cgx.utilities.optimise_ligand(
                molecule=ditopic.get_building_block(),
                name=ditopic_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            ditopic_bb.write(str(ligand_dir / f"{ditopic_name}_optl.mol"))
            ditopic_bb = ditopic_bb.clone()

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

            # Define a connectivity based on a multiplier.
            iterator = cgx.scram.IHomolepticTopologyIterator(
                building_block_counts={
                    tetra_bb: stoichiometry_l_m_c[1] * multiplier,
                    ditopic_bb: stoichiometry_l_m_c[0] * multiplier,
                    capper_bb: stoichiometry_l_m_c[2] * multiplier,
                },
                graph_type="4-4FG_6-2FG_4-1FG",
                graph_set="rx",
                max_samples=int(1e5),
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )

            for idx, topology_code in enumerate(iterator.yield_graphs()):
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
                    constructed_molecule = cgx.scram.try_except_construction(
                        iterator=iterator,
                        topology_code=topology_code,
                        building_block_configuration=None,
                        vertex_positions=vertex_positions,
                    )
                    name = f"{ligname}_{idx}_{mash_idx}"

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
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=topology_code,
                        )

                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="exp_1.png",
    )


if __name__ == "__main__":
    main()
