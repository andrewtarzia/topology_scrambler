"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import warnings
from collections import abc, defaultdict

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import openmm
import polars as pl
import stk
from rdkit import RDLogger

from model_enumeration.utilities import (
    arm_bead,
    binder_bead,
    cage_topology_options,
    convert_topo,
    core_bead,
    dihedral_state_threshold,
    eb_str,
    isomer_energy,
    tetragonal_bead,
    trigonal_bead,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")

definer_dict_2p4 = {
    # Bonds.
    "mb": ("bond", 1.5, 1e5),
    "ab": ("bond", 1.0, 1e5),
    "ac": ("bond", 1.5, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "bac": ("angle", range(90, 181, 2), 1e2),
    "aca": ("angle", 180, 1e2),
    # Torsions.
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
}

definer_dict_2p3 = {
    # Bonds.
    "nb": ("bond", 1.5, 1e5),
    "ab": ("bond", 1.0, 1e5),
    "ac": ("bond", 1.5, 1e5),
    # Angles.
    "bnb": ("angle", 120, 1e2),
    "nba": ("angle", 180, 1e2),
    "bac": ("angle", range(90, 181, 2), 1e2),
    "aca": ("angle", 180, 1e2),
    # Torsions.
    # Nonbondeds.
    "n": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_calcs",
        help="whether to run construction or not",
        action="store_true",
    )

    return parser.parse_args()


def structure_function(  # noqa: PLR0912, PLR0915, C901
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,  # noqa: ARG001
) -> None:
    """Generate a structure from a chromosome."""
    database = cgx.utilities.AtomliteDatabase(database_path)

    forcefield = chromosome.get_forcefield()
    larger, smaller = chromosome.get_precursors()
    tstr, tfunction = chromosome.get_topology_information()

    name = (
        f"{chromosome.prefix}_{larger.get_name()}_{smaller.get_name()}_"
        f"f{chromosome.get_separated_string()}"
    )
    cage = stk.ConstructedMolecule(
        tfunction((larger.get_building_block(), smaller.get_building_block())),
    )
    fina_mol_file = calculation_output / f"{name}_final.mol"
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_conformer = cgx.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    elif fina_mol_file.exists():
        ensemble = cgx.molecular.Ensemble(
            base_molecule=cage,
            base_mol_path=calculation_output / f"{name}_base.mol",
            conformer_xyz=calculation_output / f"{name}_ensemble.xyz",
            data_json=calculation_output / f"{name}_ensemble.json",
            overwrite=False,
        )
        final_conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=final_conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": final_conformer.energy_decomposition,
                "source": final_conformer.source,
                "optimised": True,
                "viable": True,
            },
        )

    else:
        logging.info("building %s", name)
        # Assign the forcefield.
        assigned_system = forcefield.assign_terms(
            molecule=cage,
            name=name,
            output_dir=calculation_output,
        )

        ensemble = cgx.molecular.Ensemble(
            base_molecule=cage,
            base_mol_path=calculation_output / f"{name}_base.mol",
            conformer_xyz=calculation_output / f"{name}_ensemble.xyz",
            data_json=calculation_output / f"{name}_ensemble.json",
            overwrite=True,
        )
        temp_molecule = cgx.utilities.run_constrained_optimisation(
            assigned_system=assigned_system,
            name=name,
            output_dir=calculation_output,
            bond_ff_scale=10,
            angle_ff_scale=10,
            max_iterations=100,
            platform=None,
        )

        conformer = cgx.utilities.run_optimisation(
            assigned_system=cgx.forcefields.AssignedSystem(
                molecule=temp_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="opt1",
            output_dir=calculation_output,
            platform=None,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")

        # Run optimisations of series of conformers with shifted out
        # building blocks.
        for test_molecule in cgx.utilities.yield_shifted_models(
            temp_molecule, forcefield, kicks=(1, 2, 3, 4)
        ):
            conformer = cgx.utilities.run_optimisation(
                assigned_system=cgx.forcefields.AssignedSystem(
                    molecule=test_molecule,
                    forcefield_terms=assigned_system.forcefield_terms,
                    system_xml=assigned_system.system_xml,
                    topology_xml=assigned_system.topology_xml,
                    bead_set=assigned_system.bead_set,
                    vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
                ),
                name=name,
                file_suffix="sopt",
                output_dir=calculation_output,
                platform=None,
            )
            ensemble.add_conformer(conformer=conformer, source="shifted")

        num_steps = 20000
        traj_freq = 500
        soft_md_trajectory = cgx.utilities.run_soft_md_cycle(
            name=name,
            assigned_system=cgx.forcefields.AssignedSystem(
                molecule=ensemble.get_lowest_e_conformer().molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            output_dir=calculation_output,
            suffix="smd",
            bond_ff_scale=10,
            angle_ff_scale=10,
            temperature=300 * openmm.unit.kelvin,
            num_steps=num_steps,
            time_step=0.5 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=traj_freq,
            traj_freq=traj_freq,
            platform=None,
        )
        if soft_md_trajectory is None:
            logging.info("!!!!! MD exploded for %s !!!!!", name)
            msg = "OpenMM Exception"
            raise ValueError(msg)

        soft_md_data = soft_md_trajectory.get_data()

        # Check that the trajectory is as long as it should be.
        if len(soft_md_data) != num_steps / traj_freq:
            logging.info("!!!!! MD failed for %s !!!!!", name)
            raise ValueError

        # Go through each conformer from soft MD.
        # Optimise them all.
        for md_conformer in soft_md_trajectory.yield_conformers():
            conformer = cgx.utilities.run_optimisation(
                assigned_system=cgx.forcefields.AssignedSystem(
                    molecule=md_conformer.molecule,
                    forcefield_terms=assigned_system.forcefield_terms,
                    system_xml=assigned_system.system_xml,
                    topology_xml=assigned_system.topology_xml,
                    bead_set=assigned_system.bead_set,
                    vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
                ),
                name=name,
                file_suffix="smd_mdc",
                output_dir=calculation_output,
                platform=None,
            )
            ensemble.add_conformer(conformer=conformer, source="smd")
        ensemble.write_conformers_to_file()

        final_conformer = ensemble.get_lowest_e_conformer()
        final_conformerid = final_conformer.conformer_id
        min_energy = final_conformer.energy_decomposition["total energy"][0]
        logging.info(
            "Min. energy conformer: %s from %s with energy: %s kJ.mol-1",
            final_conformerid,
            final_conformer.source,
            round(min_energy, 3),
        )

        # Add to atomlite database.
        database.add_molecule(molecule=final_conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": final_conformer.energy_decomposition,
                "source": final_conformer.source,
                "optimised": True,
                "viable": True,
            },
        )
        final_conformer.molecule.write(fina_mol_file)
        final_conformer.molecule.write(structure_output / f"{name}_optc.mol")

    properties = database.get_entry(key=name).properties
    if "energy_per_bb" not in properties:
        res_dict = {
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=properties["energy_decomposition"],
                number_building_blocks=cgx.topologies.stoich_map(tstr),
            ),
        }
        database.add_properties(key=name, property_dict=res_dict)

    properties = database.get_entry(key=name).properties
    if "max_uniformity" not in properties:
        host_uniformity_distances = defaultdict(list)
        host_centroid = final_conformer.molecule.get_centroid()

        for atom in final_conformer.molecule.get_atoms():
            atom_e_string = atom.__class__.__name__
            atom_position = next(
                final_conformer.molecule.get_atomic_positions(atom.get_id())
            )
            distance_from_centroid = np.linalg.norm(
                atom_position - host_centroid
            )
            host_uniformity_distances[atom_e_string].append(
                distance_from_centroid
            )

        host_uniformity = {
            i: np.std(host_uniformity_distances[i])
            for i in host_uniformity_distances
        }
        host_uniformity_means = {
            i: np.mean(host_uniformity_distances[i])
            for i in host_uniformity_distances
        }

        database.add_properties(
            key=name,
            property_dict={
                "uniformity_distances": host_uniformity_distances,
                "uniformity_means": host_uniformity_means,
                "uniformity_diffs": max(host_uniformity_means.values())
                - min(host_uniformity_means.values()),
                "uniformity": host_uniformity,
                "max_uniformity": max(host_uniformity.values()),
            },
        )

    properties = database.get_entry(key=name).properties
    if "dihedral_states" not in properties:
        g_measure = cgx.analysis.GeomMeasure(
            target_torsions=(
                # Add a custom torsion.
                cgx.terms.TargetTorsion(
                    search_string=("b", "a", "c", "a", "b"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    torsion_n=1,
                ),
            )
        )
        bond_data = g_measure.calculate_bonds(final_conformer.molecule)
        bond_data = {"_".join(i): bond_data[i] for i in bond_data}
        angle_data = g_measure.calculate_angles(final_conformer.molecule)
        angle_data = {"_".join(i): angle_data[i] for i in angle_data}
        dihedral_data = g_measure.calculate_torsions(
            molecule=final_conformer.molecule,
            absolute=False,
            as_search_string=True,
        )
        opt_pore_data = g_measure.calculate_min_distance(
            final_conformer.molecule
        )["min_distance"]

        dihedral_spread = np.std(dihedral_data["Pb_Ba_Ag_Ba_Pb"])
        all_values = dihedral_data["Pb_Ba_Ag_Ba_Pb"]
        # Then I want the distinct env as a coloured point.
        envs = {}

        # Set envs here, should contain all.
        for d in all_values:
            if len(envs) == 0:
                envs[round(d, 0)] = 1
            else:
                within_distance_from_env = [
                    (de, d)
                    for de in envs
                    if abs(d - de) < dihedral_state_threshold()
                ]

                if len(within_distance_from_env) == 0:
                    envs[round(d, 0)] = 1
                elif len(within_distance_from_env) == 1:
                    envs[within_distance_from_env[0][0]] += 1
                else:
                    distances = [
                        abs(i[0] - i[1]) for i in within_distance_from_env
                    ]
                    min_de = within_distance_from_env[
                        distances.index(min(distances))
                    ]
                    envs[min_de[0]] += 1

        envs = [(float(i), float(envs[i])) for i in envs]

        dihedral_num_states = len(envs)

        database.add_properties(
            key=name,
            property_dict={
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
                "opt_pore_data": opt_pore_data,
                "dihedral_spread": dihedral_spread,
                "dihedral_states": envs,
                "dihedral_num_states": dihedral_num_states,
            },
        )

    properties = database.get_entry(key=name).properties
    if "forcefield_dict" not in properties:
        forcefield_dict = forcefield.get_forcefield_dictionary()
        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "prefix": chromosome.prefix,
                "chromosome": tuple(int(i) for i in chromosome.name),
                "tstr": tstr,
            },
        )


def bar_function(  # noqa: C901
    data_output: pathlib.Path,
    figure_output: pathlib.Path,
    study: str,
    topology_options: abc.Sequence[tuple[str, abc.Callable]],
) -> None:
    """Plot the bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for x, (tstr, _) in enumerate(topology_options):
        prefix = f"{study}_{tstr}"
        database_path = data_output / f"{prefix}.db"
        database = cgx.utilities.AtomliteDatabase(database_path)

        target_x = "$.forcefield_dict.v_dict.b_a_c"
        if tstr in (
            "2P4",
            "3P6",
            "4P8",
            "4P82",
            "6P12",
            "6P122",
            "8P162",
            "8P16",
        ):
            target_y = "$.forcefield_dict.v_dict.b_m_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_m_b",
                "$.max_uniformity",
                "$.dihedral_num_states",
                "$.dihedral_states",
            ]
        elif tstr in ("2P3", "4P6", "4P62", "6P9", "8P12"):
            target_y = "$.forcefield_dict.v_dict.b_n_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_n_b",
                "$.max_uniformity",
                "$.dihedral_num_states",
                "$.dihedral_states",
            ]

        dataframe = database.get_property_df(
            properties=df_properties,
            allow_missing=False,
        )
        if len(dataframe) == 0:
            continue

        target_props = {
            (0.25, 0.25, 0.5): "tab:orange",
            (0.25, 0.25, 0.25, 0.25): "tab:green",
            (0.5, 0.5): "tab:purple",
            (1.0,): "tab:cyan",
        }
        counts = {"white": 0, "tab:blue": 0}
        for i in target_props.values():
            counts[i] = 0

        for xangle, yangle in it.product(
            set(dataframe[target_x]),
            set(dataframe[target_y]),
        ):
            pdata = dataframe.filter(pl.col(target_x) == xangle)
            pdata = pdata.filter(pl.col(target_y) == yangle)

            if len(pdata) != 1:
                continue

            states = pdata["$.dihedral_states"].item(0).to_list()

            if pdata["$.energy_per_bb"].item(0) <= isomer_energy():
                count = sum([i[1] for i in states])
                curren_prop = tuple(sorted([i[1] / count for i in states]))
                try:
                    colour = target_props[curren_prop]
                except KeyError:
                    colour = "tab:blue"

            else:
                colour = "white"

            counts[colour] += 1

        counts = {i: counts[i] / sum(counts.values()) for i in counts}

        bottom = 0
        for state in counts:
            p = ax.bar(
                x,
                counts[state],
                0.8,
                bottom=bottom,
                color=state,
                edgecolor="k",
                linewidth=2,
            )

            ax.bar_label(
                p,
                label_type="center",
                padding=0.1,
                color="k",
                fontsize=16,
                fmt="%.2f",
            )

            bottom += counts[state]

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("proportion", fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(topology_options)))
    ax.set_xticklabels(
        [convert_topo(i[0]) for i in topology_options],
        fontsize=16,
        rotation=90,
    )

    fig.tight_layout()
    fig.savefig(
        figure_output / f"envi_{study}_bar.png",
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_output / f"envi_{study}_bar.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_torsion_data(
    data_output: pathlib.Path,
    figure_output: pathlib.Path,
    study: str,
    topology_options: abc.Sequence[tuple[str, abc.Callable]],
) -> None:
    """Plot the torsions."""
    fig, axs = plt.subplots(
        nrows=len(topology_options),
        sharex=True,
        sharey=True,
        figsize=(8, 10),
    )
    for ax, (tstr, _) in zip(axs, topology_options, strict=True):
        prefix = f"{study}_{tstr}"
        database_path = data_output / f"{prefix}.db"
        database = cgx.utilities.AtomliteDatabase(database_path)

        ax2 = ax.twinx()
        ax2_color = "tab:red"

        target_x = "$.forcefield_dict.v_dict.b_a_c"
        if tstr in (
            "2P4",
            "3P6",
            "4P8",
            "4P82",
            "6P12",
            "6P122",
            "8P162",
            "8P16",
        ):
            target_y = "$.forcefield_dict.v_dict.b_m_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_m_b",
                "$.dihedral_num_states",
                "$.dihedral_states",
                "$.dihedral_spread",
            ]
        elif tstr in ("2P3", "4P6", "4P62", "6P9", "8P12"):
            target_y = "$.forcefield_dict.v_dict.b_n_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_n_b",
                "$.dihedral_num_states",
                "$.dihedral_states",
                "$.dihedral_spread",
            ]

        dataframe = database.get_property_df(
            properties=df_properties,
            allow_missing=False,
        )
        if len(dataframe) == 0:
            continue

        xys = []
        for xangle, yangle in it.product(
            set(dataframe[target_x]),
            set(dataframe[target_y]),
        ):
            pdata = dataframe.filter(pl.col(target_x) == xangle)
            pdata = pdata.filter(pl.col(target_y) == yangle)

            if len(pdata) != 1:
                continue

            states = pdata["$.dihedral_states"].item(0).to_list()

            target_props = {
                (0.25, 0.25, 0.5): "tab:orange",
                (0.25, 0.25, 0.25, 0.25): "tab:green",
                (0.5, 0.5): "tab:purple",
                (1.0,): "tab:cyan",
            }

            if pdata["$.energy_per_bb"].item(0) <= isomer_energy():
                count = sum([i[1] for i in states])
                curren_prop = tuple(sorted([i[1] / count for i in states]))
                try:
                    c = target_props[curren_prop]
                except KeyError:
                    c = "tab:blue"

            else:
                c = "white"
            xys.append((xangle, pdata["$.energy_per_bb"].item(0)))
            ax.scatter(
                [xangle for i in states],
                [i[0] for i in states],
                c=c,
                alpha=1.0,
                edgecolor="k",
                s=40,
                zorder=2,
            )

        xys = sorted(xys, key=lambda x: x[0])
        ax2.plot(
            [i[0] for i in xys],
            [i[1] for i in xys],
            c=ax2_color,
            zorder=0,
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.text(x=160, y=100, s=tstr, fontsize=16)
        ax.set_ylim(-180, 182)
        ax.set_yticks([-180, 0, 180])
        ax.axhline(y=0, c="k", zorder=-2)

        ax2.set_ylabel(eb_str(), color=ax2_color, fontsize=16)
        ax2.tick_params(axis="y", labelcolor=ax2_color, labelsize=16)
        ax2.set_ylim(0, 1)

    ax.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_output / f"envi_{study}_tm.png",
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_output / f"envi_{study}_tm.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_specific_torsion_data(
    data_output: pathlib.Path,
    figure_output: pathlib.Path,
    study: str,
    topology_options: abc.Sequence[tuple[str, abc.Callable]],
) -> None:
    """Plot the torsions."""
    for tstr, _ in topology_options:
        if tstr not in ("4P82", "4P6", "8P12", "8P16"):
            continue
        fig, ax = plt.subplots(figsize=(8, 3))
        prefix = f"{study}_{tstr}"
        database_path = data_output / f"{prefix}.db"
        database = cgx.utilities.AtomliteDatabase(database_path)

        target_x = "$.forcefield_dict.v_dict.b_a_c"
        if tstr in ("4P82", "8P16"):
            target_y = "$.forcefield_dict.v_dict.b_m_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_m_b",
                "$.dihedral_num_states",
                "$.dihedral_states",
                "$.dihedral_spread",
            ]
        elif tstr in ("2P3", "4P6", "4P62", "6P9", "8P12"):
            target_y = "$.forcefield_dict.v_dict.b_n_b"
            df_properties = [
                "$.energy_per_bb",
                "$.forcefield_dict.v_dict.b_a_c",
                "$.forcefield_dict.v_dict.b_n_b",
                "$.dihedral_num_states",
                "$.dihedral_states",
                "$.dihedral_spread",
            ]

        dataframe = database.get_property_df(
            properties=df_properties,
            allow_missing=False,
        )
        if len(dataframe) == 0:
            continue

        for xangle, yangle in it.product(
            set(dataframe[target_x]),
            set(dataframe[target_y]),
        ):
            pdata = dataframe.filter(pl.col(target_x) == xangle)
            pdata = pdata.filter(pl.col(target_y) == yangle)

            if len(pdata) != 1:
                continue

            states = pdata["$.dihedral_states"].item(0).to_list()

            target_props = {
                (0.25, 0.25, 0.5): "tab:orange",
                (0.25, 0.25, 0.25, 0.25): "tab:green",
                (0.5, 0.5): "tab:purple",
                (1.0,): "tab:cyan",
            }

            if pdata["$.energy_per_bb"].item(0) <= isomer_energy():
                count = sum([i[1] for i in states])
                curren_prop = tuple(sorted([i[1] / count for i in states]))
                try:
                    c = target_props[curren_prop]
                except KeyError:
                    c = "tab:blue"
            else:
                c = "white"

            ax.scatter(
                [xangle for i in states],
                [i[0] for i in states],
                c=c,
                alpha=1.0,
                edgecolor="k",
                s=60,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(tstr, fontsize=16)
        ax.set_ylim(-180, 182)
        ax.set_yticks([-180, 0, 180])

        ax.set_ylabel("torsion states [$^\\circ$]", fontsize=16)
        ax.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
        ax.axhline(y=0, alpha=0.2, zorder=-1, c="k")

        fig.tight_layout()
        fig.savefig(
            figure_output / f"envi_{study}_{tstr}_tm.png",
            dpi=360,
            bbox_inches="tight",
        )
        fig.savefig(
            figure_output / f"envi_{study}_{tstr}_tm.pdf",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()


def main() -> None:
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "envi_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "envi_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "envi_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "envi_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    studies = {
        "st1": {
            "topology": "homoleptic_2p4",
            "definer_dict": definer_dict_2p4,
            "present_beads": (
                arm_bead,
                binder_bead,
                tetragonal_bead,
                core_bead,
            ),
            "large_gene": (
                cgx.molecular.FourC1Arm(
                    bead=tetragonal_bead, abead1=binder_bead
                ),
            ),
            "small_gene": (
                cgx.molecular.TwoC1Arm(bead=core_bead, abead1=arm_bead),
            ),
        },
        "st3": {
            "topology": "homoleptic_2p3",
            "definer_dict": definer_dict_2p3,
            "present_beads": (arm_bead, binder_bead, trigonal_bead, core_bead),
            "large_gene": (
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "small_gene": (
                cgx.molecular.TwoC1Arm(bead=core_bead, abead1=arm_bead),
            ),
        },
    }

    for study in studies:
        if args.run_calcs:
            for cage_topology in cage_topology_options(
                study=studies[study]["topology"]
            ):
                prefix = f"{study}_{cage_topology[0]}"
                database_path = data_dir / f"{prefix}.db"
                cgx.utilities.AtomliteDatabase(db_file=database_path)

                chromosome_gen = cgx.systems_optimisation.ChromosomeGenerator(
                    prefix=prefix,
                    present_beads=studies[study]["present_beads"],
                    vdw_bond_cutoff=2,
                )
                chromosome_gen.add_gene(
                    iteration=[cage_topology], gene_type="topology"
                )

                chromosome_gen.add_gene(
                    iteration=studies[study]["large_gene"],
                    gene_type="precursor",
                )
                chromosome_gen.add_gene(
                    iteration=studies[study]["small_gene"],
                    gene_type="precursor",
                )

                chromosome_gen.add_forcefield_dict(
                    definer_dict=studies[study]["definer_dict"]
                )

                count = 0
                for chromosome in chromosome_gen.yield_chromosomes():
                    structure_function(
                        chromosome=chromosome,
                        database_path=database_path,
                        calculation_output=calculation_dir,
                        structure_output=structure_dir,
                        options={},
                    )
                    count += 1

                logging.info(
                    "(%s) built %s for %s", study, count, cage_topology[0]
                )

        plot_torsion_data(
            data_output=data_dir,
            figure_output=figure_dir,
            study=study,
            topology_options=cage_topology_options(
                study=studies[study]["topology"]
            ),
        )
        bar_function(
            data_output=data_dir,
            figure_output=figure_dir,
            study=study,
            topology_options=cage_topology_options(
                study=studies[study]["topology"]
            ),
        )
        plot_specific_torsion_data(
            data_output=data_dir,
            figure_output=figure_dir,
            study=study,
            topology_options=cage_topology_options(
                study=studies[study]["topology"]
            ),
        )


if __name__ == "__main__":
    main()
