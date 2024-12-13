"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import warnings
from collections import defaultdict

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openmm
import polars as pl
import stk
from ds_utilities import (
    EnvVariables,
    arm_bead,
    binder_bead,
    core_bead,
    create_zone,
    get_forcefield_dict,
    stoich_map,
    tetragonal_bead,
    tetragonal_bead2,
    trigonal_bead,
    trigonal_bead2,
)
from rdkit import RDLogger

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")

definer_dict_2p4 = {
    # Bonds.
    "mb": ("bond", 1.5, 1e5),
    "yb": ("bond", 1.5, 1e5),
    "ab": ("bond", 1.0, 1e5),
    "ac": ("bond", 1.5, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "byb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "yba": ("angle", 180, 1e2),
    "bac": ("angle", 90, 1e2),
    "aca": ("angle", 180, 1e2),
    # Torsions.
    "bacab": ("tors", "0134", 180, 50, 1),
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "y": ("nb", 10.0, 1.0),
}

definer_dict_2p3 = {
    # Bonds.
    "nb": ("bond", 1.5, 1e5),
    "xb": ("bond", 1.5, 1e5),
    "ab": ("bond", 1.0, 1e5),
    "ac": ("bond", 1.5, 1e5),
    # Angles.
    "bnb": ("angle", 120, 1e2),
    "bxb": ("angle", 120, 1e2),
    "nba": ("angle", 180, 1e2),
    "xba": ("angle", 180, 1e2),
    "bac": ("angle", 90, 1e2),
    "aca": ("angle", 180, 1e2),
    # Torsions.
    "bacab": ("tors", "0134", 180, 50, 1),
    # Nonbondeds.
    "n": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "x": ("nb", 10.0, 1.0),
}


def template_structure_function(  # noqa: PLR0915
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,  # noqa: ARG001
) -> None:
    """Generate a structure from a chromosome."""
    raise SystemExit("try refactor")
    database = cgx.utilities.AtomliteDatabase(database_path)

    forcefield = chromosome.get_forcefield()
    larger, larger2, smaller, bb_dict = chromosome.get_precursors()
    tstr, tfunction = chromosome.get_topology_information()

    name = (
        f"{chromosome.prefix}_{larger.get_name()}_{larger2.get_name()}_"
        f"{smaller.get_name()}_v{bb_dict[0]}_"
        f"f{chromosome.get_separated_string()}"
    )

    bbs = {
        bb: tuple(bb_dict[1][idx])
        for idx, bb in enumerate(
            (
                larger.get_building_block(),
                larger2.get_building_block(),
                smaller.get_building_block(),
            )
        )
    }

    template_name = (
        f"temp_{chromosome.prefix}_{larger.get_name()}_{larger2.get_name()}_"
        f"{smaller.get_name()}_v{bb_dict[0]}_"
        f"f{chromosome.get_separated_string()}"
    )
    template_file = calculation_output / f"{template_name}_final.mol"

    cage = stk.ConstructedMolecule(tfunction(building_blocks=bbs))

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
        if not template_file.exists():
            final_conformer.molecule.write(template_file)

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
                "bb_dict_idx": bb_dict[0],
                "bb_dict": bb_dict[1],
            },
        )
        if not template_file.exists():
            final_conformer.molecule.write(template_file)

    else:
        ensemble = cgx.molecular.Ensemble(
            base_molecule=cage,
            base_mol_path=calculation_output / f"{name}_base.mol",
            conformer_xyz=calculation_output / f"{name}_ensemble.xyz",
            data_json=calculation_output / f"{name}_ensemble.json",
            overwrite=True,
        )

        logging.info("building %s with template", name)
        # Assign the forcefield.
        assigned_system = forcefield.assign_terms(
            molecule=cage,
            name=name,
            output_dir=calculation_output,
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
        final_conformer.molecule.write(template_file)

        # Add to atomlite database.
        database.add_molecule(molecule=final_conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": final_conformer.energy_decomposition,
                "source": final_conformer.source,
                "optimised": True,
                "viable": True,
                "bb_dict_idx": bb_dict[0],
                "bb_dict": bb_dict[1],
            },
        )
        final_conformer.molecule.write(fina_mol_file)
        final_conformer.molecule.write(structure_output / f"{name}_optc.mol")


def structure_function(  # noqa: PLR0912, PLR0915, C901
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,
) -> None:
    """Generate a structure from a chromosome."""
    raise SystemExit("try refactor")
    database = cgx.utilities.AtomliteDatabase(database_path)

    forcefield = chromosome.get_forcefield()
    larger, larger2, smaller, bb_dict = chromosome.get_precursors()
    tstr, tfunction = chromosome.get_topology_information()

    name = (
        f"{chromosome.prefix}_{larger.get_name()}_{larger2.get_name()}_"
        f"{smaller.get_name()}_v{bb_dict[0]}_"
        f"f{chromosome.get_separated_string()}"
    )

    raise SystemExit(
        "move these conditions out to chomo iter at bottom, then use angle_hunter version"
    )
    forcefield_dict = get_forcefield_dict(forcefield)
    if options["bb_type"] == "tritopic":
        if (
            forcefield_dict["v_dict"]["b_n_b"]
            > forcefield_dict["v_dict"]["b_x_b"]
        ):
            return

        if (
            forcefield_dict["v_dict"]["b_n_b"]
            == forcefield_dict["v_dict"]["b_x_b"]
            and forcefield_dict["v_dict"]["b_n_b"] != 50  # noqa: PLR2004
        ):
            return
    elif options["bb_type"] == "tetratopic":
        if (
            forcefield_dict["v_dict"]["b_m_b"]
            > forcefield_dict["v_dict"]["b_y_b"]
        ):
            return

        if (
            forcefield_dict["v_dict"]["b_m_b"]
            == forcefield_dict["v_dict"]["b_y_b"]
            and forcefield_dict["v_dict"]["b_m_b"] != 50  # noqa: PLR2004
        ):
            return

    bbs = {
        bb: tuple(bb_dict[1][idx])
        for idx, bb in enumerate(
            (
                larger.get_building_block(),
                larger2.get_building_block(),
                smaller.get_building_block(),
            )
        )
    }

    cage = stk.ConstructedMolecule(tfunction(building_blocks=bbs))

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
                "bb_dict_idx": bb_dict[0],
                "bb_dict": bb_dict[1],
            },
        )

    else:
        ensemble = cgx.molecular.Ensemble(
            base_molecule=cage,
            base_mol_path=calculation_output / f"{name}_base.mol",
            conformer_xyz=calculation_output / f"{name}_ensemble.xyz",
            data_json=calculation_output / f"{name}_ensemble.json",
            overwrite=True,
        )
        logging.info("optimising %s from templates", name)
        for ti, template_file in enumerate(options["template_files"]):
            tmolecule = stk.BuildingBlock.init_from_file(str(template_file))

            temp_molecule = cage.with_position_matrix(
                tmolecule.get_position_matrix()
            )

            # Assign the forcefield.
            assigned_system = forcefield.assign_terms(
                molecule=temp_molecule,
                name=name,
                output_dir=calculation_output,
            )
            conformer = cgx.utilities.run_optimisation(
                assigned_system=assigned_system,
                name=name,
                file_suffix=f"temp_opt{ti}",
                output_dir=calculation_output,
                platform=None,
            )

            ensemble.add_conformer(
                conformer=conformer,
                source=f"temp_opt{ti}",
            )
        ensemble.write_conformers_to_file()

        final_conformer = ensemble.get_lowest_e_conformer()
        final_conformerid = final_conformer.conformer_id
        min_energy = final_conformer.energy_decomposition["total energy"][0]
        logging.info(
            "Min: %s from %s with: %s kJ.mol-1",
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
                "bb_dict_idx": bb_dict[0],
                "bb_dict": bb_dict[1],
            },
        )
        final_conformer.molecule.write(fina_mol_file)
        final_conformer.molecule.write(structure_output / f"{name}_optc.mol")

    properties = database.get_entry(key=name).properties
    if "energy_per_bb" not in properties:
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

        res_dict = {
            "strain_energy": fin_energy,
            "energy_per_bb": fin_energy / stoich_map(tstr),
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
        # Always want to extract target torions if present.
        g_measure = cgx.analysis.GeomMeasure(
            target_torsions=(
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
                cgx.terms.TargetTorsion(
                    search_string=("b", "a", "o", "a", "b"),
                    search_estring=("Pb", "Ba", "O", "Ba", "Pb"),
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
            absolute=True,
            as_search_string=True,
        )
        opt_pore_data = g_measure.calculate_min_distance(
            final_conformer.molecule
        )["min_distance"]

        dihedral_spread = np.std(dihedral_data["Pb_Ba_Ag_Ba_Pb"])
        all_values = dihedral_data["Pb_Ba_Ag_Ba_Pb"]

        # Then I want the distinct env as a coloured point.
        envs = []
        # Set envs here, should contain all.
        for d in all_values:
            if len(envs) == 0:
                envs.append(round(d, 0))
            else:
                within_distance_from_env = [
                    d
                    for de in envs
                    if abs(d - de) < EnvVariables.dihedral_state_threshold
                ]
                if len(within_distance_from_env) == 0:
                    envs.append(round(d, 0))
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
        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "prefix": chromosome.prefix,
                "chromosome": tuple(int(i) for i in chromosome.name),
                "tstr": tstr,
            },
        )


def low_resolution_function(  # noqa: PLR0915, C901
    database_path: pathlib.Path,
    figure_output: pathlib.Path,
    prefix: str,
) -> None:
    """Show low resolution data."""
    raise SystemExit("try refactor")
    database = cgx.utilities.AtomliteDatabase(database_path)
    tstr = prefix.split("_")[1]
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 5))

    if tstr in ("2P4", "3P6", "4P8", "4P82", "6P12", "6P122", "8P162", "8P16"):
        target_x = "$.forcefield_dict.v_dict.b_m_b"
        target_y = "$.forcefield_dict.v_dict.b_y_b"
        ax.set_xlabel("$b-m-b$ [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$b-y-b$ [$^\\circ$]", fontsize=16)
        ax2.set_xlabel("$b-m-b$ [$^\\circ$]", fontsize=16)
        ax2.set_ylabel("$b-y-b$ [$^\\circ$]", fontsize=16)
        df_properties = [
            "$.energy_per_bb",
            "$.forcefield_dict.v_dict.b_m_b",
            "$.forcefield_dict.v_dict.b_y_b",
            "$.bb_dict_idx",
        ]
        lims = (50, 90)

    elif tstr in ("2P3", "4P6", "4P62", "6P9", "8P12"):
        target_x = "$.forcefield_dict.v_dict.b_n_b"
        target_y = "$.forcefield_dict.v_dict.b_x_b"
        ax.set_xlabel("$b-n-b$ [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$b-x-b$ [$^\\circ$]", fontsize=16)
        ax2.set_xlabel("$b-n-b$ [$^\\circ$]", fontsize=16)
        ax2.set_ylabel("$b-x-b$ [$^\\circ$]", fontsize=16)
        df_properties = [
            "$.energy_per_bb",
            "$.forcefield_dict.v_dict.b_n_b",
            "$.forcefield_dict.v_dict.b_x_b",
            "$.bb_dict_idx",
        ]
        lims = (50, 120)

    vmin = 0
    vmax = 1
    dataframe = database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    logging.info("%s dataframe size: %s", prefix, len(dataframe))

    vx_stables = {i: 0 for i in set(dataframe["$.bb_dict_idx"])}
    vx_energies = {i: float("inf") for i in set(dataframe["$.bb_dict_idx"])}
    for xangle, yangle in it.product(
        set(dataframe[target_x]),
        set(dataframe[target_y]),
    ):
        pdata = dataframe.filter(pl.col(target_x) == xangle)
        pdata = pdata.filter(pl.col(target_y) == yangle)

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                bdata = pdata.filter(pl.col("$.bb_dict_idx") == bbidx)
                vx_energies[bbidx] = min(
                    (vx_energies[bbidx], bdata["$.energy_per_bb"].item(0))
                )

        if len(pdata) == 0:
            continue

        min_energy = min(pdata["$.energy_per_bb"])
        # Get stable states.
        pdata = pdata.filter(
            pl.col("$.energy_per_bb") <= EnvVariables.isomer_energy
        )

        if len(pdata) > 1:
            colour = "tab:orange"
            logging.info("found-ish %s", pdata["key"].item(0))

        elif len(pdata) == 1:
            colour = "tab:purple"
            logging.info("found %s", pdata["key"].item(0))

        else:
            colour = "white"

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                vx_stables[bbidx] += 1

        ax.scatter(
            xangle,
            yangle,
            c=colour,
            alpha=1.0,
            edgecolor="k",
            s=160,
            marker="s",
        )

        ax2.scatter(
            xangle,
            yangle,
            c=min_energy,
            alpha=1.0,
            edgecolor="k",
            s=160,
            marker="s",
            vmin=vmin,
            vmax=vmax,
            cmap="Blues_r",
        )

    ax.plot(lims, lims, c="k", ls="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(tstr, fontsize=16)

    ax2.plot(lims, lims, c="k", ls="--")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_title(tstr, fontsize=16)

    ax1.plot(
        sorted(vx_stables),
        [vx_stables[i] for i in sorted(vx_stables)],
        c="k",
        markersize=6,
        marker="o",
    )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_title(tstr, fontsize=16)
    ax1.set_xlabel("bb-state", fontsize=16)
    ax1.set_ylabel("count stable", fontsize=16)
    ax1.set_ylim(0, None)

    ax1a = ax1.twinx()
    ax1a.plot(
        sorted(vx_energies),
        [vx_energies[i] for i in sorted(vx_energies)],
        c="tab:red",
        markersize=6,
        marker="o",
    )
    ax1a.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        labelcolor="tab:red",
    )
    ax1a.set_ylabel(f"min {EnvVariables.eb_str}", fontsize=16, color="tab:red")
    ax1a.set_ylim(0, 2.0)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"min {EnvVariables.eb_str}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_output / f"{prefix}_am.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def high_resolution_function(  # noqa: PLR0915, PLR0912, C901
    database_path: pathlib.Path,
    low_res_database_path: pathlib.Path,
    figure_output: pathlib.Path,
    prefix: str,
) -> None:
    """Show low resolution data."""
    raise SystemExit("try refactor")
    database = cgx.utilities.AtomliteDatabase(database_path)
    low_res_database = cgx.utilities.AtomliteDatabase(low_res_database_path)
    tstr = prefix.split("_")[1]
    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 5))

    if tstr in ("2P4", "3P6", "4P8", "4P82", "6P12", "6P122", "8P162", "8P16"):
        target_x = "$.forcefield_dict.v_dict.b_m_b"
        target_y = "$.forcefield_dict.v_dict.b_y_b"
        ax.set_xlabel("$b-m-b$ [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$b-y-b$ [$^\\circ$]", fontsize=16)
        ax2.set_xlabel("$b-m-b$ [$^\\circ$]", fontsize=16)
        ax2.set_ylabel("$b-y-b$ [$^\\circ$]", fontsize=16)
        df_properties = [
            "$.energy_per_bb",
            "$.forcefield_dict.v_dict.b_m_b",
            "$.forcefield_dict.v_dict.b_y_b",
            "$.bb_dict_idx",
        ]
        lims = (50, 90)

    elif tstr in ("2P3", "4P6", "4P62", "6P9", "8P12"):
        target_x = "$.forcefield_dict.v_dict.b_n_b"
        target_y = "$.forcefield_dict.v_dict.b_x_b"
        ax.set_xlabel("$b-n-b$ [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$b-x-b$ [$^\\circ$]", fontsize=16)
        ax2.set_xlabel("$b-n-b$ [$^\\circ$]", fontsize=16)
        ax2.set_ylabel("$b-x-b$ [$^\\circ$]", fontsize=16)
        df_properties = [
            "$.energy_per_bb",
            "$.forcefield_dict.v_dict.b_n_b",
            "$.forcefield_dict.v_dict.b_x_b",
            "$.bb_dict_idx",
        ]
        lims = (50, 120)

    vmin = 0
    vmax = 1
    dataframe = database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    low_res_dataframe = low_res_database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )

    logging.info("%s dataframe size: %s", prefix, len(dataframe))
    logging.info(
        "%s low res dataframe size: %s", prefix, len(low_res_dataframe)
    )

    for xangle, yangle in it.product(
        set(low_res_dataframe[target_x]),
        set(low_res_dataframe[target_y]),
    ):
        pdata = low_res_dataframe.filter(pl.col(target_x) == xangle)
        pdata = pdata.filter(pl.col(target_y) == yangle)

        if len(pdata) == 0:
            continue

        min_energy = min(pdata["$.energy_per_bb"])
        # Get stable states.
        pdata = pdata.filter(
            pl.col("$.energy_per_bb") <= EnvVariables.isomer_energy
        )

        if len(pdata) > 1:
            colour = "tab:orange"

        elif len(pdata) == 1:
            colour = "tab:purple"

        else:
            colour = "white"

        ax.scatter(
            xangle,
            yangle,
            c=colour,
            alpha=1.0,
            edgecolor="k",
            s=160,
            marker="s",
        )

        ax2.scatter(
            xangle,
            yangle,
            c=min_energy,
            alpha=1.0,
            edgecolor="k",
            s=160,
            marker="s",
            vmin=vmin,
            vmax=vmax,
            cmap="Blues_r",
        )

    vx_stables = {i: 0 for i in set(dataframe["$.bb_dict_idx"])}
    vx_energies = {i: float("inf") for i in set(dataframe["$.bb_dict_idx"])}
    for xangle, yangle in it.product(
        set(dataframe[target_x]),
        set(dataframe[target_y]),
    ):
        pdata = dataframe.filter(pl.col(target_x) == xangle)
        pdata = pdata.filter(pl.col(target_y) == yangle)

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                bdata = pdata.filter(pl.col("$.bb_dict_idx") == bbidx)
                vx_energies[bbidx] = min(
                    (vx_energies[bbidx], bdata["$.energy_per_bb"].item(0))
                )

        if len(pdata) == 0:
            continue

        min_energy = min(pdata["$.energy_per_bb"])
        # Get stable states.
        pdata = pdata.filter(
            pl.col("$.energy_per_bb") <= EnvVariables.isomer_energy
        )

        if len(pdata) > 1:
            colour = "tab:orange"
            logging.info("found-ish %s", pdata["key"].item(0))

        elif len(pdata) == 1:
            colour = "tab:purple"
            logging.info("found %s", pdata["key"].item(0))

        else:
            colour = "white"

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                vx_stables[bbidx] += 1

        ax.scatter(
            xangle,
            yangle,
            c=colour,
            alpha=1.0,
            edgecolor="k",
            s=60,
            marker="o",
        )

        ax2.scatter(
            xangle,
            yangle,
            c=min_energy,
            alpha=1.0,
            edgecolor="k",
            s=60,
            marker="o",
            vmin=vmin,
            vmax=vmax,
            cmap="Blues_r",
        )

    ax.plot(lims, lims, c="k", ls="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(tstr, fontsize=16)

    ax2.plot(lims, lims, c="k", ls="--")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_title(tstr, fontsize=16)

    ax1.plot(
        sorted(vx_stables),
        [vx_stables[i] for i in sorted(vx_stables)],
        c="k",
        markersize=6,
        marker="o",
    )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_title(tstr, fontsize=16)
    ax1.set_xlabel("bb-state", fontsize=16)
    ax1.set_ylabel("count stable", fontsize=16)
    ax1.set_ylim(0, None)

    ax1a = ax1.twinx()
    ax1a.plot(
        sorted(vx_energies),
        [vx_energies[i] for i in sorted(vx_energies)],
        c="tab:red",
        markersize=6,
        marker="o",
    )
    ax1a.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        labelcolor="tab:red",
    )
    ax1a.set_ylabel(f"min {EnvVariables.eb_str}", fontsize=16, color="tab:red")
    ax1a.set_ylim(0, 2.0)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"min {EnvVariables.eb_str}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_output / f"{prefix}_am.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_function(
    database_path: pathlib.Path,
    figure_output: pathlib.Path,
    prefix: str,
) -> None:
    """Plot the angle map."""
    raise SystemExit("try refactor")
    run_resolution = 0 if "r1" not in prefix.split("_")[0] else 1

    if run_resolution == 0:
        low_resolution_function(
            database_path=database_path,
            figure_output=figure_output,
            prefix=prefix,
        )
    elif run_resolution == 1:
        database_parent = database_path.parents[0]
        preprefix = prefix.replace("r1", "")
        low_res_database_path = database_parent / database_path.name.replace(
            prefix, preprefix
        )

        high_resolution_function(
            database_path=database_path,
            low_res_database_path=low_res_database_path,
            figure_output=figure_output,
            prefix=prefix,
        )


def _parse_args() -> argparse.Namespace:
    raise SystemExit("try refactor")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_calcs",
        help="whether to run construction or not",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run script."""
    args = _parse_args()
    structure_output = EnvVariables.cg_structures
    calculation_output = EnvVariables.cg_calculations
    figure_output = EnvVariables.cg_figures
    data_output = EnvVariables.cg_outputdata

    present_beads_2p4 = (
        arm_bead,
        binder_bead,
        tetragonal_bead,
        tetragonal_bead2,
        core_bead,
    )
    present_beads_2p3 = (
        arm_bead,
        binder_bead,
        trigonal_bead,
        core_bead,
        trigonal_bead2,
    )
    large_gene_4x = (
        cgx.molecular.FourC1Arm(bead=tetragonal_bead, abead1=binder_bead),
    )
    large_gene_3x = (
        cgx.molecular.ThreeC1Arm(bead=trigonal_bead, abead1=binder_bead),
    )
    large_gene_4x2 = (
        cgx.molecular.FourC1Arm(bead=tetragonal_bead2, abead1=binder_bead),
    )
    large_gene_3x2 = (
        cgx.molecular.ThreeC1Arm(bead=trigonal_bead2, abead1=binder_bead),
    )
    small_gene = (cgx.molecular.TwoC1Arm(bead=core_bead, abead1=arm_bead),)

    low_resolution_zones_2p3 = {
        "bnb": ("angle", [50, 60, 70, 80, 90, 100, 110], 1e2),
        "bxb": ("angle", [50, 60, 70, 80, 90, 100, 110, 120], 1e2),
        "bac": ("angle", 125, 1e2),
    }
    low_resolution_zones_2p4 = {
        "bmb": ("pyramid", [50, 60, 70, 80], 1e2),
        "byb": ("pyramid", [50, 60, 70, 80, 90], 1e2),
        "bac": ("angle", 135, 1e2),
    }

    lt2bnbzones = {"4P6": create_zone(dmin=50, dmax=100, resolution=5)}
    lt2bxbzones = {"4P6": create_zone(dmin=50, dmax=120, resolution=5)}
    lt1bmbzones = {"6P12": create_zone(dmin=50, dmax=80, resolution=5)}
    lt1bybzones = {"6P12": create_zone(dmin=80, dmax=90, resolution=2)}

    studies = {
        "lt1_6P12": {
            "topology": ("6P12", stk.cage.M6L12Cube),
            "definer_dict": definer_dict_2p4,
            "present_beads": present_beads_2p4,
            "large_gene": large_gene_4x,
            "large_gene2": large_gene_4x2,
            "small_gene": small_gene,
            "bb_ratio": (1, 1),
            "definer_dict_updates": low_resolution_zones_2p4,
            "bb_type": "tetratopic",
        },
        "lt1r1_6P12": {
            "topology": ("6P12", stk.cage.M6L12Cube),
            "definer_dict": definer_dict_2p4,
            "present_beads": present_beads_2p4,
            "large_gene": large_gene_4x,
            "large_gene2": large_gene_4x2,
            "small_gene": small_gene,
            "bb_ratio": (1, 1),
            "definer_dict_updates": {
                "bmb": ("angle", lt1bmbzones["6P12"], 1e2),
                "byb": ("angle", lt1bybzones["6P12"], 1e2),
                "bac": ("angle", 135, 1e2),
            },
            "bb_type": "tetratopic",
        },
        "lt2_4P6": {
            "topology": ("4P6", stk.cage.FourPlusSix),
            "definer_dict": definer_dict_2p3,
            "present_beads": present_beads_2p3,
            "large_gene": large_gene_3x,
            "large_gene2": large_gene_3x2,
            "small_gene": small_gene,
            "bb_ratio": (1, 1),
            "definer_dict_updates": low_resolution_zones_2p3,
            "bb_type": "tritopic",
        },
        "lt2r1_4P6": {
            "topology": ("4P6", stk.cage.FourPlusSix),
            "definer_dict": definer_dict_2p3,
            "present_beads": present_beads_2p3,
            "large_gene": large_gene_3x,
            "large_gene2": large_gene_3x2,
            "small_gene": small_gene,
            "bb_ratio": (1, 1),
            "definer_dict_updates": {
                "bnb": ("angle", lt2bnbzones["4P6"], 1e2),
                "bxb": ("angle", lt2bxbzones["4P6"], 1e2),
                "bac": ("angle", 125, 1e2),
            },
            "bb_type": "tritopic",
        },
    }

    for study in studies:
        definer_dict = studies[study]["definer_dict"]
        cage_topology = studies[study]["topology"]

        for term in studies[study]["definer_dict_updates"]:
            definer_dict[term] = studies[study]["definer_dict_updates"][term]

        prefix = f"{study}"
        database_path = data_output / f"{prefix}.db"
        calculations_done_file = data_output / f"{prefix}.done"

        possible_bbdicts = cgx.scram.get_potential_bb_dicts(
            tstr=cage_topology[0],
            ratio=studies[study]["bb_ratio"],
            bb_type=studies[study]["bb_type"],
        )
        logging.info(
            "there are %s possible BB dicts for %s",
            len(possible_bbdicts),
            prefix,
        )

        if len(possible_bbdicts) == 0:
            continue

        if args.run_calcs and not calculations_done_file.exists():
            for bdict in possible_bbdicts:
                chromosome_gen = cgx.systems_optimisation.ChromosomeGenerator(
                    prefix=prefix,
                    present_beads=studies[study]["present_beads"],
                    vdw_bond_cutoff=2,
                )

                chromosome_gen.add_gene(
                    iteration=[cage_topology],
                    gene_type="topology",
                )
                chromosome_gen.add_gene(
                    iteration=studies[study]["large_gene"],
                    gene_type="precursor",
                )
                chromosome_gen.add_gene(
                    iteration=studies[study]["large_gene2"],
                    gene_type="precursor",
                )
                chromosome_gen.add_gene(
                    iteration=studies[study]["small_gene"],
                    gene_type="precursor",
                )
                chromosome_gen.add_gene(
                    iteration=(bdict,),
                    gene_type="precursor",
                )
                chromosome_gen.add_forcefield_dict(definer_dict=definer_dict)

                template_population = chromosome_gen.select_random_population(
                    generator=np.random.default_rng(1824),
                    size=3,
                )

                # Pick random structures from chromosome to build as
                # template.
                template_files = []
                for chromosome in template_population:
                    (
                        larger,
                        larger2,
                        smaller,
                        bb_dict,
                    ) = chromosome.get_precursors()

                    template_name = (
                        f"temp_{chromosome.prefix}_{larger.get_name()}_"
                        f"{larger2.get_name()}_{smaller.get_name()}_"
                        f"v{bb_dict[0]}_f{chromosome.get_separated_string()}"
                    )

                    template_structure_function(
                        chromosome=chromosome,
                        database_path=database_path,
                        calculation_output=calculation_output,
                        structure_output=structure_output,
                        options={},
                    )

                    template_files.append(
                        calculation_output / f"{template_name}_final.mol"
                    )

                # Go through all in chromosome, and only opt from templates.
                for chromosome in chromosome_gen.yield_chromosomes():
                    structure_function(
                        chromosome=chromosome,
                        database_path=database_path,
                        calculation_output=calculation_output,
                        structure_output=structure_output,
                        options={
                            "template_files": template_files,
                            "bb_type": studies[study]["bb_type"],
                        },
                    )

            # Create a done file.
            calculations_done_file.open("a").close()

        plot_function(
            database_path=database_path,
            figure_output=figure_output,
            prefix=prefix,
        )
    raise SystemExit("put the founds into a library somewhere")


if __name__ == "__main__":
    main()
