"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import warnings
from collections import abc, defaultdict

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openmm
import polars as pl
import stk
from rdkit import RDLogger

from model_enumeration.utilities import (
    arm_bead,
    binder_bead,
    core_bead,
    core_bead2,
    create_zone,
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
    "ao": ("bond", 1.5, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "bac": ("angle", 90, 1e2),
    "aca": ("angle", 180, 1e2),
    "bao": ("angle", 90, 1e2),
    "aoa": ("angle", 180, 1e2),
    # Torsions.
    "bacab": ("tors", "0134", 180, 50, 1),
    "baoab": ("tors", "0134", 180, 50, 1),
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "o": ("nb", 10.0, 1.0),
}

definer_dict_2p3 = {
    # Bonds.
    "nb": ("bond", 1.5, 1e5),
    "ab": ("bond", 1.0, 1e5),
    "ac": ("bond", 1.5, 1e5),
    "ao": ("bond", 1.5, 1e5),
    # Angles.
    "bnb": ("angle", 120, 1e2),
    "nba": ("angle", 180, 1e2),
    "bac": ("angle", 90, 1e2),
    "aca": ("angle", 180, 1e2),
    "bao": ("angle", 90, 1e2),
    "aoa": ("angle", 180, 1e2),
    # Torsions.
    "bacab": ("tors", "0134", 180, 50, 1),
    "baoab": ("tors", "0134", 180, 50, 1),
    # Nonbondeds.
    "n": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "o": ("nb", 10.0, 1.0),
}


def template_structure_function(  # noqa: PLR0915
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,  # noqa: ARG001
) -> None:
    """Generate a structure from a chromosome."""
    database = cgx.utilities.AtomliteDatabase(database_path)

    forcefield = chromosome.get_forcefield()
    larger, smaller, smaller2, bb_dict = chromosome.get_precursors()
    tstr, tfunction = chromosome.get_topology_information()

    name = (
        f"{chromosome.prefix}_{larger.get_name()}_{smaller.get_name()}_"
        f"{smaller2.get_name()}_v{bb_dict[0]}_"
        f"f{chromosome.get_separated_string()}"
    )

    bbs = {
        bb: tuple(bb_dict[1][idx])
        for idx, bb in enumerate(
            (
                larger.get_building_block(),
                smaller.get_building_block(),
                smaller2.get_building_block(),
            )
        )
    }

    template_name = (
        f"temp_{chromosome.prefix}_{larger.get_name()}_{smaller.get_name()}_"
        f"{smaller2.get_name()}_v{bb_dict[0]}_"
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
        final_conformer.molecule.write(structure_output / f"{name}_optc.mol")
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
    database = cgx.utilities.AtomliteDatabase(database_path)

    forcefield = chromosome.get_forcefield()
    larger, smaller, smaller2, bb_dict = chromosome.get_precursors()
    tstr, tfunction = chromosome.get_topology_information()

    name = (
        f"{chromosome.prefix}_{larger.get_name()}_{smaller.get_name()}_"
        f"{smaller2.get_name()}_v{bb_dict[0]}_"
        f"f{chromosome.get_separated_string()}"
    )

    forcefield_dict = forcefield.get_forcefield_dictionary()

    bbs = {
        bb: tuple(bb_dict[1][idx])
        for idx, bb in enumerate(
            (
                larger.get_building_block(),
                smaller.get_building_block(),
                smaller2.get_building_block(),
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
        final_conformer.molecule.write(structure_output / f"{name}_optc.mol")

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

            ensemble.add_conformer(conformer=conformer, source=f"temp_opt{ti}")
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
        # Always want to extract target torions if present.
        g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
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
        envs = []
        # Set envs here, should contain all.
        for d in all_values:
            if len(envs) == 0:
                envs.append(round(d, 0))
            else:
                within_distance_from_env = [
                    d
                    for de in envs
                    if abs(d - de) < dihedral_state_threshold()
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


def low_resolution_function(  # noqa: PLR0915
    database_path: pathlib.Path,
    figure_output: pathlib.Path,
    prefix: str,
) -> None:
    """Show low resolution data."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    tstr = prefix.split("_")[1]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    target_x = "$.forcefield_dict.v_dict.b_a_c"
    target_y = "$.forcefield_dict.v_dict.b_a_o"
    ax2.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
    ax2.set_ylabel("$bao$ [$^\\circ$]", fontsize=16)

    df_properties = [
        "$.energy_per_bb",
        "$.forcefield_dict.v_dict.b_a_c",
        "$.forcefield_dict.v_dict.b_a_o",
        "$.bb_dict_idx",
    ]

    vmin = 0
    vmax = 1
    dataframe = database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    logging.info("%s dataframe size: %s", prefix, len(dataframe))
    if len(dataframe) == 0:
        return

    vx_stables = {i: 0 for i in set(dataframe["$.bb_dict_idx"])}
    vx_energies = {i: float("inf") for i in set(dataframe["$.bb_dict_idx"])}
    for xangle, yangle in it.product(
        set(dataframe[target_x]),
        set(dataframe[target_y]),
    ):
        pdata = dataframe.filter(pl.col(target_x) == xangle)
        pdata = pdata.filter(pl.col(target_y) == yangle)

        if len(pdata) == 0:
            continue

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                bdata = pdata.filter(pl.col("$.bb_dict_idx") == bbidx)
                vx_energies[bbidx] = min(
                    (vx_energies[bbidx], bdata["$.energy_per_bb"].item(0))
                )

        min_energy = min(pdata["$.energy_per_bb"])
        # Get stable states.
        pdata = pdata.filter(pl.col("$.energy_per_bb") <= isomer_energy())

        if len(pdata) > 1:
            logging.info("found-ish %s", pdata["key"].item(0))

        elif len(pdata) == 1:
            logging.info("found %s", pdata["key"].item(0))

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                vx_stables[bbidx] += 1

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

    ax2.plot((90, 180), (90, 180), c="k", ls="--")
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
    ax1.set_xlabel("bb configuration", fontsize=16)
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
    ax1a.set_ylabel(f"min {eb_str()}", fontsize=16, color="tab:red")
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
    cbar.set_label(f"min {eb_str()}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_output / f"{prefix}_am.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def high_resolution_function(  # noqa: PLR0915, C901, PLR0912
    database_path: pathlib.Path,
    low_res_database_path: pathlib.Path,
    figure_output: pathlib.Path,
    structure_output: pathlib.Path,
    prefix: str,
) -> None:
    """Show low resolution data."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    low_res_database = cgx.utilities.AtomliteDatabase(low_res_database_path)
    tstr = prefix.split("_")[1]
    fig, (ax1, ax, ax2) = plt.subplots(ncols=3, figsize=(16, 5))
    figrank, axrank = plt.subplots(figsize=(8, 5))

    candidates = []

    target_x = "$.forcefield_dict.v_dict.b_a_c"
    target_y = "$.forcefield_dict.v_dict.b_a_o"
    ax.set_xlabel("$b-a-c$ [$^\\circ$]", fontsize=16)
    ax.set_ylabel("$b-a-o$ [$^\\circ$]", fontsize=16)
    ax2.set_xlabel("$b-a-c$ [$^\\circ$]", fontsize=16)
    ax2.set_ylabel("$b-a-o$ [$^\\circ$]", fontsize=16)
    df_properties = [
        "$.energy_per_bb",
        "$.forcefield_dict.v_dict.b_a_c",
        "$.forcefield_dict.v_dict.b_a_o",
        "$.bb_dict_idx",
    ]

    vmin = 0
    vmax = 1
    dataframe = database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    if len(dataframe) == 0:
        return
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
        pdata = pdata.filter(pl.col("$.energy_per_bb") <= isomer_energy())

        ax2.scatter(
            xangle,
            yangle,
            c=min_energy,
            alpha=0.5,
            edgecolor="k",
            s=160,
            marker="s",
            vmin=vmin,
            vmax=vmax,
            cmap="Blues_r",
        )

    vx_stables = {i: 0 for i in set(dataframe["$.bb_dict_idx"])}
    num_poss = len(vx_stables)
    vx_energies = {i: float("inf") for i in set(dataframe["$.bb_dict_idx"])}
    region_types = {}
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
        pdata = pdata.filter(pl.col("$.energy_per_bb") <= isomer_energy())
        if len(pdata) > 1:
            stable_string = "|".join(
                [str(i) for i in list(pdata["$.bb_dict_idx"])]
            )
            stable_count = len(pdata)
            if stable_count < 20:  # noqa: PLR2004
                logging.info("found-ish bb_ids: %s", stable_string)
            else:
                logging.info("found-ish m(%s)", stable_count)

        elif len(pdata) == 1:
            stable_string = "|".join(
                sorted([str(i) for i in list(pdata["$.bb_dict_idx"])])
            )
            stable_count = len(pdata)
            logging.info("found bbid: %s", stable_string)

        else:
            stable_string = ""
            stable_count = 0

        for bbidx in list(pdata["$.bb_dict_idx"]):
            if xangle != yangle:
                vx_stables[bbidx] += 1

        if stable_count > 0:
            if stable_string not in region_types:
                region_types[stable_string] = []
            region_types[stable_string].append((xangle, yangle))

            if stable_count < 20:  # noqa: PLR2004
                for i in pdata["key"]:
                    filename = f"{i}_optc.mol"
                    if not (structure_output / filename).exists():
                        database.get_molecule(i).write(
                            structure_output / filename
                        )
                    candidates.append(filename)

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

        # Combination ranking.
        axrank.scatter(
            xangle,
            yangle - xangle,
            c=stable_count if stable_count > 0 else "w",
            ec="k",
            vmin=0,
            vmax=10,
            s=60,
        )

    for stable_string, rtypes in region_types.items():
        stable_count = len(stable_string.split("|"))
        ax.scatter(
            [i[0] for i in rtypes],
            [i[1] for i in rtypes],
            s=60,
            alpha=1.0,
            label=stable_string if stable_count < 5 else f"m({stable_count})",  # noqa: PLR2004
        )

    ax.plot((90, 180), (90, 180), c="k", ls="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(tstr, fontsize=16)
    ax.legend(fontsize=16)

    axrank.tick_params(axis="both", which="major", labelsize=16)
    axrank.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
    axrank.set_ylabel("$bao$ - $bac$ [$^\\circ$]", fontsize=16)

    ax2.plot((90, 180), (90, 180), c="k", ls="--")
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
    ax1.set_ylabel(f"count stable of {num_poss}", fontsize=16)
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
    ax1a.set_ylabel(f"min {eb_str()}", fontsize=16, color="tab:red")
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
    cbar.set_label(f"min {eb_str()}", fontsize=16)

    with (figure_output / f"{prefix}_am.txt").open("w") as f:
        f.write(" ".join(sorted(candidates)))
        f.write("\n")

    fig.tight_layout()
    fig.savefig(
        figure_output / f"{prefix}_am.png",
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_output / f"{prefix}_am.pdf",
        dpi=360,
        bbox_inches="tight",
    )

    figrank.tight_layout()
    figrank.savefig(
        figure_output / f"{prefix}_rank.png",
        dpi=360,
        bbox_inches="tight",
    )
    figrank.savefig(
        figure_output / f"{prefix}_rank.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_calcs",
        help="whether to run construction or not",
        action="store_true",
    )

    return parser.parse_args()


def run_workflow(  # noqa: C901, PLR0913
    run_calcs: bool,
    study: str,
    study_dict: dict,
    data_dir: pathlib.Path,
    structure_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
) -> None:
    """Run the workflow."""
    definer_dict = study_dict["definer_dict"]
    cage_topology = study_dict["topology"]

    for term in study_dict["definer_dict_updates"]:
        definer_dict[term] = study_dict["definer_dict_updates"][term]

    prefix = f"{study}"
    database_path = data_dir / f"{prefix}.db"
    calculations_done_file = data_dir / f"{prefix}.done"

    possible_bbdicts = cgx.scram.get_potential_bb_dicts(
        tstr=cage_topology[0],
        ratio=study_dict["bb_ratio"],
        study_type="ditopic",
    )
    logging.info(
        "there are %s possible BB dicts for %s",
        len(possible_bbdicts),
        prefix,
    )

    if "8P16" in study:
        logging.info(
            "but for 8P16 we are only using two chosen ones with bridges!"
        )
        target_bbdict = (8, 9, 10, 11, 12, 13, 14, 15)
        chosen_bbdicts = []
        for bbdict in possible_bbdicts:
            if tuple(sorted(bbdict[1][1])) == target_bbdict:
                chosen_bbdicts.append(bbdict)
            if tuple(sorted(bbdict[1][2])) == target_bbdict:
                chosen_bbdicts.append(bbdict)

        possible_bbdicts = tuple(chosen_bbdicts)

    logging.info("buidling %s BB dicts for %s", len(possible_bbdicts), prefix)
    if len(possible_bbdicts) == 0:
        return

    if run_calcs and not calculations_done_file.exists():
        for bdict in possible_bbdicts:
            chromosome_gen = cgx.systems_optimisation.ChromosomeGenerator(
                prefix=prefix,
                present_beads=study_dict["present_beads"],
                vdw_bond_cutoff=2,
            )

            chromosome_gen.add_gene(
                iteration=[cage_topology],
                gene_type="topology",
            )
            chromosome_gen.add_gene(
                iteration=study_dict["large_gene"],
                gene_type="precursor",
            )
            chromosome_gen.add_gene(
                iteration=study_dict["small_gene"],
                gene_type="precursor",
            )
            chromosome_gen.add_gene(
                iteration=study_dict["small_gene2"],
                gene_type="precursor",
            )
            chromosome_gen.add_gene(
                iteration=(bdict,),
                gene_type="precursor",
            )
            chromosome_gen.add_forcefield_dict(definer_dict=definer_dict)

            # Pick random structures from chromosome to build as
            # template.
            template_population = chromosome_gen.select_random_population(
                generator=np.random.default_rng(1824),
                size=3,
            )
            template_files = []
            for chromosome in template_population:
                (
                    larger,
                    smaller,
                    smaller2,
                    bb_dict,
                ) = chromosome.get_precursors()

                template_name = (
                    f"temp_{chromosome.prefix}_{larger.get_name()}_"
                    f"{smaller.get_name()}_{smaller2.get_name()}_"
                    f"v{bb_dict[0]}_f{chromosome.get_separated_string()}"
                )

                template_structure_function(
                    chromosome=chromosome,
                    database_path=database_path,
                    calculation_output=calculation_dir,
                    structure_output=structure_dir,
                    options={},
                )

                template_files.append(
                    calculation_dir / f"{template_name}_final.mol"
                )

            # Go through all in chromosome, and only opt from templates.
            for chromosome in chromosome_gen.yield_chromosomes():
                forcefield_dict = (
                    chromosome.get_forcefield().get_forcefield_dictionary()
                )
                # Skip matching bao < bac.
                if (
                    forcefield_dict["v_dict"]["b_a_c"]
                    > forcefield_dict["v_dict"]["b_a_o"]
                ):
                    continue

                # Include one case of bac == bao.
                if (
                    forcefield_dict["v_dict"]["b_a_c"]
                    == forcefield_dict["v_dict"]["b_a_o"]
                    and forcefield_dict["v_dict"]["b_a_c"] != 90  # noqa: PLR2004
                ):
                    continue

                structure_function(
                    chromosome=chromosome,
                    database_path=database_path,
                    calculation_output=calculation_dir,
                    structure_output=structure_dir,
                    options={"template_files": template_files},
                )

        # Create a done file.
        calculations_done_file.open("a").close()


def get_high_res_zones(
    lr_data: pathlib.Path,
) -> dict[str, abc.Sequence[float]]:
    """Get zones near low energy low-res points."""
    database = cgx.utilities.AtomliteDatabase(lr_data)

    df_properties = [
        "$.energy_per_bb",
        "$.forcefield_dict.v_dict.b_a_c",
        "$.forcefield_dict.v_dict.b_a_o",
        "$.bb_dict_idx",
    ]
    dataframe = database.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    if len(dataframe) == 0:
        return {"bac": [], "bao": []}

    pdata = dataframe.filter(pl.col("$.energy_per_bb") <= isomer_energy() * 3)

    xys = {
        (i, j)
        for i, j in zip(
            pdata["$.forcefield_dict.v_dict.b_a_c"],
            pdata["$.forcefield_dict.v_dict.b_a_o"],
            strict=False,
        )
    }

    points = []
    resolution = 5
    for x, y in xys:
        points.extend(
            (i, j)
            for i, j in it.product(
                range(x - resolution * 2, x + resolution * 2 + 1, resolution),
                range(y - resolution * 2, y + resolution * 2 + 1, resolution),
            )
        )
    sets = set(points)
    return {
        "bac": sorted({i[0] for i in sets if i[0] <= 180 and i[0] >= 0}),  # noqa: PLR2004
        "bao": sorted({i[1] for i in sets if i[1] <= 180 and i[1] >= 0}),  # noqa: PLR2004
    }


def main() -> None:
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "angle_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "angle_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "angle_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "angle_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    present_beads_2p4 = (
        arm_bead,
        binder_bead,
        tetragonal_bead,
        core_bead,
        core_bead2,
    )
    present_beads_2p3 = (
        arm_bead,
        binder_bead,
        trigonal_bead,
        core_bead,
        core_bead2,
    )
    large_gene_4x = (
        cgx.molecular.FourC1Arm(bead=tetragonal_bead, abead1=binder_bead),
    )
    large_gene_3x = (
        cgx.molecular.ThreeC1Arm(bead=trigonal_bead, abead1=binder_bead),
    )
    small_gene = (cgx.molecular.TwoC1Arm(bead=core_bead, abead1=arm_bead),)
    small_gene2 = (cgx.molecular.TwoC1Arm(bead=core_bead2, abead1=arm_bead),)

    low_resolution_zones = {
        "bac": ("angle", [90, 105, 120, 135, 150, 165], 1e2),
        "bao": ("angle", [90, 105, 120, 135, 150, 165, 180], 1e2),
    }

    topology_list = (
        # Scanning ditopic + tetratopic in 1:1 ratio.
        ("3P6", stk.cage.M3L6, (1, 1), "2+4"),
        ("4P8", cgx.topologies.CGM4L8, (1, 1), "2+4"),
        ("4P82", cgx.topologies.M4L82, (1, 1), "2+4"),
        ("6P12", stk.cage.M6L12Cube, (1, 1), "2+4"),
        ("6P122", cgx.topologies.M6L122, (1, 1), "2+4"),
        ("8P16", stk.cage.EightPlusSixteen, (1, 1), "2+4"),
        # Scanning ditopic + tritopic.
        # By definition, sometimes the ratio cannot be 1:1.
        ("4P6", stk.cage.FourPlusSix, (1, 1), "2+3"),
        ("4P62", stk.cage.FourPlusSix2, (1, 1), "2+3"),
    )

    lr_studies = {
        f"lr_{tstr}": {
            "topology": (tstr, tfunc),
            "definer_dict": definer_dict_2p4
            if mtype == "2+4"
            else definer_dict_2p3,
            "present_beads": present_beads_2p4
            if mtype == "2+4"
            else present_beads_2p3,
            "large_gene": large_gene_4x if mtype == "2+4" else large_gene_3x,
            "small_gene": small_gene,
            "small_gene2": small_gene2,
            "bb_ratio": ratio,
            "definer_dict_updates": low_resolution_zones,
        }
        for tstr, tfunc, ratio, mtype in topology_list
    }

    for study, study_dict in lr_studies.items():
        run_workflow(
            run_calcs=args.run_calcs,
            study=study,
            study_dict=study_dict,
            data_dir=data_dir,
            calculation_dir=calculation_dir,
            structure_dir=structure_dir,
        )

        low_resolution_function(
            database_path=data_dir / f"{study}.db",
            figure_output=figure_dir,
            prefix=study,
        )

    # Get the high-res zones.
    hr_studies = {}
    for tstr, tfunc, ratio, mtype in topology_list:
        lr_data = data_dir / f"lr_{tstr}.db"
        hr_zones = get_high_res_zones(lr_data)

        if hr_zones["bac"] == []:
            continue

        hr_studies[f"hr_{tstr}"] = {
            "topology": (tstr, tfunc),
            "definer_dict": definer_dict_2p4
            if mtype == "2+4"
            else definer_dict_2p3,
            "present_beads": present_beads_2p4
            if mtype == "2+4"
            else present_beads_2p3,
            "large_gene": large_gene_4x if mtype == "2+4" else large_gene_3x,
            "small_gene": small_gene,
            "small_gene2": small_gene2,
            "bb_ratio": ratio,
            "definer_dict_updates": {
                "bac": ("angle", hr_zones["bac"], 1e2),
                "bao": ("angle", hr_zones["bao"], 1e2),
            },
        }

    for study, study_dict in hr_studies.items():
        run_workflow(
            run_calcs=args.run_calcs,
            study=study,
            study_dict=study_dict,
            data_dir=data_dir,
            calculation_dir=calculation_dir,
            structure_dir=structure_dir,
        )

        lr_study = study.replace("hr_", "lr_")

        high_resolution_function(
            database_path=data_dir / f"{study}.db",
            low_res_database_path=data_dir / f"{lr_study}.db",
            figure_output=figure_dir,
            structure_output=structure_dir,
            prefix=study,
        )
    raise SystemExit("change all study names below")
    raise SystemExit("use the scam bb config method")
    raise SystemExit(
        "turn into a chemiscope and single plot of knietic self-sort, basically."
    )
    raise SystemExit("put the founds into a library somewhere")
    high_res_baczones = {
        "3P6": create_zone(dmin=95, dmax=115, resolution=5),
        "4P8": create_zone(dmin=90, dmax=140, resolution=5),
        "4P82": create_zone(dmin=90, dmax=140, resolution=5),
        "6P12": create_zone(dmin=100, dmax=150, resolution=5),
        "6P122": create_zone(dmin=90, dmax=130, resolution=5),
        "8P16": create_zone(dmin=100, dmax=150, resolution=5),
        "2P3": create_zone(dmin=90, dmax=100, resolution=2),
        "4P6": create_zone(dmin=100, dmax=140, resolution=2),
        "4P62": create_zone(dmin=90, dmax=140, resolution=2),
        "6P9": create_zone(dmin=90, dmax=150, resolution=2),
        "8P12": create_zone(dmin=90, dmax=150, resolution=2),
    }
    high_res_baozones = {
        "3P6": create_zone(dmin=110, dmax=130, resolution=5),
        "4P8": create_zone(dmin=100, dmax=150, resolution=5),
        "4P82": create_zone(dmin=120, dmax=160, resolution=5),
        "6P12": create_zone(dmin=120, dmax=180, resolution=5),
        "6P122": create_zone(dmin=120, dmax=180, resolution=5),
        "8P16": create_zone(dmin=140, dmax=180, resolution=5),
        "2P3": create_zone(dmin=90, dmax=120, resolution=5),
        "4P6": create_zone(dmin=110, dmax=160, resolution=5),
        "4P62": create_zone(dmin=100, dmax=140, resolution=5),
        "6P9": create_zone(dmin=90, dmax=180, resolution=5),
        "8P12": create_zone(dmin=90, dmax=180, resolution=5),
    }


if __name__ == "__main__":
    main()
