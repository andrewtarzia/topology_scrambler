"""Script to generate and optimise CG models."""

import logging
import pathlib
import stk
import os
from copy import deepcopy
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from openmm import openmm
from rdkit import RDLogger
import stko

from min_utilities import (
    binder_bead,
    abead_d,
    abead_c,
    cbead_d,
    cbead_c,
    ebead_c,
    ebead_d,
    tetra_bead,
    forcefield_lf_ls1,
    forcefield_lf_ls9,
    SixBead,
    forcefield_la_st52,
    forcefield_la_st5,
)
import cgexplore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def eb_str(no_unit=False):
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def analyse_cage(database_path, name, forcefield, iterator):
    database = cgexplore.utilities.AtomliteDatabase(database_path)
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
            sum(energy_decomp[i] for i in energy_decomp if "total energy" not in i)
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
                openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(openmm.unit.angstrom)

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(openmm.unit.degrees)
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
            v_dict[at.bead_class] = at.sigma.value_in_unit(openmm.unit.angstrom)
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

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy / iterator.get_num_building_blocks(),
                "pair": name.split("_")[0] + "_" + name.split("_")[1],
                "num_components": num_components,
                "multiplier": name.split("_")[2],
            },
        )


def optimise_cage(
    molecule,
    name,
    output_dir,
    forcefield,
    platform,
    database_path,
):
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")

    database = cgexplore.utilities.AtomliteDatabase(database_path)
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return cgexplore.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if os.path.exists(fina_mol_file):
        ensemble = cgexplore.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
            conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
            data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
            overwrite=False,
        )
        conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": conformer.energy_decomposition,
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)

    ensemble = cgexplore.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )
    temp_molecule = cgexplore.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    logging.info(f"optimisation of {name}")
    conformer = cgexplore.utilities.run_optimisation(
        assigned_system=cgexplore.forcefields.AssignedSystem(
            molecule=temp_molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        name=name,
        file_suffix="opt1",
        output_dir=output_dir,
        # max_iterations=50,
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in cgexplore.utilities.yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
                molecule=test_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="sopt",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = cgexplore.utilities.run_soft_md_cycle(
        name=name,
        assigned_system=cgexplore.forcefields.AssignedSystem(
            molecule=ensemble.get_lowest_e_conformer().molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        output_dir=output_dir,
        suffix="smd",
        bond_ff_scale=10,
        angle_ff_scale=10,
        temperature=300 * openmm.unit.kelvin,
        num_steps=num_steps,
        time_step=0.5 * openmm.unit.femtoseconds,
        friction=1.0 / openmm.unit.picosecond,
        reporting_freq=traj_freq,
        traj_freq=traj_freq,
        platform=platform,
    )
    if soft_md_trajectory is None:
        logging.info(f"!!!!! {name} MD exploded !!!!!")
        # md_exploded = True
        raise ValueError("OpenMM Exception")

    soft_md_data = soft_md_trajectory.get_data()
    logging.info(f"collected trajectory {len(soft_md_data)} confs long")
    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        # md_failed = True
        raise ValueError()

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
                molecule=md_conformer.molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="smd")
    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        f"Min. energy conformer: {min_energy_conformerid} from "
        f"{min_energy_conformer.source}"
        f" with energy: {min_energy} kJ.mol-1"
    )

    # Add to atomlite database.
    database.add_molecule(molecule=min_energy_conformer.molecule, key=name)
    database.add_properties(
        key=name,
        property_dict={
            "energy_decomposition": min_energy_conformer.energy_decomposition,
            "source": min_energy_conformer.source,
            "optimised": True,
        },
    )
    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


class UnalignedM2L4(stk.cage.Cage):
    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([0, 0.5, 0])),
        stk.cage.UnaligningVertex(1, np.array([0, -0.5, 0])),
        stk.cage.UnaligningVertex(2, np.array([1, 0, 0]), False),
        stk.cage.UnaligningVertex(3, np.array([0, 0, 1]), False),
        stk.cage.UnaligningVertex(4, np.array([-1, 0, 0]), False),
        stk.cage.UnaligningVertex(5, np.array([0, 0, -1]), False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[2]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[3]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[2]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[3]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[5]),
    )


class UnalignedM3L6(stk.cage.Cage):
    _R, _theta = 1, 0

    _vertex_prototypes = (
        stk.cage.UnaligningVertex(
            id=0,
            position=np.array([_R * np.cos(_theta), _R * np.sin(_theta), 0]),
        ),
        stk.cage.UnaligningVertex(
            id=1,
            position=np.array(
                [
                    _R * np.cos(_theta + (4 * np.pi / 3)),
                    _R * np.sin(_theta + (4 * np.pi / 3)),
                    0,
                ]
            ),
        ),
        stk.cage.UnaligningVertex(
            id=2,
            position=np.array(
                [
                    _R * np.cos(_theta + (2 * np.pi / 3)),
                    _R * np.sin(_theta + (2 * np.pi / 3)),
                    0,
                ]
            ),
        ),
        stk.cage.UnaligningVertex(
            id=3,
            position=np.array(
                [
                    _R * np.cos((_theta + np.pi / 4)),
                    _R * np.sin((_theta + np.pi / 4)),
                    0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
        stk.cage.UnaligningVertex(
            id=4,
            position=np.array(
                [
                    _R * np.cos((_theta + 1 * np.pi / 3)),
                    _R * np.sin((_theta + 1 * np.pi / 3)),
                    -0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
        stk.cage.UnaligningVertex(
            id=5,
            position=np.array(
                [
                    _R * np.cos((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                    _R * np.sin((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                    0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
        stk.cage.UnaligningVertex(
            id=6,
            position=np.array(
                [
                    _R * np.cos((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                    _R * np.sin((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                    -0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
        stk.cage.UnaligningVertex(
            id=7,
            position=np.array(
                [
                    _R * np.cos((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                    _R * np.sin((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                    0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
        stk.cage.UnaligningVertex(
            id=8,
            position=np.array(
                [
                    _R * np.cos((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                    _R * np.sin((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                    -0.5,
                ]
            ),
            use_neighbor_placement=False,
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[3]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[6]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[3]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[4]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[8]),
    )


class UnalignedM4L8(stk.cage.Cage):
    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([1, 0, 0])),
        stk.cage.UnaligningVertex(1, np.array([0, 1, 0])),
        stk.cage.UnaligningVertex(2, np.array([-1, 0, 0])),
        stk.cage.UnaligningVertex(3, np.array([0, -1, 0])),
        stk.cage.UnaligningVertex(4, np.array([1, 1, 0.5]), False),
        stk.cage.UnaligningVertex(5, np.array([1, 1, -0.5]), False),
        stk.cage.UnaligningVertex(6, np.array([1, -1, 0.5]), False),
        stk.cage.UnaligningVertex(7, np.array([1, -1, -0.5]), False),
        stk.cage.UnaligningVertex(8, np.array([-1, -1, 0.5]), False),
        stk.cage.UnaligningVertex(9, np.array([-1, -1, -0.5]), False),
        stk.cage.UnaligningVertex(10, np.array([-1, 1, 0.5]), False),
        stk.cage.UnaligningVertex(11, np.array([-1, 1, -0.5]), False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
    )


class UnalignedM6L12(stk.cage.Cage):
    _x = np.sqrt(2)
    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([_x, 0, 0])),
        stk.cage.UnaligningVertex(1, np.array([0, _x, 0])),
        stk.cage.UnaligningVertex(2, np.array([-_x, 0, 0])),
        stk.cage.UnaligningVertex(3, np.array([0, -_x, 0])),
        stk.cage.UnaligningVertex(4, np.array([0, 0, _x])),
        stk.cage.UnaligningVertex(5, np.array([0, 0, -_x])),
        stk.cage.UnaligningVertex(6, np.array([1, 1, 0]), False),
        stk.cage.UnaligningVertex(7, np.array([1, -1, 0]), False),
        stk.cage.UnaligningVertex(8, np.array([1, 0, 1]), False),
        stk.cage.UnaligningVertex(9, np.array([1, 0, -1]), False),
        stk.cage.UnaligningVertex(10, np.array([-1, 1, 0]), False),
        stk.cage.UnaligningVertex(11, np.array([-1, -1, 0]), False),
        stk.cage.UnaligningVertex(12, np.array([-1, 0, 1]), False),
        stk.cage.UnaligningVertex(13, np.array([-1, 0, -1]), False),
        stk.cage.UnaligningVertex(14, np.array([0, 1, 1]), False),
        stk.cage.UnaligningVertex(15, np.array([0, 1, -1]), False),
        stk.cage.UnaligningVertex(16, np.array([0, -1, 1]), False),
        stk.cage.UnaligningVertex(17, np.array([0, -1, -1]), False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[8]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[9]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[6]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[14]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[15]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[12]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[11]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[16]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[17]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[8]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[12]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[14]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[16]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[9]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[13]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[15]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[17]),
    )


class UnalignedM12L24(stk.cage.Cage):
    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([1, 0, 0])),
        stk.cage.UnaligningVertex(1, np.array([-1, 0, 0])),
        stk.cage.UnaligningVertex(2, np.array([0, 1, 0])),
        stk.cage.UnaligningVertex(3, np.array([0, -1, 0])),
        stk.cage.UnaligningVertex(4, np.array([0.5, 0.5, 0.707])),
        stk.cage.UnaligningVertex(5, np.array([0.5, -0.5, 0.707])),
        stk.cage.UnaligningVertex(6, np.array([-0.5, 0.5, 0.707])),
        stk.cage.UnaligningVertex(7, np.array([-0.5, -0.5, 0.707])),
        stk.cage.UnaligningVertex(8, np.array([0.5, 0.5, -0.707])),
        stk.cage.UnaligningVertex(9, np.array([0.5, -0.5, -0.707])),
        stk.cage.UnaligningVertex(10, np.array([-0.5, 0.5, -0.707])),
        stk.cage.UnaligningVertex(11, np.array([-0.5, -0.5, -0.707])),
        stk.cage.UnaligningVertex(12, np.array([0.9, 0.31, 0.31]), False),
        stk.cage.UnaligningVertex(13, np.array([0.9, 0.31, -0.31]), False),
        stk.cage.UnaligningVertex(14, np.array([0.9, -0.31, 0.31]), False),
        stk.cage.UnaligningVertex(15, np.array([0.9, -0.31, -0.31]), False),
        stk.cage.UnaligningVertex(16, np.array([-0.9, 0.31, 0.31]), False),
        stk.cage.UnaligningVertex(17, np.array([-0.9, 0.31, -0.31]), False),
        stk.cage.UnaligningVertex(18, np.array([-0.9, -0.31, 0.31]), False),
        stk.cage.UnaligningVertex(19, np.array([-0.9, -0.31, -0.31]), False),
        stk.cage.UnaligningVertex(20, np.array([0.31, 0.9, 0.31]), False),
        stk.cage.UnaligningVertex(21, np.array([0.31, 0.9, -0.31]), False),
        stk.cage.UnaligningVertex(22, np.array([-0.31, 0.9, 0.31]), False),
        stk.cage.UnaligningVertex(23, np.array([-0.31, 0.9, -0.31]), False),
        stk.cage.UnaligningVertex(24, np.array([0.31, -0.9, 0.31]), False),
        stk.cage.UnaligningVertex(25, np.array([0.31, -0.9, -0.31]), False),
        stk.cage.UnaligningVertex(26, np.array([-0.31, -0.9, 0.31]), False),
        stk.cage.UnaligningVertex(27, np.array([-0.31, -0.9, -0.31]), False),
        stk.cage.UnaligningVertex(28, np.array([0.58, 0, 0.82]), False),
        stk.cage.UnaligningVertex(29, np.array([-0.58, 0, 0.82]), False),
        stk.cage.UnaligningVertex(30, np.array([0, 0.58, 0.82]), False),
        stk.cage.UnaligningVertex(31, np.array([0, -0.58, 0.82]), False),
        stk.cage.UnaligningVertex(32, np.array([0.58, 0, -0.82]), False),
        stk.cage.UnaligningVertex(33, np.array([-0.58, 0, -0.82]), False),
        stk.cage.UnaligningVertex(34, np.array([0, 0.58, -0.82]), False),
        stk.cage.UnaligningVertex(35, np.array([0, -0.58, -0.82]), False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[12]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[13]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[14]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[15]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[16]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[17]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[18]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[19]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[20]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[21]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[22]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[23]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[24]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[25]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[26]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[27]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[28]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[30]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[12]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[20]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[14]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[24]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[28]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[31]),
        stk.Edge(24, _vertex_prototypes[6], _vertex_prototypes[16]),
        stk.Edge(25, _vertex_prototypes[6], _vertex_prototypes[29]),
        stk.Edge(26, _vertex_prototypes[6], _vertex_prototypes[30]),
        stk.Edge(27, _vertex_prototypes[6], _vertex_prototypes[22]),
        stk.Edge(28, _vertex_prototypes[7], _vertex_prototypes[18]),
        stk.Edge(29, _vertex_prototypes[7], _vertex_prototypes[26]),
        stk.Edge(30, _vertex_prototypes[7], _vertex_prototypes[31]),
        stk.Edge(31, _vertex_prototypes[7], _vertex_prototypes[29]),
        stk.Edge(32, _vertex_prototypes[8], _vertex_prototypes[13]),
        stk.Edge(33, _vertex_prototypes[8], _vertex_prototypes[32]),
        stk.Edge(34, _vertex_prototypes[8], _vertex_prototypes[34]),
        stk.Edge(35, _vertex_prototypes[8], _vertex_prototypes[21]),
        stk.Edge(36, _vertex_prototypes[9], _vertex_prototypes[15]),
        stk.Edge(37, _vertex_prototypes[9], _vertex_prototypes[32]),
        stk.Edge(38, _vertex_prototypes[9], _vertex_prototypes[35]),
        stk.Edge(39, _vertex_prototypes[9], _vertex_prototypes[25]),
        stk.Edge(40, _vertex_prototypes[10], _vertex_prototypes[17]),
        stk.Edge(41, _vertex_prototypes[10], _vertex_prototypes[23]),
        stk.Edge(42, _vertex_prototypes[10], _vertex_prototypes[34]),
        stk.Edge(43, _vertex_prototypes[10], _vertex_prototypes[33]),
        stk.Edge(44, _vertex_prototypes[11], _vertex_prototypes[19]),
        stk.Edge(45, _vertex_prototypes[11], _vertex_prototypes[33]),
        stk.Edge(46, _vertex_prototypes[11], _vertex_prototypes[27]),
        stk.Edge(47, _vertex_prototypes[11], _vertex_prototypes[35]),
    )


class TopologyIterator:
    def __init__(
        self,
        tetra_bb,
        converging_bb,
        diverging_bb,
        multiplier,
        stoichiometry,
    ):
        if stoichiometry == (1, 1, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    converging_bb: (2, 3),
                    diverging_bb: (4, 5),
                }
                self._underlying_topology = UnalignedM2L4

            elif multiplier == 2:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5),
                    diverging_bb: (6, 7, 8),
                }
                self._underlying_topology = UnalignedM3L6

            elif multiplier == 3:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    converging_bb: (4, 5, 6, 7),
                    diverging_bb: (8, 9, 10, 11),
                }
                self._underlying_topology = UnalignedM4L8

        if stoichiometry == (4, 2, 3):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                }
                self._underlying_topology = UnalignedM3L6

            elif multiplier == 2:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3, 4, 5),
                    converging_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                    diverging_bb: (14, 15, 16, 17),
                }
                self._underlying_topology = UnalignedM6L12

            elif multiplier == 4:
                self._building_blocks = {
                    tetra_bb: range(0, 12),
                    converging_bb: range(12, 28),
                    diverging_bb: range(28, 36),
                }
                self._underlying_topology = UnalignedM12L24

        self._edges = self._underlying_topology._edge_prototypes
        self._vertices = self._underlying_topology._vertex_prototypes

    def get_num_building_blocks(self):
        return len(self._vertices)

    def get_constructed_molecules(self):
        # If it fails to build, skip.
        try:
            constructed = stk.ConstructedMolecule(
                self._underlying_topology(self._building_blocks)
            )
            yield constructed
        except ValueError:
            pass

        vertex_connections = {}
        for edge in self._edges:
            if edge.get_vertex1_id() not in vertex_connections:
                vertex_connections[edge.get_vertex1_id()] = 0
            vertex_connections[edge.get_vertex1_id()] += 1

            if edge.get_vertex2_id() not in vertex_connections:
                vertex_connections[edge.get_vertex2_id()] = 0
            vertex_connections[edge.get_vertex2_id()] += 1

        possible_pairs = set(
            tuple(sorted((i, j)))
            for i, j in it.product(vertex_connections, repeat=2)
            if vertex_connections[i] != vertex_connections[j]
        )
        type1 = [i for i in vertex_connections if vertex_connections[i] == 4]
        type2 = [i for i in vertex_connections if vertex_connections[i] == 2]

        rng = np.random.default_rng(seed=100)

        combinations_tested = set()

        for step in range(100):
            # Scramble the edges.
            remaining_connections = deepcopy(vertex_connections)
            available_type1s = deepcopy(type1)
            available_type2s = deepcopy(type2)

            new_edges = []
            combination = []
            for ie in range(len(self._edges)):
                try:
                    vertex1 = rng.choice(available_type1s)
                    vertex2 = rng.choice(available_type2s)
                except ValueError:
                    if len(remaining_connections) == 1:
                        vertex1 = list(remaining_connections.keys())[0]
                        vertex2 = list(remaining_connections.keys())[0]

                if tuple(sorted((vertex1, vertex2))) not in possible_pairs:
                    msg = "this should not happen"
                    raise RuntimeError(msg)
                    break

                new_edge = stk.Edge(
                    id=len(new_edges),
                    vertex1=self._vertices[vertex1],
                    vertex2=self._vertices[vertex2],
                )
                new_edges.append(new_edge)

                remaining_connections[vertex1] += -1
                remaining_connections[vertex2] += -1

                remaining_connections = {
                    i: remaining_connections[i]
                    for i in remaining_connections
                    if remaining_connections[i] != 0
                }

                available_type1s = [i for i in type1 if i in remaining_connections]
                available_type2s = [i for i in type2 if i in remaining_connections]
                combination.append(tuple(sorted((vertex1, vertex2))))

            # If you broke early, do not try to build.
            if len(new_edges) != len(self._edges):
                continue

            if tuple(sorted(combination)) in combinations_tested:
                continue

            combinations_tested.add(tuple(sorted(combination)))

            new_topology = deepcopy(self._underlying_topology)
            new_topology._edge_prototypes = new_edges

            # If it fails to build, skip.
            do_not_coordinate = False

            try:
                constructed = stk.ConstructedMolecule(
                    new_topology(self._building_blocks)
                )
                yield constructed
            except ValueError:
                do_not_coordinate = True

            if do_not_coordinate:
                continue

            # Scramble the vertex positions.
            for step2 in range(20):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertices = [
                    i.with_position(coordinates[j])
                    for j, i in enumerate(self._vertices)
                ]
                new_topology = deepcopy(self._underlying_topology)
                new_topology._vertex_prototypes = new_vertices
                try:
                    constructed = stk.ConstructedMolecule(
                        new_topology(self._building_blocks)
                    )
                    yield constructed
                except ValueError:
                    pass

        return
        for step in range(20):
            two_edges = rng.choice(self._edges, 2, replace=False)
            old_e1 = two_edges[0]
            old_e2 = two_edges[1]
            print(old_e1, old_e2)
            new_e1_v1 = self._vertices[old_e1.get_vertex1_id()]
            new_e1_v2 = self._vertices[old_e2.get_vertex2_id()]
            new_e2_v1 = self._vertices[old_e2.get_vertex1_id()]
            new_e2_v2 = self._vertices[old_e1.get_vertex2_id()]
            new_edge1 = stk.Edge(
                id=two_edges[0].get_id(),
                vertex1=new_e1_v1,
                vertex2=new_e1_v2,
            )
            new_edge2 = stk.Edge(
                id=two_edges[1].get_id(),
                vertex1=new_e2_v1,
                vertex2=new_e2_v2,
            )
            print(new_edge1, new_edge2)
            new_edges = [i for i in self._edges]
            new_edges[old_e1.get_id()] = new_edge1
            new_edges[old_e2.get_id()] = new_edge2
            print(new_edges)
            new_topology = deepcopy(self._underlying_topology)
            new_topology._edge_prototypes = new_edges
            yield new_topology(self._building_blocks)


def main():
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "min_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "min_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "min_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "min_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "min_run.db"

    # Define bead libraries.
    present_beads = (cbead_d, abead_d, cbead_c, abead_c, binder_bead, tetra_bead)
    cgexplore.molecular.BeadLibrary(beads=present_beads)

    pairs = {
        "lf_ls1": {
            "forcefield": forcefield_lf_ls1,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
        },
        "lf_ls9": {
            "forcefield": forcefield_lf_ls9,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
        },
        "la_st5": {
            "forcefield": forcefield_la_st5,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
        },
        "la_st52": {
            "forcefield": forcefield_la_st52,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
        },
    }

    for pair in pairs:
        forcefield = pairs[pair]["forcefield"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]
        # Prepare ligands.
        for i, precursor in enumerate((converging, diverging, tetra)):
            name = f"{precursor.get_name()}_f{forcefield.get_identifier()}"
            building_block = cgexplore.utilities.optimise_ligand(
                molecule=precursor.get_building_block(),
                name=name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            building_block.write(str(ligand_dir / f"{name}_optl.mol"))
            if i == 0:
                converging_bb = building_block.clone()
            elif i == 1:
                diverging_bb = building_block.clone()
            elif i == 2:
                tetra_bb = building_block.clone()

        for multiplier in (1, 2, 3):
            # Define a connectivity based on a multiplier.
            iterator = TopologyIterator(
                multiplier=multiplier,
                stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                tetra_bb=tetra_bb,
                converging_bb=converging_bb,
                diverging_bb=diverging_bb,
            )
            for idx, acage in enumerate(iterator.get_constructed_molecules()):
                # Initialise positions based on that connectivity.
                name = f"{pair}_{multiplier}_{idx}"
                acage.write(str(structure_dir / f"{name}_unopt.mol"))

                # Optimise and save.
                logging.info(f"building {name}")

                conformer = optimise_cage(
                    molecule=acage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                )
                if conformer is not None:
                    conformer.molecule.write(str(structure_dir / f"{name}_optc.mol"))

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    iterator=iterator,
                )
            logging.info("figure out how to use multiplier!")

        fig, ax = plt.subplots(figsize=(8, 5))
        energies = {}
        num_components = {}
        multiplier = {}
        mmap = {1: "o", 2: "s", 3: "D"}
        cmap = {1: "tab:blue", 2: "tab:orange"}
        for entry in cgexplore.utilities.AtomliteDatabase(database_path).get_entries():
            if pair != entry.properties["pair"]:
                continue
            energy = entry.properties["energy_per_bb"]
            energies[entry.key] = energy
            if entry.properties["num_components"] < 3:
                num_components[entry.key] = cmap[entry.properties["num_components"]]
            else:
                num_components[entry.key] = "k"

            multiplier[entry.key] = mmap[entry.properties["multiplier"]]

        min_energy = min(energies, key=energies.get)
        ax.plot(
            [energies[i] for i in energies],
            c="k",
            # marker="o",
            # markeredgecolor="k",
            # markersize=8,
            # mfc=num_components.values(),
        )
        ax.scatter(
            [i for i in range(len(energies.values()))],
            [energies[i] for i in energies],
            c=[num_components[i] for i in energies],
            marker=[multiplier[i] for i in energies],
            ec="k",
            s=80,
            zorder=2,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(eb_str(), fontsize=16)
        # ax.set_ylabel("count", fontsize=16)
        ax.set_title(f"{min_energy}: {round(energies[min_energy],2)}", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"min_1_{pair}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
