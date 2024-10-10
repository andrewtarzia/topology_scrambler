"""Utilities module."""

import logging
import pathlib

import cgexplore
import networkx as nx
import numpy as np
import stk
import stko
from openmm import OpenMMException, openmm

from scram._internal.topologies.custom_topology import CustomTopology
from scram._internal.topologies.enumeration import (
    IHomolepticTopologyIterator,
    TopologyIterator,
)
from scram._internal.topologies.topology_code import TopologyCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def prepare_building_block(
    precursor: cgexplore.molecular.Precursor,
    forcefield: cgexplore.forcefields.ForceField,
    calculation_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
) -> stk.BuildingBlock:
    """Prepare a building block."""
    name = f"{precursor.get_name()}_f{forcefield.get_identifier()}"
    building_block = cgexplore.utilities.optimise_ligand(
        molecule=precursor.get_building_block(),
        name=name,
        output_dir=calculation_dir,
        forcefield=forcefield,
        platform=None,
    )
    building_block.write(str(ligand_dir / f"{name}_optl.mol"))
    return building_block.clone()


def graph_optimise_cage(  # noqa: PLR0913
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    platform: str | None,
    database_path: pathlib.Path,
) -> cgexplore.molecular.Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_wipfinal.mol"

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
    if fina_mol_file.exists():
        ensemble = cgexplore.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=output_dir / f"{name}_base.mol",
            conformer_xyz=output_dir / f"{name}_ensemble.xyz",
            data_json=output_dir / f"{name}_ensemble.json",
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
    if (output_dir / f"{name}_ensemblewip.xyz").exists():
        (output_dir / f"{name}_ensemblewip.xyz").unlink()
    ensemble = cgexplore.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemblewip.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
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
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
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
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    stko_graph = stko.Network.init_from_molecule(conformer.molecule)
    for i, nx_positions in enumerate(
        (
            nx.spectral_layout(stko_graph.get_graph(), dim=3),
            nx.get_node_attributes(
                nx.random_geometric_graph(
                    n=conformer.molecule.get_num_atoms(), radius=1, dim=3
                ),
                "pos",
            ),
            nx.spring_layout(stko_graph.get_graph(), dim=3),
            nx.kamada_kawai_layout(stko_graph.get_graph(), dim=3),
        )
    ):
        try:
            # We allow these to independantly failed because the nx graphs can
            # be ridiculous.
            pos_mat = np.array([nx_positions[i] for i in nx_positions])
            if pos_mat.shape[1] != 3:  # noqa: PLR2004
                msg = "built a non 3D graph"
                raise RuntimeError(msg)

            test_molecule = conformer.molecule.with_position_matrix(
                pos_mat * 10
            )
            conformer = cgexplore.utilities.run_optimisation(
                assigned_system=forcefield.assign_terms(
                    test_molecule, name, output_dir
                ),
                name=name,
                file_suffix="nopt",
                output_dir=output_dir,
                platform=platform,
            )

            ensemble.add_conformer(conformer=conformer, source=f"nx{i}")
        except OpenMMException:
            logging.info("failed graph opt of %s", name)

    # Try with graph positions.
    rng = np.random.default_rng(seed=100)
    for attempt in range(10):
        pos_mat = rng.random(size=(conformer.molecule.get_num_atoms(), 3))
        test_molecule = conformer.molecule.with_position_matrix(pos_mat * 10)
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
            file_suffix=f"ropt{attempt}",
            output_dir=output_dir,
            platform=platform,
        )

        ensemble.add_conformer(conformer=conformer, source="shifted")

    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        "%s from %s with energy: %s kJ.mol-1",
        min_energy_conformerid,
        min_energy_conformer.source,
        round(min_energy, 2),
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


def optimise_cage(  # noqa: PLR0913, C901
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    platform: str | None,
    database_path: pathlib.Path,
) -> cgexplore.molecular.Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_final.mol"

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
    if fina_mol_file.exists():
        ensemble = cgexplore.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=output_dir / f"{name}_base.mol",
            conformer_xyz=output_dir / f"{name}_ensemble.xyz",
            data_json=output_dir / f"{name}_ensemble.json",
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
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
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
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
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
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

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
    failed_md = False
    if soft_md_trajectory is None:
        failed_md = True

    if not failed_md:
        soft_md_data = soft_md_trajectory.get_data()
        # Check that the trajectory is as long as it should be.
        if len(soft_md_data) != num_steps / traj_freq:
            failed_md = True

        # Go through each conformer from soft MD.
        # Optimise them all.
        for md_conformer in soft_md_trajectory.yield_conformers():
            if failed_md:
                continue
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
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="smd")

    # Add neighbours to systematic scan.
    if "scan" in name:
        si, sj = name.split("_")[1].split("-")

        potential_names = [
            f"scan_{int(si)-1}-{int(sj)-1}",
            f"scan_{int(si)-1}-{int(sj)}",
            f"scan_{int(si)}-{int(sj)-1}",
        ]

        for potential_name in potential_names:
            potential_file = output_dir / f"{potential_name}_final.mol"
            if not potential_file.exists():
                continue
            test_molecule = temp_molecule.with_structure_from_file(
                potential_file
            )
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
                file_suffix="ns",
                output_dir=output_dir,
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="ns")

    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        "%s from %s with energy: %s kJ.mol-1",
        min_energy_conformerid,
        min_energy_conformer.source,
        round(min_energy, 2),
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


def try_except_construction(
    iterator: TopologyIterator | IHomolepticTopologyIterator,
    topology_code: TopologyCode,
    vertex_positions: dict[int, np.ndarray] | None = None,
) -> stk.ConstructedMolecule:
    """Try construction with alignment, then without."""
    try:
        # Try with aligning vertices.
        constructed_molecule = stk.ConstructedMolecule(
            CustomTopology(
                building_blocks=iterator.building_blocks,
                vertex_prototypes=iterator.get_vertex_prototypes(
                    unaligning=False
                ),
                # Convert to edge prototypes.
                edge_prototypes=topology_code.edges_from_connection(
                    iterator.get_vertex_prototypes(unaligning=False)
                ),
                vertex_alignments=None,
                vertex_positions=vertex_positions,
                scale_multiplier=iterator.scale_multiplier,
            )
        )

    except ValueError:
        # Try with unaligning.
        constructed_molecule = stk.ConstructedMolecule(
            CustomTopology(
                building_blocks=iterator.building_blocks,
                vertex_prototypes=iterator.get_vertex_prototypes(
                    unaligning=True
                ),
                # Convert to edge prototypes.
                edge_prototypes=topology_code.edges_from_connection(
                    iterator.get_vertex_prototypes(unaligning=True)
                ),
                vertex_alignments=None,
                vertex_positions=vertex_positions,
                scale_multiplier=iterator.scale_multiplier,
            )
        )
    return constructed_molecule
