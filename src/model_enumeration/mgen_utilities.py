"""Utilities module."""

import logging
import pathlib
from collections import abc
from copy import deepcopy

import atomlite
import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import stk
import stko
from openmm import openmm
from scipy import optimize

from model_enumeration.utilities import eb_str


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    database.add_properties(key=name, property_dict={"lowest_e_of_mash": True})

    if "bond_data" not in properties:
        g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
        bond_data = g_measure.calculate_bonds(final_molecule)
        bond_data = {"_".join(i): bond_data[i] for i in bond_data}
        angle_data = g_measure.calculate_angles(final_molecule)
        angle_data = {"_".join(i): angle_data[i] for i in angle_data}
        dihedral_data = g_measure.calculate_torsions(
            molecule=final_molecule,
            absolute=False,
            as_search_string=True,
        )

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
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
            },
        )


def ff_opt_plot(
    target: str,
    key_target: str,
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(5, 5))

    try:
        entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
            key_target
        )
    except RuntimeError:
        return

    if (
        "optimisation_success" not in entry.properties
        or target not in entry.key
    ):
        msg = (
            f"optimisation_success not in entry properties, "
            f"or {target} not in {entry.key}"
        )
        raise RuntimeError(msg)

    xticks = [0]
    xlbls = [eb_str()]
    rmsd = entry.properties["optimisation_rmsd"]
    term_dict = {
        term: entry.properties["optimisation_x"][int(i)]
        for i, term in entry.properties["optimisation_map"].items()
    }

    ffdict = entry.properties["forcefield_dict"]["v_dict"]
    init_term_dict = {term: ffdict["_".join(list(term))] for term in term_dict}

    xticks.extend([i + len(xticks) for i in range(len(term_dict))])
    xlbls.extend(list(term_dict))
    orig = [entry.properties["energy_per_bb"]] + [
        val for i, val in init_term_dict.items()
    ]
    new = [
        entry.properties["optimisation_energy_per_bb"],
    ] + [val for i, val in term_dict.items()]
    p = ax.bar(
        range(len(orig)),
        [((j - i) / i) * 100 for i, j in zip(orig, new, strict=True)],
        color="tab:blue",
        alpha=1,
        width=0.8,
        zorder=-1,
        ec="k",
    )
    ax.bar_label(
        p,
        labels=[round(j, 1) for i, j in zip(orig, new, strict=True)],
        rotation=0,
        label_type="center",
        padding=0,
        fontsize=12,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlbls, fontsize=16, rotation=90)
    ax.axhline(y=0, c="k")
    ax.set_title(f"{key_target}: rmsd= {round(rmsd, 2)}", fontsize=16)
    ax.set_ylim(-100, 100)

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


def get_lowest_energy_entry(
    entries: abc.Sequence[atomlite.Entry],
) -> atomlite.Entry:
    """Get the lowest energy_per_bb entry."""
    sorted_list = sorted(entries, key=lambda x: x[1])
    return sorted_list[0][0].key


def target_optimisation(  # noqa: C901, PLR0913, PLR0915
    database_path: pathlib.Path,
    calculation_dir: pathlib.Path,
    target_key: str,
    definer_dict: dict,
    modifiable_terms: list[str],
    forcefield: cgx.forcefields.ForceField,
) -> None:
    """Optimise the FF terms based on a target."""
    target_entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
        target_key
    )
    if "lowest_e_of_mash" not in target_entry.properties:
        msg = (
            f"{target_entry.key} is not the lowest E of mash. Are you sure"
            " about it?"
        )
        raise RuntimeError(msg)

    num_building_blocks = target_entry.properties["num_bbs"]
    input_cage = stk.BuildingBlock.init_from_rdkit_mol(
        atomlite.json_to_rdkit(target_entry.molecule)
    )
    name = target_key + "_ffopt"
    ff_database_path = calculation_dir / f"{name}_ffopt.db"
    optimised_file = calculation_dir / f"{name}_ffopt.mol"

    if ff_database_path.exists():
        properties = (
            cgx.utilities.AtomliteDatabase(ff_database_path)
            .get_entry(target_key)
            .properties
        )

    else:
        ff_map = dict(enumerate(modifiable_terms))

        initial_ff_params = []
        bounds = []
        for i in modifiable_terms:
            if definer_dict[i][0] in ("bond", "nb"):
                angle = False
                max_change = 0.5
            elif definer_dict[i][0] in ("angle", "pyramid", "tors"):
                angle = True
                max_change = 20
            else:
                raise RuntimeError

            if (
                definer_dict[i][0] == "bond"
                or definer_dict[i][0] == "angle"
                or definer_dict[i][0] == "pyramid"
            ):
                initial_ff_params.append(definer_dict[i][1])
                value = definer_dict[i][1]
            elif definer_dict[i][0] == "tors" or definer_dict[i][0] == "nb":
                initial_ff_params.append(definer_dict[i][2])
                value = definer_dict[i][2]
            else:
                raise RuntimeError

            bounds.append(
                (
                    max((value - max_change, 0)),
                    value + max_change
                    if not angle
                    else (min((value + max_change, 180))),
                )
            )

        def structure_f(
            params: abc.Sequence[float],
        ) -> cgx.molecular.Conformer:
            # Get FF.
            temp_definer_dict = deepcopy(definer_dict)
            for i, value in enumerate(params):
                term = ff_map[i]
                if (
                    temp_definer_dict[term][0] == "bond"
                    or temp_definer_dict[term][0] == "angle"
                    or temp_definer_dict[term][0] == "pyramid"
                ):
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        value,
                        temp_definer_dict[term][2],
                    )
                elif temp_definer_dict[term][0] == "tors":
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        temp_definer_dict[term][1],
                        value,
                        temp_definer_dict[term][3],
                        temp_definer_dict[term][4],
                    )
                elif temp_definer_dict[term][0] == "nb":
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        temp_definer_dict[term][1],
                        value,
                    )
                else:
                    raise RuntimeError

            temp_forcefield = (
                cgx.systems_optimisation.get_forcefield_from_dict(
                    identifier="ffopt",
                    prefix="ffopt",
                    vdw_bond_cutoff=forcefield.get_vdw_bond_cutoff(),
                    present_beads=forcefield.get_present_beads(),
                    definer_dict=temp_definer_dict,
                )
            )

            # Run optimisation.
            return cgx.utilities.run_optimisation(
                assigned_system=temp_forcefield.assign_terms(
                    input_cage,
                    name,
                    calculation_dir,
                ),
                name=name,
                file_suffix="ffopt",
                output_dir=calculation_dir,
                platform=None,
            )

        def f(params: abc.Sequence[float]) -> float:
            if any(i < 0 for i in params):
                return 100
            conformer = structure_f(params)

            # Return Energy.
            return cgx.utilities.get_energy_per_bb(
                energy_decomposition=conformer.energy_decomposition,
                number_building_blocks=num_building_blocks,
            )

        result = optimize.dual_annealing(
            f,
            bounds,
            x0=initial_ff_params,
            minimizer_kwargs={"method": "BFGS", "tol": 0.01},
            maxiter=10,
            maxfun=400,
            rng=np.random.default_rng(2785),
        )
        logging.info("optimisation %s with E: %s", result.success, result.fun)

        min_conformer = structure_f(result.x)
        if (
            cgx.utilities.get_energy_per_bb(
                energy_decomposition=min_conformer.energy_decomposition,
                number_building_blocks=num_building_blocks,
            )
            > result.fun * 1.1
        ):
            raise RuntimeError

        min_conformer.molecule.write(optimised_file)

        properties = {
            "optimisation_success": result.success,
            "optimisation_energy_per_bb": float(result.fun),
            "optimisation_x": [float(i) for i in result.x],
            "optimisation_map": ff_map,
            "optimisation_rmsd": stko.KabschRmsdCalculator(
                input_cage
            ).calculate(min_conformer.molecule),
        }
        cgx.utilities.AtomliteDatabase(ff_database_path).add_molecule(
            molecule=min_conformer.molecule, key=target_key
        )

    # Add properties to the entry.
    cgx.utilities.AtomliteDatabase(database_path).add_properties(
        key=target_key,
        property_dict=properties,
    )
    cgx.utilities.AtomliteDatabase(ff_database_path).add_properties(
        key=target_key,
        property_dict=properties,
    )


def get_stk_topology_code(
    graph_type: str,
) -> tuple[cgx.scram.TopologyCode, np.ndarray]:
    """Get the default stk graph."""
    knowns = {
        "1P2": cgx.topologies.UnalignedM1L2,
        "2P4": stk.cage.M2L4Lantern,
        "3P6": stk.cage.M3L6,
        "4P8": cgx.topologies.CGM4L8,
        "6P12": stk.cage.M6L12Cube,
        "12P24": cgx.topologies.CGM12L24,
    }

    if graph_type not in knowns:
        msg = f"{graph_type} not known"
        raise RuntimeError(msg)

    target = knowns[graph_type]
    vps = target._vertex_prototypes  # noqa: SLF001
    eps = target._edge_prototypes  # noqa: SLF001

    combination = [(i.get_vertex1_id(), i.get_vertex2_id()) for i in eps]
    tc = cgx.scram.TopologyCode(
        vertex_map=combination,
        as_string=cgx.scram.vmap_to_str(combination),
    )

    positions = [i.get_position() for i in vps]

    return tc, positions


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


def get_regraphed_molecule(
    scale: str,
    topology_code: cgx.scram.TopologyCode,
    iterator: cgx.scram.TopologyIterator,
    bb_config: cgx.scram.BuildingBlockConfiguration | None,
) -> stk.ConstructedMolecule:
    """Take a graph that considers all atoms, and get atom positions."""
    logging.info(
        "Caution with this because it currently can change the cis/trans in "
        "m2l4"
    )

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


def optimise_cage(  # noqa: PLR0913, C901
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    potential_names: list[str],
    forcefield: cgx.forcefields.ForceField,
    platform: str | None,
    database_path: pathlib.Path,
) -> cgx.molecular.Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_final.mol"

    database = cgx.utilities.AtomliteDatabase(database_path)
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return cgx.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(  # type:ignore[arg-type]
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = cgx.molecular.Ensemble(
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
                "energy_decomposition": conformer.energy_decomposition,  # type:ignore[dict-item]
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)

    ensemble = cgx.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
        overwrite=True,
    )

    temp_molecule = cgx.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
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
        output_dir=output_dir,
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in cgx.utilities.yield_shifted_models(
        temp_molecule,
        forcefield,
        kicks=(1, 2, 3, 4),
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
            output_dir=output_dir,
            platform=platform,
        )

        ensemble.add_conformer(conformer=conformer, source="shifted")

    # Scan potential name files.
    for potential_name in potential_names:
        potential_file = output_dir / f"{potential_name}_final.mol"

        if not potential_file.exists():
            continue

        test_molecule = stk.BuildingBlock.init_from_file(potential_file)

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
            file_suffix="ns",
            output_dir=output_dir,
            platform=platform,
        )

        ensemble.add_conformer(conformer=conformer, source="ns")

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
        soft_md_data = soft_md_trajectory.get_data()  # type:ignore[union-attr]
        # Check that the trajectory is as long as it should be.
        if len(soft_md_data) != num_steps / traj_freq:
            failed_md = True

        # Go through each conformer from soft MD.
        # Optimise them all.
        for md_conformer in soft_md_trajectory.yield_conformers():  # type:ignore[union-attr]
            if failed_md:
                continue
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
                output_dir=output_dir,
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="smd")

    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy: float = min_energy_conformer.energy_decomposition[
        "total energy"
    ][0]
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
            "energy_decomposition": min_energy_conformer.energy_decomposition,  # type:ignore[dict-item]
            "source": min_energy_conformer.source,
            "optimised": True,
        },
    )
    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


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


def passes_graph_bb_iso(
    topology_code: cgx.scram.TopologyCode,
    bb_config: cgx.scram.BuildingBlockConfiguration,
    run_topology_codes: abc.Sequence[
        tuple[cgx.scram.TopologyCode, cgx.scram.BuildingBlockConfiguration]
    ],
) -> bool:
    """Check if a graph and bb config pases isomorphism check."""
    # Testing bb-config aware graph check.
    # Convert TopologyCode to a graph.
    current_graph = get_bb_topology_code_graph(
        topology_code=topology_code,
        bb_config=bb_config,
    )

    # Check that graph for isomorphism with others graphs.
    passed_iso = True
    for tc, bc in run_topology_codes:
        test_graph = get_bb_topology_code_graph(topology_code=tc, bb_config=bc)

        if rx.is_isomorphic(
            current_graph,
            test_graph,
            node_matcher=lambda x, y: x.split("-")[1] == y.split("-")[1],
        ):
            passed_iso = False
            break
    return passed_iso


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


ligand_name_conversion = {
    "lf": "t1",
    "ls2": "r6",
    "ls3": "r7",
    "ls4": "r8",
    "ls5": "r9",
    "ls7": "r10",
    "ls8": "r11",
    "ls10": "r12",
    # Diverging-tarzia_2024.
    "l1": "r3",
    "l2": "r4",
    "l3": "r5",
    # Converging-tarzia_2024.
    "la": "m2",
    "lb": "m3",
    "lc": "m4",
    "ld": "m5",
    # Experimental.
    "e10": "t2",
    "e11": "t3",
    "e12": "t4",
    "e13": "t5",
    "e14": "t6",
    "e16": "r1",
    "e17": "m1",
    "e18": "r2",
}


class StericTwoC1Arm(cgx.molecular.Precursor):
    """A `TwoC1Arm` Precursor."""

    def __init__(
        self,
        bead: cgx.molecular.CgBead,
        abead1: cgx.molecular.CgBead,
        steric_bead: cgx.molecular.CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = (
            f"s2C1{bead.bead_type}{abead1.bead_type}{steric_bead.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            steric_bead.bead_type: steric_bead,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{bead.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=f"[{abead1.element_string}][{bead.element_string}]"
            f"([{steric_bead.element_string}])[{abead1.element_string}]",
            functional_groups=new_fgs,
            position_matrix=np.array(
                [[-3, 0, 0], [0, 0, 0], [0, 1, 0], [3, 0, 0]]
            ),
        )


# Small ligands.
cbead_d = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
abead_d = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)

# Sixbead small ligands.
c2bead_d = cgx.molecular.CgBead(
    element_string="Zn",
    bead_class="z",
    bead_type="z",
    coordination=2,
)
a2bead_d = cgx.molecular.CgBead(
    element_string="Rh",
    bead_class="r",
    bead_type="r",
    coordination=2,
)
e2bead_d = cgx.molecular.CgBead(
    element_string="Mn",
    bead_class="f",
    bead_type="f",
    coordination=2,
)

# Large ligands.
cbead_c = cgx.molecular.CgBead(
    element_string="Ni",
    bead_class="d",
    bead_type="d",
    coordination=2,
)
abead_c = cgx.molecular.CgBead(
    element_string="Fe",
    bead_class="e",
    bead_type="e",
    coordination=2,
)
ebead_c = cgx.molecular.CgBead(
    element_string="Ga",
    bead_class="g",
    bead_type="g",
    coordination=2,
)

# Constant.
binder_bead = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
tetra_bead = cgx.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)
steric_bead = cgx.molecular.CgBead(
    element_string="S",
    bead_class="s",
    bead_type="s",
    coordination=1,
)
trigonal_bead = cgx.molecular.CgBead(
    element_string="C",
    bead_class="n",
    bead_type="n",
    coordination=3,
)

constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.0, 1e5),
    "cs": ("bond", 1.0, 1e5),
    # Angles.
    "bmb": ("pyramid", 90, 1e2),
    "mba": ("angle", 180, 1e2),
    "mbg": ("angle", 180, 1e2),
    "mbf": ("angle", 180, 1e2),
    "aca": ("angle", 180, 1e2),
    "acs": ("angle", 90, 1e2),
    # Torsions.
    "bacs": ("tors", "0123", 180, 50, 1),
    "bacab": ("tors", "0134", 180, 50, 1),
    "edde": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "mbge": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "rzzr": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    "mbfr": ("tors", "0123", 180.0, 50.0, 1),  # type: ignore[assignment]
    # Nonbondeds.
    "m": ("nb", 10.0, 1.0),
    "d": ("nb", 10.0, 1.0),
    "e": ("nb", 10.0, 1.0),
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "g": ("nb", 10.0, 1.0),
    "s": ("nb", 10.0, 1.0),
    "z": ("nb", 10.0, 1.0),
    "f": ("nb", 10.0, 1.0),
    "r": ("nb", 10.0, 1.0),
}


def precursors_to_forcefield(  # noqa: C901, PLR0913, PLR0915
    pair: str,
    large: cgx.molecular.Precursor,
    small: cgx.molecular.Precursor,
    large_meas: dict[str, float],
    small_meas: dict[str, float],
    constant_definer_dict: dict[str, tuple],
    vdw_bond_cutoff: int | None = None,
) -> cgx.forcefields.ForceField:
    """Get a forcefield from precursor definitions."""
    # Define bead libraries.
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
    cgx.molecular.BeadLibrary(present_beads)

    definer_dict = deepcopy(constant_definer_dict)

    cg_scale = 2

    if isinstance(large, cgx.molecular.SixBead):
        beads = large.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        definer_dict["dd"] = ("bond", large_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", large_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", large_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", large_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", large_meas["dde"], 1e2)
        definer_dict["egb"] = ("angle", large_meas["egb"], 1e2)
        definer_dict["deg"] = ("angle", large_meas["deg"], 1e2)
    elif isinstance(large, cgx.molecular.TwoC1Arm):
        beads = large.get_bead_set()
        if "e" not in beads or "d" not in beads:
            raise RuntimeError
        definer_dict["be"] = ("bond", large_meas["ba"] / cg_scale, 1e5)
        ac = large_meas["aa"] / 2
        definer_dict["ed"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bed"] = ("angle", large_meas["bac"], 1e2)
    else:
        raise NotImplementedError

    if isinstance(small, cgx.molecular.TwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads:
            raise RuntimeError
        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)

    elif isinstance(small, StericTwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads or "s" not in beads:
            raise RuntimeError

        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)
        definer_dict["s"] = ("nb", 10.0, small_meas["s"])

    elif isinstance(small, cgx.molecular.SixBead):
        beads = small.get_bead_set()
        if "z" not in beads or "r" not in beads or "f" not in beads:
            raise RuntimeError
        definer_dict["zz"] = ("bond", small_meas["dd"] / cg_scale, 1e5)
        definer_dict["zr"] = ("bond", small_meas["de"] / cg_scale, 1e5)
        definer_dict["rf"] = ("bond", small_meas["eg"] / cg_scale, 1e5)
        definer_dict["fb"] = ("bond", small_meas["gb"] / cg_scale, 1e5)
        definer_dict["zzr"] = ("angle", small_meas["dde"], 1e2)
        definer_dict["rfb"] = ("angle", small_meas["egb"], 1e2)
        definer_dict["zrf"] = ("angle", small_meas["deg"], 1e2)

    else:
        raise NotImplementedError

    return cgx.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=vdw_bond_cutoff,
        present_beads=present_beads,
        definer_dict=precursors_to_definer_dict(
            large=large,
            small=small,
            large_meas=large_meas,
            small_meas=small_meas,
            constant_definer_dict=constant_definer_dict,
        ),
    )


def precursors_to_definer_dict(  # noqa: C901, PLR0915
    large: cgx.molecular.Precursor,
    small: cgx.molecular.Precursor,
    large_meas: dict[str, float],
    small_meas: dict[str, float],
    constant_definer_dict: dict[str, tuple],
) -> dict[str, tuple]:
    """Get a forcefield from precursor definitions."""
    # Define bead libraries.
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
    cgx.molecular.BeadLibrary(present_beads)

    definer_dict = deepcopy(constant_definer_dict)

    cg_scale = 2

    if isinstance(large, cgx.molecular.SixBead):
        beads = large.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        definer_dict["dd"] = ("bond", large_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", large_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", large_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", large_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", large_meas["dde"], 1e2)
        definer_dict["egb"] = ("angle", large_meas["egb"], 1e2)
        definer_dict["deg"] = ("angle", large_meas["deg"], 1e2)
    elif isinstance(large, cgx.molecular.TwoC1Arm):
        beads = large.get_bead_set()
        if "e" not in beads or "d" not in beads:
            raise RuntimeError
        definer_dict["be"] = ("bond", large_meas["ba"] / cg_scale, 1e5)
        ac = large_meas["aa"] / 2
        definer_dict["ed"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bed"] = ("angle", large_meas["bac"], 1e2)
    else:
        raise NotImplementedError

    if isinstance(small, cgx.molecular.TwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads:
            raise RuntimeError
        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)

    elif isinstance(small, StericTwoC1Arm):
        beads = small.get_bead_set()
        if "a" not in beads or "c" not in beads or "s" not in beads:
            raise RuntimeError

        definer_dict["ba"] = ("bond", small_meas["ba"] / cg_scale, 1e5)
        ac = small_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", small_meas["bac"], 1e2)
        definer_dict["s"] = ("nb", 10.0, small_meas["s"])

    elif isinstance(small, cgx.molecular.SixBead):
        beads = small.get_bead_set()
        if "z" not in beads or "r" not in beads or "f" not in beads:
            raise RuntimeError
        definer_dict["zz"] = ("bond", small_meas["dd"] / cg_scale, 1e5)
        definer_dict["zr"] = ("bond", small_meas["de"] / cg_scale, 1e5)
        definer_dict["rf"] = ("bond", small_meas["eg"] / cg_scale, 1e5)
        definer_dict["fb"] = ("bond", small_meas["gb"] / cg_scale, 1e5)
        definer_dict["zzr"] = ("angle", small_meas["dde"], 1e2)
        definer_dict["rfb"] = ("angle", small_meas["egb"], 1e2)
        definer_dict["zrf"] = ("angle", small_meas["deg"], 1e2)

    else:
        raise NotImplementedError

    return definer_dict
