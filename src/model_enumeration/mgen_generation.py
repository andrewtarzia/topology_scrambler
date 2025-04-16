"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import shutil
from collections import abc, defaultdict
from copy import deepcopy

import atomlite
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import stk
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from scipy import optimize

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
    precursors_to_definer_dict,
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


def get_lowest_energy_entry(
    entries: abc.Sequence[atomlite.Entry],
) -> atomlite.Entry:
    """Get the lowest energy_per_bb entry."""
    sorted_list = sorted(entries, key=lambda x: x[1])
    return sorted_list[0][0].key


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


def make_topt_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    pairs: dict[str, dict[str, tuple | int]],
    ffopt_targets: dict[str, str],
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

    for pair in pairs:
        try:
            mix_target = ffopt_targets[pair]
        except (KeyError, IndexError):
            try:
                mix_target = get_lowest_energy_entry(
                    [
                        (i, i.properties["energy_per_bb"])
                        for i in cgx.utilities.AtomliteDatabase(
                            database_path
                        ).get_entries()
                        if pair in i.key and "lowest_e_of_mash" in i.properties
                    ]
                )
            except IndexError:
                continue

        try:
            entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
                mix_target
            )
        except RuntimeError:
            continue

        if "optimisation_energy_per_bb" not in entry.properties:
            continue
        d_energy = (
            entry.properties["optimisation_energy_per_bb"]
            - entry.properties["energy_per_bb"]
        )
        if d_energy > 0:
            logging.info("%s has energy change > 0", entry.key)

        term_dict = {
            term: entry.properties["optimisation_x"][int(i)]
            for i, term in entry.properties["optimisation_map"].items()
        }

        ffdict = entry.properties["forcefield_dict"]["v_dict"]
        init_term_dict = {
            term: ffdict["_".join(list(term))] for term in term_dict
        }

        orig = [val for i, val in init_term_dict.items()]
        new = [val for i, val in term_dict.items()]
        ax.scatter(
            sum(
                [
                    abs((j - i) / i) * 100
                    for i, j in zip(orig, new, strict=True)
                ]
            ),
            d_energy,
            c=entry.properties["energy_per_bb"],
            alpha=1,
            ec="k",
            s=120,
            vmin=0,
            vmax=1,
            cmap="Blues_r",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("sum relative change in FF terms", fontsize=16)
    ax.set_ylabel(rf"$\Delta$ {eb_str()}", fontsize=16)
    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"original {eb_str()}", fontsize=16)

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


def make_topt_s6_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    mixtures: dict[str, dict[str, tuple | int]],
) -> dict:
    """Visualise energies."""
    modifiable = ["nb", "bnb", "bac", "ba", "ac"]
    fig, axs = plt.subplots(
        ncols=len(modifiable),
        sharey=True,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    for mix in mixtures:
        try:
            mix_target = mixtures[mix]["target"]

        except KeyError:
            continue

        entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
            mix_target
        )

        if entry.properties["multiplier"] != 1:
            continue

        if "optimisation_energy_per_bb" not in entry.properties:
            raise RuntimeError

        term_dict = {
            term: entry.properties["optimisation_x"][int(i)]
            for i, term in entry.properties["optimisation_map"].items()
        }

        ffdict = entry.properties["forcefield_dict"]["v_dict"]
        init_term_dict = {
            term: ffdict["_".join(list(term))] for term in term_dict
        }

        orig = [val for i, val in init_term_dict.items()]
        new = [val for i, val in term_dict.items()]
        for i, ax in enumerate(flat_axs):
            ax.scatter(
                new[i],
                entry.properties["optimisation_energy_per_bb"],
                alpha=1,
                ec="k",
                s=80,
            )
            ax.plot(
                (orig[i], new[i]),
                (
                    entry.properties["optimisation_energy_per_bb"],
                    entry.properties["optimisation_energy_per_bb"],
                ),
                c="k",
                alpha=1,
                lw=1,
                zorder=-2,
                marker="s",
                markersize=3,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel(modifiable[i], fontsize=16)
            d = new[i] - orig[i]
            ax.set_title(rf"avg. $|\Delta|$={round(d, 2)}", fontsize=16)
            ax.set_yscale("log")
            if i == 0:
                ax.set_ylabel(f"opt. {eb_str()}", fontsize=16)

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
    width_height: tuple[float, float] = (7, 10),
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=width_height)
    energies = {}

    xs = []

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        if multi not in xs:
            xs.append(multi)

        pair = tuple(entry.properties["pair"].split("_"))
        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

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


def study_3_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    cmap = {
        "cs3l1_cs3l1p": "tab:blue",
        "cs3l1_cs3l6p": "tab:orange",
        "cs3l2_cs3l1p": "tab:green",
        "cs3l2_cs3l6p": "tab:red",
    }

    xs = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        pair = entry.properties["pair"]
        string = entry.key.split("_")
        string = (
            string[0].split("cs3")[1]
            + "-"
            + string[1].split("cs3")[1]
            + "-t"
            + string[3]
            + "-"
            + string[5]
        )
        if ey < isomer_energy():
            xs[string] = len(xs)

            p = ax.bar(
                xs[string],
                ey,
                fc=cmap[pair],
                alpha=1.0,
                ec="k",
            )
            ax.bar_label(
                p,
                labels=[round(ey, 2)],
                rotation=90,
                label_type="edge",
                padding=8,
                fontsize=12,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylim(0, None)

    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=90)
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


def study_3_plot_5(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    cmap = {
        "cs3l1_cs3l1p": "tab:blue",
        "cs3l1_cs3l6p": "tab:orange",
        "cs3l2_cs3l1p": "tab:green",
        "cs3l2_cs3l6p": "tab:red",
    }
    xmap = {
        "cs3l1_cs3l1p": -0.3,
        "cs3l1_cs3l6p": -0.1,
        "cs3l2_cs3l1p": 0.1,
        "cs3l2_cs3l6p": 0.3,
    }

    xs = {"isomer A": 0, "isomer B": 1}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        pair = entry.properties["pair"]
        tidx = entry.properties["topology_idx"]
        bidx = entry.properties["bb_config_idx"]
        if (tidx, bidx) == (2, 89):
            x = xs["isomer A"] + xmap[pair]
        elif (tidx, bidx) == (2, 112):
            x = xs["isomer B"] + xmap[pair]
        else:
            continue
        string = pair.split("_")
        string = string[0].split("cs3")[1] + "-" + string[1].split("cs3")[1]

        p = ax.bar(
            x,
            ey,
            width=0.1,
            fc=cmap[pair],
            alpha=1.0,
            ec="k",
        )
        ax.bar_label(
            p,
            labels=[f"{string}: {round(ey, 2)}"],
            rotation=90,
            label_type="edge",
            padding=8,
            fontsize=12,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylim(0, 1.2)

    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=0)
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


def study_3_plot_2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(16, 5))

    counts = {}
    counts_low = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ey = entry.properties["energy_per_bb"]
        m = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if pair not in counts:
            counts[pair] = 0
            counts_low[pair] = 0

        if m == 6:  # noqa: PLR2004
            if ey < 1:
                counts[pair] += 1
            if ey < isomer_energy():
                counts_low[pair] += 1

        if pair != "cs3l1_cs3l1p":
            continue

        ax.scatter(
            m,
            ey,
            c="tab:blue",
            s=120,
            ec="k",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks([2, 3, 4, 5, 6])

    bar = ax2.bar(
        range(len(counts_low)),
        list(counts_low.values()),
        align="center",
        fc="tab:orange",
    )
    ax2.bar_label(bar, fmt="%.f", fontsize=16)
    ax2.bar(
        range(len(counts)),
        list(counts.values()),
        align="center",
        fc="none",
        ec="k",
    )

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_ylabel("count", fontsize=16)
    ax2.set_xticks(range(len(counts)), list(counts.keys()))

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

    tmap = {
        "mix1": "tab:blue",
        "mix2": "tab:orange",
        "mix3": "tab:green",
        "mix4": "tab:red",
        "mix5": "tab:purple",
    }
    targets = {
        i: {"0": float("inf"), "1": float("inf"), "2": float("inf")}
        for i in tmap
    }

    possible_pos = [-0.3, -0.15, 0, 0.15, 0.3]

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
            entry.properties["sidx"]
            + possible_pos[int(entry.properties["mix"][-1]) - 1],
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

    for xi, (pair, edict) in enumerate(targets.items()):
        ax.bar(
            [
                int(i) + possible_pos[xi]
                for i, ed in edict.items()
                if ed != float("inf")
            ],
            [ed for ed in edict.values() if ed != float("inf")],
            alpha=1.0,
            width=0.1,
            fc=tmap[pair],
            ec="k",
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


def study_5_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    tmap = {
        "mix1": "tab:blue",
        "mix2": "tab:orange",
        "mix3": "tab:green",
        "mix4": "tab:red",
        "mix5": "tab:purple",
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if (
            entry.properties["sidx"] == 0
            and entry.properties["topology_idx"] == 0
            and entry.properties["bb_config_idx"] == 2  # noqa: PLR2004
        ):
            x = int(entry.properties["mix"][-1]) - 1
            y = entry.properties["energy_per_bb"]

            ax.bar(
                x,
                y,
                alpha=1.0,
                width=0.8,
                fc=tmap[entry.properties["mix"]],
                ec="k",
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(tmap)))
    ax.set_xticklabels(list(tmap), fontsize=16)

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
    fig, (ax) = plt.subplots(ncols=1, figsize=(16, 5))

    multis = {
        1: (multi_cmap["1"], -0.2),
        2: (multi_cmap["2"], 0.0),
        3: (multi_cmap["3"], 0.2),
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
    ax.set_xticklabels(list(xs), fontsize=16, rotation=90)

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


def study_6_plot_2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(16, 5))

    multis = {
        1: (multi_cmap["1"], -0.2),
        2: (multi_cmap["2"], 0.0),
        3: (multi_cmap["3"], 0.2),
    }

    xs = {}
    lbls = set()
    mix_mins = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if entry.properties["mix"] not in xs:
            xs[entry.properties["mix"]] = len(xs)
            mix_mins[entry.properties["mix"]] = {
                i: (float("inf"), None) for i in multis
            }

        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]

        if y < mix_mins[entry.properties["mix"]][multi][0]:
            mix_mins[entry.properties["mix"]][multi] = (y, entry.key)

    for mix, mdict in mix_mins.items():
        for multi, (y, key) in mdict.items():
            lbl = multi
            if key is None:
                continue
            p = ax.bar(
                xs[mix] + multis[multi][1],
                y,
                fc=multis[multi][0],
                width=0.2,
                ec="k",
                label=lbl if lbl not in lbls else None,
            )
            padding = 0 if multi == 1 else 8
            ltype = "center" if multi == 1 else "edge"
            string = key.split("_")
            string = "t: " + string[2]
            ax.bar_label(
                p,
                labels=[string],
                rotation=90,
                label_type=ltype,
                padding=padding,
                fontsize=12,
            )
            lbls.add(lbl)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.legend(fontsize=16)
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16, rotation=90)

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


def study_6_plot_3(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 2))

    l_positions = {
        "cs6l1": 3.8,
        "cs6l1b": 3.8,
        "cs6l2": 5.7,
        "cs6l2b": 5.7,
        "cs6l5": 8.0,
        "cs6l5b": 8.0,
        "cs6l6": 9.9,
        "cs6l6b": 9.9,
        "cs6l9": 14.2,
        "cs6l9b": 14.2,
    }
    x_positions = {
        "cs6l1": -0.2,
        "cs6l1b": 0.2,
        "cs6l2": 1.8,
        "cs6l2b": 2.2,
        "cs6l5": 3.8,
        "cs6l5b": 4.2,
        "cs6l6": 5.8,
        "cs6l6b": 6.2,
        "cs6l9": 7.8,
        "cs6l9b": 8.2,
    }
    y_positions = {"cs6zr1": 0.1, "cs6zr2": 0.0}

    # create the new map
    cmap = plt.cm.Blues_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        mix, multi, idx, midx = entry.key.split("_")
        if multi == "1":
            yp = 0
            if idx != "0":
                continue
        elif multi == "2":
            yp = 0.3
            if idx != "4":
                continue
        if entry.properties["di"] not in x_positions:
            continue
        x = x_positions[entry.properties["di"]]
        y = y_positions[entry.properties["tri"]] + yp
        c = entry.properties["energy_per_bb"]

        ax.scatter(
            x,
            y,
            c=c,
            marker="s",
            s=200,
            edgecolor="k",
            cmap=cmap,
            norm=norm,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([x_positions[i] + 0.2 for i in x_positions if "b" not in i])
    ax.set_xticklabels(
        [l_positions[i] for i in x_positions if "b" not in i], fontsize=16
    )
    ax.set_yticks(
        [y_positions[i] for i in y_positions]
        + [y_positions[i] + 0.3 for i in y_positions]
    )
    ax.set_yticklabels(list(y_positions) * 2, fontsize=16)
    ax.set_ylim(-0.1, 0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(eb_str(), fontsize=16)

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


def study_6_plot_4(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if "cc3" in entry.key:
            continue

        e = entry.properties["energy_per_bb"]
        opte = entry.properties.get("optimisation_energy_per_bb", None)
        m = entry.properties["multiplier"]

        if opte is not None:
            ax.scatter(
                e,
                opte,
                c=multi_cmap[str(m)],
                ec="k",
                s=120,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"input {eb_str()}", fontsize=16)
    ax.set_ylabel(f"opt-ff {eb_str()}", fontsize=16)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.plot((0, 20), (0, 20), c="k", ls="--", zorder=-2)
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


def make_summary_plot2(  # noqa: C901, PLR0912
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    structure_dir: pathlib.Path,
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

        pair = tuple(entry.properties["pair"].split("_"))
        if len(pair) > 3:  # noqa: PLR2004
            msg = f"is {pair} right? ({entry.properties['pair']})"
            raise RuntimeError(msg)
        if len(pair) == 3:  # noqa: PLR2004
            pair = (pair[0], pair[1] + "_" + pair[2])

        x = [i[0] for i in pairs].index(pair)
        x_count[multi][x] += 1
        energy = entry.properties["energy_per_bb"]

        if energy < 1:
            stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            ).write(structure_dir / f"{entry.key}_optc.mol")

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
        "--opt_ff",
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
        "--starships",
        action="store_true",
        help="set to run and or visualise starshsips (w/o --run, only viz)",
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


def binder_vector_angles_plot_unsymm(  # noqa: C901, PLR0915
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
            np.mean(xdict[0]),
            np.mean(ydict[0]),
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


def case_study_1_2(run: bool, opt_ff: bool) -> None:  # noqa: C901, PLR0912, PLR0915
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
    ffopt_modifiable = {
        "la_l1": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lb_l1": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lc_l1": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "ld_l1": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "la_l2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lb_l2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lc_l2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "ld_l2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "la_l3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lb_l3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lc_l3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "ld_l3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e10_e16": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e13_e12": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e17_e16": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e17_e10": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e10_e11": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e14_e16": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e14_e18": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e10_e18": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "e10_e12": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e14_e11": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e14_e12": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e13_e11": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e14_e13": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "e12_e11": ["deg", "dd", "de", "dde", "zrf", "zz", "zr", "zzr"],
        "lf_l2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lf_ls2": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lf_ls3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lf_l3": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
        "lf_ls10": ["deg", "dd", "de", "dde", "ba", "ac", "bac"],
    }
    ffopt_targets = {
        "lf_l2": "lf_l2_3_2_1_b1",
        "lf_ls2": "lf_ls2_3_2_0_b1",
        "lf_ls3": "lf_ls3_3_-1_0_b5",
        "lf_l3": "lf_l3_4_9_4_b3",
        "lf_ls10": "lf_ls10_4_9_4_b3",
    }

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
        if run:
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

                # Use known topology codes.
                stk_topology_code, stk_positions = get_stk_topology_code(
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
                )
                vertex_positions = {
                    nidx: np.array(stk_positions[nidx]) * 10
                    for nidx in stk_topology_code.get_nx_graph().nodes
                }
                sidx = -1
                midx = 0
                run_topology_codes = []
                for bb_config in possible_bbdicts:
                    name = (
                        f"{pair}_{multiplier}_{sidx}_{midx}_b{bb_config.idx}"
                    )

                    if not passes_graph_bb_iso(
                        topology_code=stk_topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append((stk_topology_code, bb_config))
                    try:
                        constructed_molecule = stk.ConstructedMolecule(
                            cgx.topologies.CustomTopology(  # type: ignore[arg-type]
                                building_blocks=bb_config.get_building_block_dictionary(),
                                vertex_prototypes=iterator.get_vertex_prototypes(
                                    unaligning=False
                                ),
                                # Convert to edge prototypes.
                                edge_prototypes=stk_topology_code.edges_from_connection(
                                    iterator.get_vertex_prototypes(
                                        unaligning=False
                                    )
                                ),
                                vertex_alignments=None,
                                vertex_positions=vertex_positions,
                                scale_multiplier=iterator.scale_multiplier,
                                optimizer=stk.MCHammer(),
                            )
                        )
                    except ValueError:
                        continue
                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )

                    # Optimise and save.
                    logging.info("building %s", name)
                    try:
                        conformer = optimise_cage(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                            potential_names=[],
                        )
                        if conformer is not None:
                            num_components = len(
                                stko.Network.init_from_molecule(
                                    conformer.molecule
                                ).get_connected_components()
                            )
                            energy_per_bb = cgx.utilities.get_energy_per_bb(
                                energy_decomposition=(
                                    conformer.energy_decomposition
                                ),
                                number_building_blocks=(
                                    iterator.get_num_building_blocks()
                                ),
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
                                "topology_idx": sidx,
                                "mash_idx": midx,
                                "topology_code_vmap": tuple(
                                    (int(i[0]), int(i[1]))
                                    for i in stk_topology_code.vertex_map
                                ),
                                "bb_config_idx": bb_config.idx,
                            }
                            cgx.utilities.AtomliteDatabase(
                                database_path
                            ).add_properties(
                                key=name,
                                property_dict=properties,
                            )

                            analyse_cage(
                                database_path=database_path,
                                name=name,
                            )
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )
                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

                run_topology_codes = []
                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Filter graphs for 1-loops.
                    if contains_parallels(topology_code):
                        continue

                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
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
                                conformer = optimise_cage(
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

        if opt_ff:
            if pair not in ffopt_modifiable:
                msg = f"{pair} missing!"
                raise RuntimeError(msg)

            ffoptcalculation_dir = calculation_dir / "ff_opt"
            ffoptcalculation_dir.mkdir(exist_ok=True)

            if pair in ffopt_targets:
                mix_target = ffopt_targets[pair]

            else:
                logging.info("getting lowest energy to be target")
                mix_target = get_lowest_energy_entry(
                    [
                        (i, i.properties["energy_per_bb"])
                        for i in cgx.utilities.AtomliteDatabase(
                            database_path
                        ).get_entries()
                        if pair in i.key and "lowest_e_of_mash" in i.properties
                    ]
                )

            definer_dict = precursors_to_definer_dict(
                large=pairs[pair]["large"],
                small=pairs[pair]["small"],
                large_meas=ligand_measures[pairs[pair]["large_name"]],
                small_meas=ligand_measures[pairs[pair]["small_name"]],
                constant_definer_dict=constant_definer_dict,
            )

            modifiable = ffopt_modifiable[pair]
            logging.info(
                "running optimisation of %s molecules over %s",
                mix_target,
                modifiable,
            )

            target_optimisation(
                database_path=database_path,
                target_key=mix_target,
                calculation_dir=ffoptcalculation_dir,
                definer_dict=definer_dict,
                modifiable_terms=modifiable,
                forcefield=forcefield,
            )

    make_topt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_10.png",
        pairs=pairs,
        ffopt_targets=ffopt_targets,
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
        structure_dir=structure_dir,
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
        try:
            mix_target = ffopt_targets[pair]
        except (IndexError, KeyError):
            mix_target = get_lowest_energy_entry(
                [
                    (i, i.properties["energy_per_bb"])
                    for i in cgx.utilities.AtomliteDatabase(
                        database_path
                    ).get_entries()
                    if pair in i.key and "lowest_e_of_mash" in i.properties
                ]
            )

        ff_opt_plot(
            database_path=database_path,
            target=pair,
            figure_dir=figure_dir,
            filename=f"mgen_10_{pair}.png",
            key_target=mix_target,
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


def case_study_3(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
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
                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
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

    study_3_plot_5(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_5.png",
    )
    study_3_plot_2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
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
        structure_dir=structure_dir,
    )
    study_3_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )


def case_study_5(run: bool, opt_ff: bool) -> None:  # noqa: C901, PLR0912, PLR0915
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

    stoichiometries = ((2, 2, 1, 1), (2, 2, 0, 2), (2, 2, 2, 0))
    vdw_cutoff = 2
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
        "lin2": {"ba": 2.8, "aa": 1.5, "bac": 175},
        "mxy": {"be": 7.6, "ee": 5.0, "bed": 90},
        "pxy": {"be": 7.7, "ee": 5.8, "bed": 110},
        "fxy": {"be": 7.7, "ee": 5.8, "bed": 120},
        "tetra": {"mb": 2.0, "bmb": 90},
        "sqp": {"bf": 2.0, "bfb": 90, "fbm": 90},
    }

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
        },
        "mix4": {
            "linear": (
                "lin2",
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
        },
        "mix5": {
            "linear": (
                "lin2",
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
                vdw_bond_cutoff=vdw_cutoff,
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

            for sidx, stoichiometry in enumerate(stoichiometries):
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
                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
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

    study_5_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )
    study_5_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
    )


def case_study_6(run: bool, opt_ff: bool) -> None:  # noqa: C901, PLR0912, PLR0915
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
    multipliers = (1, 2)  # 3)
    # Very approximate.
    ligand_measures = {
        "cs6l1": {"ba": 3.8 / 3, "aa": 3.8 / 3, "bac": 180},
        "cs6l1b": {"ba": 3.8 / 3, "aa": 3.8 / 3, "bac": 160},
        "cs6l2": {"ba": 5.7 / 3, "aa": 5.7 / 3, "bac": 180},
        "cs6l2b": {"ba": 5.7 / 3, "aa": 5.7 / 3, "bac": 160},
        "cs6l5": {"ba": 8.0 / 3, "aa": 8.0 / 3, "bac": 180},
        "cs6l5b": {"ba": 8.0 / 3, "aa": 8.0 / 3, "bac": 160},
        "cs6l6": {"ba": 9.9 / 3, "aa": 9.9 / 3, "bac": 180},
        "cs6l6b": {"ba": 9.9 / 3, "aa": 9.9 / 3, "bac": 160},
        "cs6l9": {"ba": 14.2 / 3, "aa": 14.2 / 3, "bac": 180},
        "cs6l9b": {"ba": 14.2 / 3, "aa": 14.2 / 3, "bac": 160},
        "cs6zr1": {"bnb": 60, "nb": 3.5},
        "cs6zr2": {"bnb": 70, "nb": 3.5},
        "cs6cc31": {"bnb": 120, "nb": 2.9},
        "cs6cc32": {"ba": 1.5, "aa": 1.5, "bac": 115},
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
            "target": "l1zr1_1_0_2",
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
            "target": "l1zr2_1_0_3",
        },
        "l1bzr1": {
            "linear": (
                "cs6l1b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1bzr1_1_0_0",
        },
        "l1bzr2": {
            "linear": (
                "cs6l1b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l1bzr2_1_0_3",
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
            "target": "l2zr1_1_0_3",
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
            "target": "l2zr2_1_0_0",
        },
        "l2bzr1": {
            "linear": (
                "cs6l2b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2bzr1_1_0_2",
        },
        "l2bzr2": {
            "linear": (
                "cs6l2b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l2bzr2_1_0_4",
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
            "target": "l5zr1_1_0_1",
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
            "target": "l5zr2_1_0_4",
        },
        "l5bzr1": {
            "linear": (
                "cs6l5b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5bzr1_1_0_2",
        },
        "l5bzr2": {
            "linear": (
                "cs6l5b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l5bzr2_1_0_2",
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
            "target": "l6zr1_1_0_5",
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
            "target": "l6zr2_1_0_3",
        },
        "l6bzr1": {
            "linear": (
                "cs6l6b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6bzr1_1_0_1",
        },
        "l6bzr2": {
            "linear": (
                "cs6l6b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l6bzr2_1_0_0",
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
            "target": "l9zr1_1_0_4",
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
            "target": "l9zr2_1_0_3",
        },
        "l9bzr1": {
            "linear": (
                "cs6l9b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr1",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9bzr1_1_0_5",
        },
        "l9bzr2": {
            "linear": (
                "cs6l9b",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "trigonal": (
                "cs6zr2",
                cgx.molecular.ThreeC1Arm(
                    bead=trigonal_bead, abead1=binder_bead
                ),
            ),
            "target": "l9bzr2_1_0_0",
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
            "target": "cc3_2_4_4",
        },
    }

    cg_scale = 2
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
        if "b" in linear_name:
            cs6_definer_dict["bacab"] = ("tors", "0134", 180, 50, 1)

        forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
            identifier=f"{mix}ff",
            prefix=f"{mix}ff",
            vdw_bond_cutoff=vdw_cutoff,
            present_beads=present_beads,
            definer_dict=cs6_definer_dict,
        )
        if run:
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

        if opt_ff:
            ffoptcalculation_dir = calculation_dir / "ff_opt"
            ffoptcalculation_dir.mkdir(exist_ok=True)
            try:
                mix_target = mdict["target"]
            except KeyError:
                continue

            if "cc3" in mix:
                # Do not include the tritopic BB in optimisation, otherwise
                # another minimum is found by changing those angles.
                modifiable = ["bac", "ba", "ac"]
            else:
                modifiable = ["nb", "bnb", "bac", "ba", "ac"]
            logging.info(
                "running optimisation of %s molecules over %s",
                mix_target,
                modifiable,
            )
            target_optimisation(
                database_path=database_path,
                target_key=mix_target,
                calculation_dir=ffoptcalculation_dir,
                definer_dict=cs6_definer_dict,
                modifiable_terms=modifiable,
                forcefield=forcefield,
            )

    make_topt_s6_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_10.png",
        mixtures={i: mixtures[i] for i in mixtures if "cc3" not in i},
    )
    study_6_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )
    study_6_plot_2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
    )
    study_6_plot_3(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
    )
    study_6_plot_4(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
    )
    for mix, mdict in mixtures.items():
        try:
            mix_target = mdict["target"]
            ff_opt_plot(
                database_path=database_path,
                target=mix,
                figure_dir=figure_dir,
                filename=f"mgen_5_{mix}.png",
                key_target=mix_target,
            )
        except KeyError:
            continue


def case_study_starships(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run starship case study studying Pd(II) heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgenstar_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgenstar_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgenstar_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgenstar_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgenstar_cg"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgenstar.db"

    ligand_measures = {
        "la": {
            "dd": 7.0,
            "de": 1.5,
            "dde": 170,
            "eg": 1.4,
            "gb": 1.4,
            "egb": 120,
            "deg": 180,
        },
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
    }

    pairs = {
        "la_st5": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st52": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_c1": {
            "large_name": "la",
            "small_name": "c1",
            "stoichiometry_L_L_M": (4, 2, 3),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "la_st5_11": {
            "large_name": "la",
            "small_name": "st5",
            "stoichiometry_L_L_M": (1, 1, 1),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
            "vdw_cutoff": 2,
        },
        "la_st52_11": {
            "large_name": "la",
            "small_name": "st52",
            "stoichiometry_L_L_M": (1, 1, 1),
            "large": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "small": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
            "vdw_cutoff": 2,
        },
    }
    pairs_to_predict = [
        (
            (i.split("_")[0], "_".join(i.split("_")[1:])),
            pairs[i]["multipliers"],
        )
        for i in pairs
    ]

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

        if run:
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
                nmetals = pairs[pair]["stoichiometry_L_L_M"][2] * multiplier
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: pairs[pair]["stoichiometry_L_L_M"][2]
                        * multiplier,
                        large_bb: pairs[pair]["stoichiometry_L_L_M"][0]
                        * multiplier,
                        small_bb: pairs[pair]["stoichiometry_L_L_M"][1]
                        * multiplier,
                    },
                    graph_type=f"{1 * nmetals}P{2 * nmetals}",
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

                # Use known topology codes.
                stk_topology_code, stk_positions = get_stk_topology_code(
                    graph_type=f"{1 * nmetals}P{2 * nmetals}",
                )

                vertex_positions = {
                    nidx: np.array(stk_positions[nidx]) * 10
                    for nidx in stk_topology_code.get_nx_graph().nodes
                }
                sidx = -1
                midx = 0
                run_topology_codes = []
                for bb_config in possible_bbdicts:
                    name = (
                        f"{pair}_{multiplier}_{sidx}_{midx}_b{bb_config.idx}"
                    )

                    if not passes_graph_bb_iso(
                        topology_code=stk_topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append((stk_topology_code, bb_config))
                    try:
                        constructed_molecule = stk.ConstructedMolecule(
                            cgx.topologies.CustomTopology(  # type: ignore[arg-type]
                                building_blocks=bb_config.get_building_block_dictionary(),
                                vertex_prototypes=iterator.get_vertex_prototypes(
                                    unaligning=False
                                ),
                                # Convert to edge prototypes.
                                edge_prototypes=stk_topology_code.edges_from_connection(
                                    iterator.get_vertex_prototypes(
                                        unaligning=False
                                    )
                                ),
                                vertex_alignments=None,
                                vertex_positions=vertex_positions,
                                scale_multiplier=iterator.scale_multiplier,
                                optimizer=stk.MCHammer(),
                            )
                        )
                    except ValueError:
                        continue
                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )

                    # Optimise and save.
                    logging.info("building %s", name)
                    try:
                        conformer = optimise_cage(
                            molecule=constructed_molecule,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                            potential_names=[],
                        )
                        if conformer is not None:
                            num_components = len(
                                stko.Network.init_from_molecule(
                                    conformer.molecule
                                ).get_connected_components()
                            )
                            energy_per_bb = cgx.utilities.get_energy_per_bb(
                                energy_decomposition=(
                                    conformer.energy_decomposition
                                ),
                                number_building_blocks=(
                                    iterator.get_num_building_blocks()
                                ),
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
                                "topology_idx": sidx,
                                "mash_idx": midx,
                                "topology_code_vmap": tuple(
                                    (int(i[0]), int(i[1]))
                                    for i in stk_topology_code.vertex_map
                                ),
                                "bb_config_idx": bb_config.idx,
                            }
                            cgx.utilities.AtomliteDatabase(
                                database_path
                            ).add_properties(
                                key=name,
                                property_dict=properties,
                            )

                            analyse_cage(
                                database_path=database_path,
                                name=name,
                            )
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )
                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

                run_topology_codes = []
                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Filter graphs for 1-loops.
                    if contains_parallels(topology_code):
                        continue

                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
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
                                conformer = optimise_cage(
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
        structure_dir=structure_dir,
    )
    make_opt_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_5.png",
    )
    binder_vector_angles_plot_unsymm(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_7.png",
    )


def main() -> None:
    """Run script."""
    args = _parse_args()

    if args.study1:
        case_study_1_2(args.run, args.opt_ff)
    if args.study3:
        case_study_3(args.run)

    if args.study5:
        case_study_5(args.run, args.opt_ff)
    if args.study6:
        case_study_6(args.run, args.opt_ff)

    if args.starships:
        case_study_starships(args.run)


if __name__ == "__main__":
    main()
