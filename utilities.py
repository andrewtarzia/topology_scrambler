"""Utilities module."""

import pathlib
from collections import Counter
import stk

import matplotlib.pyplot as plt
import logging
import json
import numpy as np
import stko
from topologies import CustomTopology


def optimisation_sequence(  # noqa: PLR0915
    cage: stk.Molecule,
    name: str,
    calculation_dir: pathlib.Path,
) -> stk.Molecule:
    """Cage optimisation sequence."""
    gulp_dir = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")

    gulp1_output = calculation_dir / f"{name}_gulp1.mol"
    gulp2_output = calculation_dir / f"{name}_gulp2.mol"

    if not gulp1_output.exists():
        output_dir = calculation_dir / f"{name}_gulp1"

        logging.info("    UFF4MOF optimisation 1 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_dir,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(cage)
        gulp1_mol = gulp_opt.optimize(mol=cage)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info("    loading %s", gulp1_output)
        gulp1_mol = cage.with_structure_from_file(gulp1_output)

    if not gulp2_output.exists():
        output_dir = calculation_dir / f"{name}_gulp2"
        logging.info("    UFF4MOF optimisation 2 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_dir,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=False,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        logging.info("    loading %s", gulp2_output)
        gulp2_mol = cage.with_structure_from_file(gulp2_output)

    return cage.with_structure_from_file(gulp2_output)


def plot_xy(
    xproperty: str,
    ensemble: dict[str:dict],
    min_energy: float,
    figure_dir: pathlib.Path,
    ligand_name: str,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if xproperty in ("binder_angles",):
        ax.scatter(
            [ensemble[i][xproperty][0] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )
        ax.scatter(
            [ensemble[i][xproperty][1] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    elif xproperty in ("torsion_state",):
        xs = [Counter(ensemble[i][xproperty]) for i in ensemble]

        xs = [i.get("b", 0) for i in xs]

        ax.scatter(
            xs,
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    else:
        ax.scatter(
            [ensemble[i][xproperty] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xproperty, fontsize=16)
    ax.set_ylabel("relative energy [kJ/mol]", fontsize=16)
    if xproperty == "binder_adjacent_torsion":
        ax.set_xlim(-180, 180)

    if xproperty == "binder_angles":
        ax.set_xlim(0, 180)

    if xproperty == "binder_binder_angle":
        ax.set_xlim(0, 180)

    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"xy_{xproperty}_{ligand_name}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def get_vertex_positions(name, topology_code, structure_dir, calculation_dir):
    vertex_file = calculation_dir / f"{name}_vertices.json"
    with vertex_file.open("r") as f:
        centroids = json.load(f)
    return {int(i): np.array(centroids[i]) for i in centroids}


def react_factory():
    return stk.DativeReactionFactory(
        stk.GenericReactionFactory(
            bond_orders={
                frozenset({stko.functional_groups.ThreeSiteFG, stk.SingleAtom}): 9,
            },
        ),
    )


def atomise(
    vertices,
    name,
    topology_code,
    structure_dir,
    calculation_dir,
    atomistic_dir,
    atomistic_calculation_dir,
    building_blocks,
    optimizer,
):
    angled_vertices = tuple(
        stk.cage.NonLinearVertex(
            id=i.get_id(),
            position=i.get_position(),
            aligner_edge=i.get_aligner_edge(),
            use_neighbor_placement=i.use_neighbor_placement,
        )
        if i.__class__.__name__ == "NonLinearVertex"
        else stk.cage.AngledVertex(
            id=i.get_id(),
            position=i.get_position(),
            aligner_edge=i.get_aligner_edge(),
            use_neighbor_placement=i.use_neighbor_placement,
        )
        for i in vertices
    )

    fake_vertices = tuple(
        stk.cage.UnaligningVertex(
            id=i.get_id(),
            position=i.get_position(),
            aligner_edge=i.get_aligner_edge(),
            use_neighbor_placement=i.use_neighbor_placement,
        )
        for i in vertices
    )

    logging.info("loading positions of %s", name)
    vertex_positions = get_vertex_positions(
        name=name,
        topology_code=topology_code,
        structure_dir=structure_dir,
        calculation_dir=calculation_dir,
    )

    try:
        cage_name = f"co_{name}"
        optc_file = atomistic_dir / f"{cage_name}_optc.mol"
        if not optc_file.exists():
            logging.info("building AA model of %s", cage_name)
            cage_molecule = stk.ConstructedMolecule(
                CustomTopology(
                    building_blocks=building_blocks,
                    vertex_prototypes=vertices,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=vertices[vmap[0]],
                            vertex2=vertices[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=vertex_positions,
                    scale_multiplier=20.0,
                    optimizer=optimizer,
                    reaction_factory=react_factory(),
                )
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(atomistic_dir / f"{cage_name}_unopt.mol")
            cage_molecule = optimisation_sequence(
                cage=cage_molecule,
                name=cage_name,
                calculation_dir=atomistic_calculation_dir,
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)
    except ValueError:
        pass

    try:
        cage_name = f"cu_{name}"
        optc_file = atomistic_dir / f"{cage_name}_optc.mol"
        if not optc_file.exists():
            logging.info("using unaligning for %s", cage_name)
            cage_molecule = stk.ConstructedMolecule(
                CustomTopology(
                    building_blocks=building_blocks,
                    vertex_prototypes=fake_vertices,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=fake_vertices[vmap[0]],
                            vertex2=fake_vertices[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=vertex_positions,
                    scale_multiplier=20.0,
                    optimizer=optimizer,
                    reaction_factory=react_factory(),
                )
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(atomistic_dir / f"{cage_name}_unopt.mol")
            cage_molecule = optimisation_sequence(
                cage=cage_molecule,
                name=cage_name,
                calculation_dir=atomistic_calculation_dir,
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)
    except ValueError:
        pass

    try:
        cage_name = f"ca_{name}"
        optc_file = atomistic_dir / f"{cage_name}_optc.mol"
        if not optc_file.exists():
            logging.info("using angled vertices for %s", cage_name)
            cage_molecule = stk.ConstructedMolecule(
                CustomTopology(
                    building_blocks=building_blocks,
                    vertex_prototypes=angled_vertices,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=angled_vertices[vmap[0]],
                            vertex2=angled_vertices[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=vertex_positions,
                    scale_multiplier=20.0,
                    optimizer=optimizer,
                    reaction_factory=react_factory(),
                )
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(atomistic_dir / f"{cage_name}_unopt.mol")
            cage_molecule = optimisation_sequence(
                cage=cage_molecule,
                name=cage_name,
                calculation_dir=atomistic_calculation_dir,
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)
    except ValueError:
        pass
