"""Utilities module."""

import json
import logging
import pathlib

import bbprep
import numpy as np
import stk
import stko

from scram._internal.topologies.enumeration import (
    CustomTopology,
    TopologyCode,
)


def get_vertex_positions(
    name: str,
    calculation_dir: pathlib.Path,
) -> dict[int, np.ndarray]:
    """Get vertex positions."""
    vertex_file = calculation_dir / f"{name}_vertices.json"
    with vertex_file.open("r") as f:
        centroids = json.load(f)
    return {int(i): np.array(centroids[i]) for i in centroids}


react_factory = stk.DativeReactionFactory(
    stk.GenericReactionFactory(
        bond_orders={
            frozenset({stko.functional_groups.ThreeSiteFG, stk.SingleAtom}): 9,
        },
    ),
)


def atomise(  # noqa: PLR0913
    vertices: list[stk.Vertex],
    name: str,
    topology_code: TopologyCode,
    calculation_dir: pathlib.Path,
    atomistic_dir: pathlib.Path,
    atomistic_calculation_dir: pathlib.Path,
    building_blocks: dict[stk.BuildingBlock : tuple[int, ...]],
    optimizer: stk.Optimizer,
) -> None:
    """Make a toy model atomistic."""
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
                    reaction_factory=react_factory,
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
                    reaction_factory=react_factory,
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
                    reaction_factory=react_factory,
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


def get_ligand_bb(
    path: pathlib.Path,
    optl_path: pathlib.Path,
) -> stk.BuildingBlock:
    """Get building block for the target ligand and prepare for cage model."""
    try:
        return stk.BuildingBlock.init_from_file(
            path=path,
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        )
    except OSError:
        temp = stk.BuildingBlock.init_from_file(
            path=optl_path,
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        )
        generator = bbprep.generators.ETKDG(num_confs=100)
        ensemble = generator.generate_conformers(temp)
        process = bbprep.DitopicFitter(ensemble=ensemble)
        min_molecule = process.get_minimum()
        min_molecule.molecule.write(path)

    return stk.BuildingBlock.init_from_file(
        path=path,
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
        ),
    )
