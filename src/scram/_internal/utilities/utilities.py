"""Utilities module."""

import json
import logging
import pathlib
from collections import Counter

import bbprep
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko

from scram._internal.topologies.enumeration import (
    CustomTopology,
    TopologyCode,
)


def extract_ensemble(molecule: stk.Molecule, crest_run: pathlib.Path) -> dict:
    """Extract and save an ensemble from a crest run."""
    ensemble_dir = crest_run / "ensemble"
    num_atoms = molecule.get_num_atoms()
    ensemble = {}
    ensemble_dir.mkdir(exist_ok=True, parents=True)

    # Calculate geometrical properties.
    conformer_file = crest_run / "crest_conformers.xyz"
    with conformer_file.open("r") as f:
        linex = f.readlines()

    split_line = linex[0].rstrip()
    for i, conformers in enumerate("".join(linex).split(split_line)):
        lines = conformers.split("\n")

        if len(lines) != num_atoms + 3:
            continue

        energy = lines[1]
        position_matrix = []
        for line in lines[2:]:
            splits = line.rstrip().split()
            if len(splits) != 4:  # noqa: PLR2004
                continue
            symb, x, y, z = splits
            x = float(x)
            y = float(y)
            z = float(z)

            position_matrix.append(np.array((x, y, z)))

        conf_molecule = molecule.with_position_matrix(
            np.array(position_matrix)
        )

        calc = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        adjacent_centroids = calc.get_adjacent_centroids(conf_molecule)
        adjacent_distance = np.linalg.norm(
            adjacent_centroids[0] - adjacent_centroids[1]
        )

        ensemble[i] = {
            "energy": float(energy),
            "molecule": conf_molecule,
            "binder_angles": calc.get_binder_angles(conf_molecule),
            "binder_binder_angle": calc.get_binder_binder_angle(conf_molecule),
            "binder_distance": calc.get_binder_distance(conf_molecule),
            "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(
                conf_molecule
            ),
            "adjacent_distance": adjacent_distance,
            "binder_com_angle": calc.get_binder_centroid_angle(conf_molecule),
        }

        ensemble[i]["molecule"].write(ensemble_dir / f"conf_{i}.mol")
    return ensemble


def optimisation_sequence(
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
