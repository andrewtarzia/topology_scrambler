"""Perform analysis on ligand conformations."""

import logging
import pathlib

import cgexplore as cgx
import numpy as np
import stk
import stko
from rdkit import RDLogger

from atomistic_prototyping.build_aa_cages import desymm_optimisation_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
react_factory = stk.DativeReactionFactory(
    stk.GenericReactionFactory(
        bond_orders={
            frozenset({stko.functional_groups.ThreeSiteFG, stk.SingleAtom}): 9,
        },
    ),
)


class M4L6C4_1(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([0, 1, 0])),
        stk.cage.NonLinearVertex(2, np.array([-1, 0, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -1, 0])),
        stk.cage.LinearVertex(
            4, np.array([1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([1, 1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([1, -1, 0.0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            9, np.array([-1, 1, 0.0]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[9]),
        stk.Edge(6, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(7, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(9, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(10, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[3], _vertex_prototypes[6]),
    )


class M4L6C4_2(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([0, 1, 0])),
        stk.cage.NonLinearVertex(2, np.array([-1, 0, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -1, 0])),
        stk.cage.LinearVertex(
            4, np.array([1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([1, 1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([0, 0, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            9, np.array([0, 0, 0.5]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[9]),
        stk.Edge(6, _vertex_prototypes[2], _vertex_prototypes[6]),
        stk.Edge(7, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(9, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(10, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[3], _vertex_prototypes[9]),
    )


class M4L6C4_3(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, np.array([0, 0, np.sqrt(6) / 2])),
        stk.cage.NonLinearVertex(
            1, np.array([-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
        stk.cage.NonLinearVertex(
            2, np.array([1, -np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
        stk.cage.NonLinearVertex(
            3, np.array([0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
    )

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex.init_at_center(
            id=4,
            vertices=(_non_linears[0], _non_linears[1]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=5,
            vertices=(_non_linears[0], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=6,
            vertices=(_non_linears[0], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=7,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=8,
            vertices=(_non_linears[1], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=9,
            vertices=(_non_linears[2], _non_linears[3]),
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(6, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(7, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(9, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(10, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[3], _vertex_prototypes[9]),
    )


class M4L6C4_4(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1, 0, 0])),
        stk.cage.LinearVertex(1, np.array([0, 1, 0])),
        stk.cage.NonLinearVertex(2, np.array([-1, 0, 0])),
        stk.cage.LinearVertex(3, np.array([0, -1, 0])),
        stk.cage.LinearVertex(
            4, np.array([1, 1, 0.0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([-1, 1, 0.0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([0, 0, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([-1, -1, 0.0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([1, -1, 0.0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            9, np.array([0, 0, 0.5]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[8]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[9]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(7, _vertex_prototypes[2], _vertex_prototypes[6]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(10, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(11, _vertex_prototypes[3], _vertex_prototypes[8]),
    )


def shortest_distance_to_plane(plane: np.ndarray, point: np.ndarray) -> float:
    """Calculate the perpendicular distance from a point and a plane."""
    top = (
        plane[0] * point[0]
        + plane[1] * point[1]
        + plane[2] * point[2]
        - plane[3]
    )
    bottom = np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    return top / bottom


def extract_crest_ensemble(
    molecule: stk.Molecule,
    crest_run: pathlib.Path,
) -> dict:
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
        conf_molecule.write(ensemble_dir / f"conf_{i}.mol")

        centroid = conf_molecule.get_centroid(
            atom_ids=list(range(len(list(conf_molecule.get_atoms()))))
        )
        normal = conf_molecule.get_plane_normal(
            atom_ids=list(range(len(list(conf_molecule.get_atoms()))))
        )
        # Plane of equation ax + by + cz = d.
        plane_of_best_fit = np.append(normal, np.sum(normal * centroid))
        deviation_atom_ids = [
            i.get_id()
            for i in conf_molecule.get_atoms()
            if i.get_atomic_number() == 35  # noqa: PLR2004
        ]

        br_distance_from_plane = [
            shortest_distance_to_plane(
                plane=plane_of_best_fit,
                point=next(
                    conf_molecule.get_atomic_positions(atom_ids=i.get_id()),
                ),
            )
            for i in conf_molecule.get_atoms()
            if i.get_id() in deviation_atom_ids
        ]

        ensemble[i] = {
            "energy": float(energy),
            "distance_from_plane_for_bromines": br_distance_from_plane,
        }

    return ensemble


def complex_optimisation_sequence(  # noqa: PLR0913
    mol: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
    xtb_path: pathlib.Path,
    gulp_path: pathlib.Path,
    crest_path: pathlib.Path,
) -> stk.Molecule:
    """Cage optimisation sequence."""
    if not xtb_path.exists():
        msg = f"xtb is not installed here: {xtb_path}"
        raise ValueError(msg)
    if not crest_path.exists():
        msg = f"crest is not installed here: {crest_path}"
        raise ValueError(msg)
    if not gulp_path.exists():
        msg = f"gulp is not installed here: {gulp_path}"
        raise ValueError(msg)

    gulp1_output = calc_dir / f"{name}_gulp1.mol"
    gulp2_output = calc_dir / f"{name}_gulp2.mol"
    crest_output = calc_dir / f"{name}_crest.mol"

    if not gulp1_output.exists():
        output_dir = calc_dir / f"{name}_gulp1"

        logging.info("    UFF4MOF optimisation 1 of %s CG: True", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(mol)
        gulp1_mol = gulp_opt.optimize(mol=mol)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info("    loading %s", gulp1_output)
        gulp1_mol = mol.with_structure_from_file(str(gulp1_output))

    if not gulp2_output.exists():
        output_dir = calc_dir / f"{name}_gulp2"

        logging.info("    UFF4MOF optimisation 2 of %s CG: False", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
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
        gulp2_mol = mol.with_structure_from_file(str(gulp2_output))

    crest_run = calc_dir / f"{name}_crest"
    if not crest_output.exists():
        # Run calculation.
        logging.info("    CREST generation of %s", name)
        optimiser = cgx.atomistic.Crest(
            crest_path=crest_path,
            xtb_path=xtb_path,
            output_dir=crest_run,
            gfn_method="gfnff",
            num_cores=12,
            unlimited_memory=True,
            solvent=None,
            solvent_model="alpb",
            solvent_grid="verytight",
            additional_commands=(
                "--optlev crude",
                # No z matrix sorting.
                "--nosz",
                "--keepdir",
                # Set energy threshold (kcal.mol)
                "--ewin 10",
                "--mquick",
            ),
            charge=charge,
            electronic_temperature=300,
            num_unpaired_electrons=0,
        )

        opt_molecule = optimiser.optimize(gulp2_mol)
        opt_molecule.write(crest_output)

    else:
        logging.info("    loading %s", crest_output)
        opt_molecule = mol.with_structure_from_file(str(crest_output))

    # Analyse ensemble.
    ensemble_data = extract_crest_ensemble(
        molecule=opt_molecule,
        crest_run=crest_run,
    )

    chosen_score = 0
    chosen_idx = 0
    for idx, data in ensemble_data.items():
        if all(i < 0 for i in data["distance_from_plane_for_bromines"]) or all(
            i > 0 for i in data["distance_from_plane_for_bromines"]
        ):
            score = sum(
                abs(i) for i in data["distance_from_plane_for_bromines"]
            )
            if score > chosen_score:
                chosen_idx = idx
                chosen_score = score

    return mol.with_structure_from_file(
        str(crest_run / "ensemble" / f"conf_{chosen_idx}.mol")
    )


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/unreacted/")
    ligand_dir = wd / "aa_ligands"
    cage_dir = wd / "aa_cages"
    cage_dir.mkdir(exist_ok=True)
    calculations_dir = wd / "aa_calculations"
    calculations_dir.mkdir(exist_ok=True)

    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")
    xtb_path = pathlib.Path("/home/atarzia/miniforge3/envs/tscram/bin/xtb")
    crest_path = pathlib.Path("/home/atarzia/software/crest_301/crest")

    pd = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0, 0, 0]],
    )

    reactor = stk.BuildingBlock(
        smiles="C1=CC(=CN=C1)Br",
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
        ),
    )

    capper = stk.BuildingBlock(
        smiles="CC#N",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts="[#7X1]~[#6]~[#6]", bonders=(0,), deleters=()
            ),
        ),
    )

    complexes = {
        "c1": {reactor: (0, 1, 2, 3)},
        "c2": {reactor: (0, 1, 2), capper: (3,)},
        "c3": {reactor: (0, 2), capper: (1, 3)},
    }
    complex_bbs = {}
    for complex_name, complex_bbdict in complexes.items():
        opt_file = ligand_dir / f"{complex_name}_optc.mol"
        if not opt_file.exists():
            molecule = stk.ConstructedMolecule(
                topology_graph=stk.metal_complex.SquarePlanar(
                    metals=pd,
                    ligands=complex_bbdict,
                    optimizer=stk.MCHammer(),
                ),
            )
            molecule.write(ligand_dir / f"{complex_name}_unopt.mol")

            molecule = complex_optimisation_sequence(
                mol=molecule,
                name=complex_name,
                charge=2,
                calc_dir=calculations_dir,
                xtb_path=xtb_path,
                gulp_path=gulp_path,
                crest_path=crest_path,
            )
            molecule.write(opt_file)

        complex_bbs[complex_name] = stk.BuildingBlock.init_from_file(
            path=opt_file,
            functional_groups=(stk.BromoFactory(),),
        )

    ditopic = stk.BuildingBlock.init_from_file(
        path=ligand_dir / "alcltp_lowe.mol",
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory(
                smarts="[#6]~[#7X3H1](~[H])~[#6]",
                bonders=(1,),
                deleters=(2,),
            ),
        ),
    )

    topology_graphs = {
        "2P4": {
            "graph": stk.cage.M2L4Lantern,
            "bbs": {complex_bbs["c1"]: (0, 1), ditopic: (2, 3, 4, 5)},
            "charge": 4,
            "vas": {0: 0, 1: 0},
        },
        "4P6C4_1": {
            "graph": M4L6C4_1,
            "bbs": {
                complex_bbs["c2"]: (0, 1, 2, 3),
                ditopic: (4, 5, 6, 7, 8, 9),
            },
            "vas": {0: 0, 1: 1, 2: 0, 3: 2},
            "charge": 8,
        },
        "4P6C4_2": {
            "graph": M4L6C4_2,
            "bbs": {
                complex_bbs["c2"]: (0, 1, 2, 3),
                ditopic: (4, 5, 6, 7, 8, 9),
            },
            "charge": 8,
            "vas": {0: 0, 1: 0, 2: 1, 3: 0},
        },
        "4P6C4_3": {
            "graph": M4L6C4_3,
            "bbs": {
                complex_bbs["c2"]: (0, 1, 2, 3),
                ditopic: (4, 5, 6, 7, 8, 9),
            },
            "charge": 8,
            "vas": {0: 0, 1: 0, 2: 2, 3: 2},
        },
        "4P6C4_4": {
            "graph": M4L6C4_4,
            "bbs": {
                complex_bbs["c1"]: (0, 2),
                complex_bbs["c3"]: (1, 3),
                ditopic: (4, 5, 6, 7, 8, 9),
            },
            "charge": 8,
            "vas": {0: 0, 1: 0, 2: 0, 3: 0},
        },
    }
    for tstr, tdict in topology_graphs.items():
        name = f"alc_{tstr}"
        optc_file = cage_dir / f"{name}_optc.mol"

        if optc_file.exists():
            continue

        cage_molecule = stk.ConstructedMolecule(
            topology_graph=tdict["graph"](
                building_blocks=tdict["bbs"],
                vertex_alignments=tdict["vas"],
                reaction_factory=react_factory,
                scale_multiplier=1,
            )
        )
        cage_molecule.write(cage_dir / f"{name}_nooptforviz.mol")
        cage_molecule = stk.ConstructedMolecule(
            topology_graph=tdict["graph"](
                building_blocks=tdict["bbs"],
                vertex_alignments=tdict["vas"],
                optimizer=stk.MCHammer(),
                reaction_factory=react_factory,
                scale_multiplier=1,
            )
        )
        cage_molecule.write(cage_dir / f"{name}_unopt.mol")

        cage_molecule = desymm_optimisation_sequence(
            mol=cage_molecule,
            name=name,
            charge=tdict["charge"],
            calc_dir=calculations_dir,
            gulp_path=gulp_path,
            xtb_path=xtb_path,
            solvent_str="acetonitrile",
        )
        if cage_molecule is None:
            continue
        cage_molecule = cage_molecule.with_centroid((0, 0, 0))
        cage_molecule.write(optc_file)


if __name__ == "__main__":
    main()
