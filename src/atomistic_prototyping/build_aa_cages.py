"""Perform analysis on ligand conformations."""

import logging
import pathlib

import numpy as np
import stk
import stko
from rdkit import RDLogger

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
        # stk.cage.LinearVertex(
        #     4, np.array([1, 1, 0.5]), use_neighbor_placement=False
        # ),
        stk.cage.UnaligningVertex(
            4, np.array([1, 1, 0.7]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([1, 1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        # stk.cage.LinearVertex(
        #     9, np.array([-1, -1, -0.5]), use_neighbor_placement=False
        # ),
        stk.cage.UnaligningVertex(
            9, np.array([-1, -1, -0.7]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            10, np.array([-1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            11, np.array([-1, 1, -0.5]), use_neighbor_placement=False
        ),
        ###
        stk.cage.UnaligningVertex(
            12, np.array([1, 1, 0.4]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            13, np.array([-1, -1, -0.4]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[12]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
    )


class M4L6C4_2(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([0, 1, 0])),
        stk.cage.NonLinearVertex(2, np.array([-1, 0, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -1, 0])),
        # stk.cage.LinearVertex(
        #     4, np.array([1, 1, 0.5]), use_neighbor_placement=False
        # ),
        stk.cage.UnaligningVertex(
            4, np.array([1, 1, 0.7]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([1, 1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        # stk.cage.LinearVertex(
        #     9, np.array([-1, -1, -0.5]), use_neighbor_placement=False
        # ),
        stk.cage.UnaligningVertex(
            9, np.array([-1, -1, -0.7]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            10, np.array([-1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            11, np.array([-1, 1, -0.5]), use_neighbor_placement=False
        ),
        ###
        stk.cage.UnaligningVertex(
            12, np.array([1, 1, 0.4]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            13, np.array([-1, -1, -0.4]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[12]),
        stk.Edge(5, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
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
        ###
        stk.cage.UnaligningVertex(10, np.array([0, 0, np.sqrt(6) / 2]) * 1.2),
        stk.cage.UnaligningVertex(
            11, np.array([-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]) * 1.2
        ),
        stk.cage.UnaligningVertex(
            12, np.array([1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]) * 1.2
        ),
        stk.cage.UnaligningVertex(
            13, np.array([0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6]) * 1.2
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
        stk.Edge(12, _vertex_prototypes[0], _vertex_prototypes[10]),
        stk.Edge(13, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(14, _vertex_prototypes[2], _vertex_prototypes[12]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[13]),
    )


class M4L6C4_4(stk.cage.Cage):  # noqa: N801
    """Represents a cage topology graph."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([0, 1, 0])),
        stk.cage.NonLinearVertex(2, np.array([-1, 0, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -1, 0])),
        stk.cage.LinearVertex(
            4, np.array([0, 0, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([0, 0, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            9, np.array([-1, 1, -0.5]), use_neighbor_placement=False
        ),
        ###
        stk.cage.UnaligningVertex(
            10, np.array([1.1, -0.1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            11, np.array([1.1, 0.1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            12, np.array([-1.1, -0.1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            13, np.array([-1.1, 0.1, 0.5]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[10]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[11]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[6]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[9]),
        stk.Edge(10, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(11, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(4, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(5, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[12]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[4]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[5]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
    )


def desymm_optimisation_sequence(  # noqa: PLR0915, PLR0913, PLR0912
    mol: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
    gulp_path: pathlib.Path,
    xtb_path: pathlib.Path,
    solvent_str: str | None,
) -> stk.Molecule:
    """Cage optimisation sequence."""
    gulp1_output = calc_dir / f"{name}_gulp1.mol"
    gulp2_output = calc_dir / f"{name}_gulp2.mol"
    gulpmd_output = calc_dir / f"{name}_gulpmd.mol"
    xtbopt_output = calc_dir / f"{name}_xtb.mol"
    xtbsolvopt_output = calc_dir / f"{name}_xtb_dmso.mol"

    if not xtb_path.exists():
        msg = f"xtb is not installed here: {xtb_path}"
        raise ValueError(msg)
    if not gulp_path.exists():
        msg = f"gulp is not installed here: {gulp_path}"
        raise ValueError(msg)

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

    if not gulpmd_output.exists():
        logging.info("    UFF4MOF equilib MD of %s", name)
        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=gulp_path,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmde",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.25,
            equilbration=0.5,
            production=0.5,
            N_conformers=2,
            opt_conformers=False,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulp2_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulp2_mol)

        logging.info("    UFF4MOF production MD of %s", name)
        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=gulp_path,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmd",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.75,
            equilbration=0.5,
            production=200.0,
            N_conformers=40,
            opt_conformers=True,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulpmd_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulpmd_mol)
        gulpmd_mol.write(gulpmd_output)
    else:
        logging.info("    loading %s", gulpmd_output)
        gulpmd_mol = mol.with_structure_from_file(str(gulpmd_output))

    if not xtbopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbopt"
        logging.info("    xtb optimisation of %s", name)
        xtb_opt = stko.XTB(
            xtb_path=xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent=None,
        )
        xtbopt_mol = xtb_opt.optimize(mol=gulpmd_mol)
        xtbopt_mol.write(xtbopt_output)
    else:
        logging.info("    loading %s", xtbopt_output)
        xtbopt_mol = mol.with_structure_from_file(str(xtbopt_output))

    if solvent_str is None:
        return mol.with_structure_from_file(str(xtbopt_output))

    if not xtbsolvopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbsolvopt"
        logging.info(
            "    solvated xtb optimisation of %s with %s", name, solvent_str
        )
        xtb_opt = stko.XTB(
            xtb_path=xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent_model="alpb",
            solvent=solvent_str,
            solvent_grid="verytight",
        )
        xtbsolvopt_mol = xtb_opt.optimize(mol=xtbopt_mol)
        xtbsolvopt_mol.write(xtbsolvopt_output)
    else:
        logging.info("    loading %s", xtbsolvopt_output)
        xtbsolvopt_mol = mol.with_structure_from_file(str(xtbsolvopt_output))

    return mol.with_structure_from_file(str(xtbsolvopt_output))


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

    pd = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0, 0, 0]],
    )

    ditopic = stk.BuildingBlock.init_from_file(
        path=ligand_dir / "ltp_prep.mol",
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
        ),
    )

    capper = stk.BuildingBlock(
        smiles="CC1=CC=NC=C1",
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
        ),
    )

    topology_graphs = {
        "2P4": stk.cage.M2L4Lantern,
        "4P6C4_1": M4L6C4_1,
        "4P6C4_2": M4L6C4_2,
        "4P6C4_3": M4L6C4_3,
        "4P6C4_4": M4L6C4_4,
    }
    for tstr, tfun in topology_graphs.items():
        match tstr:
            case "2P4":
                charge = 2 * 2
                building_blocks = (pd, ditopic)
            case "4P6C4_1":
                charge = 2 * 4
                building_blocks = (pd, ditopic, capper)
            case "4P6C4_2":
                charge = 2 * 4
                building_blocks = (pd, ditopic, capper)
            case "4P6C4_3":
                charge = 2 * 4
                building_blocks = (pd, ditopic, capper)
            case "4P6C4_4":
                charge = 2 * 4
                building_blocks = (pd, ditopic, capper)
            case _:
                raise RuntimeError

        name = f"{tstr}"
        optc_file = cage_dir / f"{name}_optc.mol"

        if optc_file.exists():
            continue

        cage_molecule = stk.ConstructedMolecule(
            topology_graph=tfun(
                building_blocks=building_blocks,
                optimizer=stk.MCHammer(),
                reaction_factory=react_factory,
                scale_multiplier=1,
            )
        )
        cage_molecule.write(cage_dir / f"{name}_unopt.mol")

        cage_molecule = desymm_optimisation_sequence(
            mol=cage_molecule,
            name=name,
            charge=charge,
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
