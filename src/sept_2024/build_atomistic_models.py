"""Script to generate and optimise CG models."""

import logging
import pathlib

import cgexplore as cgx
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


def calculate_xtb_energy(  # noqa: PLR0913
    molecule: stk.Molecule,
    name: str,
    charge: int,
    solvent: str,
    calc_dir: pathlib.Path,
    xtb_path: pathlib.Path,
) -> float:
    """Calculate energy."""
    output_dir = calc_dir / f"{name}_xtbey"
    output_file = calc_dir / f"{name}_xtb.ey"

    if output_file.exists():
        with output_file.open("r") as f:
            lines = f.readlines()
        for line in lines:
            energy = float(line.rstrip())
            break
    else:
        logging.info("xtb energy calculation of %s", name)
        xtb = stko.XTBEnergy(
            xtb_path=xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            num_unpaired_electrons=0,
            unlimited_memory=True,
            solvent_model="alpb",
            solvent=solvent,
            solvent_grid="verytight",
        )
        energy = xtb.get_energy(mol=molecule)
        with output_file.open("w") as f:
            f.write(f"{energy}\n")

    # In a.u.
    return energy


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    calculation_dir = wd / "ratom_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "ratom_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "ligands"
    data_dir = wd / "ratom_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")
    xtb_path = pathlib.Path("/home/atarzia/miniforge3/envs/tscram/bin/xtb")

    buildingblocks = {
        "la": cgx.atomistic.get_ditopic_aligned_bb(
            path=ligand_dir / "la_prep.mol",
            optl_path=ligand_dir / "la_optl.mol",
        ),
        "las": cgx.atomistic.get_ditopic_aligned_bb(
            path=ligand_dir / "las_prep.mol",
            optl_path=ligand_dir / "las_optl.mol",
        ),
        "las2": stk.BuildingBlock.init_from_file(
            path=ligand_dir / "las2_manual.mol",
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        ),
        "st5": stk.BuildingBlock.init_from_file(
            path=ligand_dir / "st5_manual.mol",
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        ),
        "pd": stk.BuildingBlock(
            smiles="[Pd+2]",
            functional_groups=(
                stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
            ),
            position_matrix=[[0, 0, 0]],
        ),
    }

    pairs = (("la", "st5"), ("las", "st5"), ("las2", "st5"))
    topology_graphs = {
        # "3P6": stk.cage.M3L6,  # noqa: ERA001
        "4P8": cgx.topologies.CGM4L8,
        "4P82": cgx.topologies.M4L82,
    }

    for tstr in topology_graphs:
        match tstr:
            case "3P6":
                charge = 2 * 3
            case "4P8":
                charge = 2 * 4
            case "4P82":
                charge = 2 * 4
            case _:
                raise RuntimeError

        for pair in pairs:
            bb1, bb2 = pair
            name = f"p_{tstr}_{bb1}_{bb2}"
            optc_file = structure_dir / f"{name}_optc.mol"
            match tstr:
                case "3P6":
                    building_block_dict = {
                        buildingblocks["pd"]: (0, 1, 2),
                        buildingblocks[bb1]: (3, 4, 5, 6),
                        buildingblocks[bb2]: (7, 8),
                    }
                    optimiser = stk.MCHammer(target_bond_length=2.0)
                    scale = 1

                case "4P8":
                    building_block_dict = {
                        buildingblocks["pd"]: (0, 1, 2, 3),
                        buildingblocks[bb1]: (5, 7, 9, 11),
                        buildingblocks[bb2]: (4, 6, 8, 10),
                    }
                    optimiser = stk.MCHammer(target_bond_length=2.0)
                    scale = 1

                case "4P82":
                    building_block_dict = {
                        buildingblocks["pd"]: (0, 1, 2, 3),
                        buildingblocks[bb1]: (5, 6, 7, 8),
                        buildingblocks[bb2]: (4, 9, 10, 11),
                    }
                    optimiser = stk.MCHammer(target_bond_length=5.0)
                    scale = 2

                case _:
                    raise RuntimeError

            if optc_file.exists():
                continue

            cage_molecule = stk.ConstructedMolecule(
                topology_graph=topology_graphs[tstr](
                    building_blocks=building_block_dict,
                    optimizer=optimiser,
                    reaction_factory=react_factory,
                    scale_multiplier=scale,
                )
            )
            cage_molecule.write(structure_dir / f"{name}_unopt.mol")

            cage_molecule = desymm_optimisation_sequence(
                mol=cage_molecule,
                name=name,
                charge=charge,
                calc_dir=calculation_dir,
                gulp_path=gulp_path,
                xtb_path=xtb_path,
                solvent_str="acetonitrile",
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)

        for buildingblock in buildingblocks:
            if buildingblock == "pd":
                continue
            name = f"h_{tstr}_{buildingblock}"
            optc_file = structure_dir / f"{name}_optc.mol"
            if optc_file.exists():
                continue

            cage_molecule = stk.ConstructedMolecule(
                topology_graph=topology_graphs[tstr](
                    building_blocks=(
                        buildingblocks["pd"],
                        buildingblocks[buildingblock],
                    ),
                    optimizer=stk.MCHammer(target_bond_length=2.0),
                    reaction_factory=react_factory,
                )
            )
            cage_molecule.write(structure_dir / f"{name}_unopt.mol")

            cage_molecule = desymm_optimisation_sequence(
                mol=cage_molecule,
                name=name,
                calc_dir=calculation_dir,
                charge=charge,
                gulp_path=gulp_path,
                xtb_path=xtb_path,
                solvent_str="acetonitrile",
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)

    # Get all energies.
    logging.info("system energies:")
    for tstr in topology_graphs:
        match tstr:
            case "3P6":
                charge = 2 * 3
            case "4P8":
                charge = 2 * 4
            case "4P82":
                charge = 2 * 4
            case _:
                raise RuntimeError

        for buildingblock in buildingblocks:
            if buildingblock == "pd":
                continue
            name = f"h_{tstr}_{buildingblock}"
            optc_file = structure_dir / f"{name}_optc.mol"
            if not optc_file.exists():
                raise RuntimeError

            logging.info(
                "%s: Extb/acetonitrile/au = %s",
                name,
                calculate_xtb_energy(
                    molecule=stk.BuildingBlock.init_from_file(optc_file),
                    name=name,
                    charge=charge,
                    calc_dir=calculation_dir,
                    solvent="acetonitrile",
                    xtb_path=xtb_path,
                ),
            )

        for pair in pairs:
            bb1, bb2 = pair
            name = f"p_{tstr}_{bb1}_{bb2}"
            optc_file = structure_dir / f"{name}_optc.mol"

            if not optc_file.exists():
                raise RuntimeError

            logging.info(
                "%s: Extb/acetonitrile/au = %s",
                name,
                calculate_xtb_energy(
                    molecule=stk.BuildingBlock.init_from_file(optc_file),
                    name=name,
                    charge=charge,
                    calc_dir=calculation_dir,
                    solvent="acetonitrile",
                    xtb_path=xtb_path,
                ),
            )


if __name__ == "__main__":
    main()
