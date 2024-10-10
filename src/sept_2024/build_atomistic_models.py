"""Script to generate and optimise CG models."""

import logging
import pathlib

import stk
import stko
from rdkit import RDLogger

import scram

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
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
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
        "la": scram.atomistic.get_ligand_bb(
            path=ligand_dir / "la_prep.mol",
            optl_path=ligand_dir / "la_optl.mol",
        ),
        "las": scram.atomistic.get_ligand_bb(
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
        "4P8": scram.topologies.CGM4L8,
        "4P82": scram.topologies.M4L82,
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

            cage_molecule = scram.atomistic.desymm_optimisation_sequence(
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

            cage_molecule = scram.atomistic.desymm_optimisation_sequence(
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
                "Extb/acetonitrile/au: %s",
                calculate_xtb_energy(
                    molecule=stk.BuildingBlock.init_from_file(optc_file),
                    name=name,
                    charge=charge,
                    calc_dir=calculation_dir,
                    solvent="acetonitrile",
                ),
            )

        for pair in pairs:
            bb1, bb2 = pair
            name = f"p_{tstr}_{bb1}_{bb2}"
            optc_file = structure_dir / f"{name}_optc.mol"

            if not optc_file.exists():
                raise RuntimeError

            logging.info(
                "Extb/acetonitrile/au: %s",
                calculate_xtb_energy(
                    molecule=stk.BuildingBlock.init_from_file(optc_file),
                    name=name,
                    charge=charge,
                    calc_dir=calculation_dir,
                    solvent="acetonitrile",
                ),
            )


if __name__ == "__main__":
    main()
