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


def main() -> None:  # noqa: PLR0915
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
                    charge = 2 * 3
                    optimiser = stk.MCHammer(target_bond_length=2.0)
                    scale = 1

                case "4P8":
                    building_block_dict = {
                        buildingblocks["pd"]: (0, 1, 2, 3),
                        buildingblocks[bb1]: (5, 7, 9, 11),
                        buildingblocks[bb2]: (4, 6, 8, 10),
                    }
                    charge = 2 * 4
                    optimiser = stk.MCHammer(target_bond_length=2.0)
                    scale = 1

                case "4P82":
                    building_block_dict = {
                        buildingblocks["pd"]: (0, 1, 2, 3),
                        buildingblocks[bb1]: (5, 6, 7, 8),
                        buildingblocks[bb2]: (4, 9, 10, 11),
                    }
                    charge = 2 * 4
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


if __name__ == "__main__":
    main()
