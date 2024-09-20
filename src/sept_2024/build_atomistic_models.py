"""Script to generate and optimise CG models."""

import logging
import pathlib

import stk
import stko
from atomistic_utilities import (
    desymm_optimisation_sequence,
    get_ligand_bb,
)
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

    buildingblocks = {
        "la": get_ligand_bb(
            path=ligand_dir / "la_prep.mol",
            optl_path=ligand_dir / "la_optl.mol",
        ),
        "las": get_ligand_bb(
            path=ligand_dir / "las_prep.mol",
            optl_path=ligand_dir / "las_optl.mol",
        ),
        "c1": get_ligand_bb(
            path=ligand_dir / "c1_prep.mol",
            optl_path=ligand_dir / "c1_optl.mol",
        ),
        "st5": get_ligand_bb(
            path=ligand_dir / "st5_prep.mol",
            optl_path=ligand_dir / "st5_optl.mol",
        ),
        "pd": stk.BuildingBlock(
            smiles="[Pd+2]",
            functional_groups=(
                stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
            ),
            position_matrix=[[0, 0, 0]],
        ),
    }

    pairs = (("la", "st5"), ("la", "c1"), ("las", "st5"))
    topology_graphs = {
        "3P6": stk.cage.M3L6,
        # "4P8": scram.topologies.CGM4L8,  # noqa: ERA001
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
                        buildingblocks[bb1]: (4, 9, 10, 11),
                        buildingblocks[bb2]: (5, 6, 7, 8),
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

            cage_molecule = desymm_optimisation_sequence(
                mol=cage_molecule,
                name=name,
                charge=charge,
                calc_dir=calculation_dir,
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
            )
            cage_molecule = cage_molecule.with_centroid((0, 0, 0))
            cage_molecule.write(optc_file)


if __name__ == "__main__":
    main()
