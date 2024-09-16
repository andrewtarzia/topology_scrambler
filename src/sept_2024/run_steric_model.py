"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore
import stko
from openmm import OpenMMException
from rdkit import RDLogger
from run_cg_model import analyse_cage, make_plot
from utilities import (
    StericSixBead,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    save_vertex_positions,
    steric_bead,
    tetra_bead,
)

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "steric_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "steric_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "steric_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "steric_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "steric.db"

    ligand_measures = {
        "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4, "s": 2},
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
        "c12": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 120},
        "c13": {"ba": 2.8, "aa": 3.4, "bac": 100, "bacab": 180},
        "c14": {"ba": 2.8, "aa": 3.4, "bac": 110, "bacab": 180},
        "c15": {"ba": 2.8, "aa": 3.4, "bac": 120, "bacab": 180},
    }

    pairs = {
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1,),
        },
        "la_st5_11": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (4,),
        },
        "la_st52_11": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (4,),
        },
    }

    for pair in pairs:
        converging_name = pairs[pair]["converging_name"]
        diverging_name = pairs[pair]["diverging_name"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]

        forcefield = precursors_to_forcefield(
            pair=pair,
            diverging=diverging,
            converging=converging,
            conv_meas=ligand_measures[converging_name],
            dive_meas=ligand_measures[diverging_name],
        )

        converging_bb = scram.toy.prepare_building_block(
            precursor=converging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        diverging_bb = scram.toy.prepare_building_block(
            precursor=diverging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        tetra_bb = scram.toy.prepare_building_block(
            precursor=tetra,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        for multiplier in pairs[pair]["multipliers"]:
            if args.run:
                # Define a connectivity based on a multiplier.
                iterator = scram.topologies.TopologyIterator(
                    multiplier=multiplier,
                    stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                    tetra_bb=tetra_bb,
                    converging_bb=converging_bb,
                    diverging_bb=diverging_bb,
                )
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    name = f"{pair}_{multiplier}_{idx}"
                    acage.write(structure_dir / f"{name}_unopt.mol")

                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )
                    if num_components != 1:
                        continue

                    # Optimise and save.
                    logging.info("building %s", name)

                    try:
                        conformer = scram.toy.optimise_cage(
                            molecule=acage,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        if conformer is not None:
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        save_vertex_positions(
                            name=name,
                            calculation_dir=calculation_dir,
                            structure_dir=structure_dir,
                            molecule=acage,
                        )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=constructed.topology_code,
                        )

                    except OpenMMException:
                        pass

            _ = make_plot(
                database_path=database_path,
                pair=pair,
                structure_dir=structure_dir,
                figure_dir=figure_dir,
                filename=f"sterics_1_{pair}.png",
            )


if __name__ == "__main__":
    main()
