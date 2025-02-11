"""Script to generate and optimise CG models."""

import logging
import pathlib

import cgexplore as cgx
import networkx as nx
import numpy as np
import rustworkx as rx
import stko
from rdkit import RDLogger

from model_enumeration.mgen_generation import (
    get_bb_topology_code_graph,
    get_stk_topology_code,
)
from model_enumeration.mgen_utilities import (
    StericTwoC1Arm,
    a2bead_d,
    abead_c,
    abead_d,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "gtest_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "gtest_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "gtest_ligands"
    ligand_dir.mkdir(exist_ok=True)

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        # From prep.
        "lf": {
            "egb": 120,
            "deg": 180,
            "dd": 8.0,
            "de": 4.3,
            "dde": 133,
            "eg": 1.4,
            "gb": 1.4,
        },
        # From optl.
        "l2": {"ba": 2.8, "aa": 4.9, "bac": 150, "s": 0.0},
    }

    ligand_types = {"lf": "sixbead", "l2": "twoarm"}

    pairs_to_predict = [("lf", "l2")]

    pairs = {}
    for large, small in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        else:
            msg = large
            raise NotImplementedError(msg)

        if ligand_types[small] == "twoarm":
            small_prec = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
        elif ligand_types[small] == "stwoarm":
            small_prec = StericTwoC1Arm(
                bead=cbead_d, abead1=abead_d, steric_bead=steric_bead
            )
        elif ligand_types[small] == "sixbead":
            small_prec = cgx.molecular.SixBead(
                bead=c2bead_d,
                abead1=a2bead_d,
                abead2=e2bead_d,
            )
        else:
            msg = small
            raise NotImplementedError(msg)

        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "multipliers": (1, 2, 3, 4),
            "vdw_cutoff": 2,
        }

    for pair in pairs:
        large_name = pairs[pair]["large_name"]
        small_name = pairs[pair]["small_name"]

        large = pairs[pair]["large"]
        small = pairs[pair]["small"]
        tetra = pairs[pair]["tetra"]

        forcefield = precursors_to_forcefield(
            pair=pair,
            large=large,
            small=small,
            large_meas=ligand_measures[large_name],
            small_meas=ligand_measures[small_name],
            vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
        )

        small_bb = cgx.utilities.optimise_ligand(
            molecule=small.get_building_block(),
            name=f"{pair}_{small.get_name()}",
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        ).clone()
        small_bb.write(str(ligand_dir / f"{pair}_{small.get_name()}_optl.mol"))

        tetra_bb = cgx.utilities.optimise_ligand(
            molecule=tetra.get_building_block(),
            name=tetra.get_name(),
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        ).clone()
        tetra_bb.write(str(ligand_dir / f"{tetra.get_name()}_optl.mol"))

        large_bb = cgx.utilities.optimise_ligand(
            molecule=large.get_building_block(),
            name=f"{pair}_{large.get_name()}",
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        ).clone()
        large_bb.write(str(ligand_dir / f"{pair}_{large.get_name()}_optl.mol"))

        for multiplier in pairs[pair]["multipliers"]:
            logging.info("doing: pair %s, multi %s", pair, multiplier)
            # Define a connectivity based on a multiplier.
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    tetra_bb: stoichiometry_l_l_m[2] * multiplier,
                    large_bb: stoichiometry_l_l_m[0] * multiplier,
                    small_bb: stoichiometry_l_l_m[1] * multiplier,
                },
                graph_type=f"{1 * multiplier}P{2 * multiplier}",
                graph_set="rx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )

            possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                iterator=iterator
            )
            logging.info(
                "building block iteration has %s options",
                len(possible_bbdicts),
            )

            logging.info(
                "producing between %s and %s structures",
                len(possible_bbdicts) * iterator.count_graphs() * 1,
                len(possible_bbdicts) * iterator.count_graphs() * 4,
            )

            # Use known topology codes.
            stk_topology_code, stk_positions = get_stk_topology_code(
                graph_type=f"{1 * multiplier}P{2 * multiplier}",
            )
            sidx = -1
            midx = 0
            run_topology_codes = []
            for bb_config in possible_bbdicts:
                name = f"{pair}_{multiplier}_{sidx}_{midx}_b{bb_config.idx}"

                # Testing bb-config aware graph check.
                # Convert TopologyCode to a graph.
                current_graph = get_bb_topology_code_graph(
                    topology_code=stk_topology_code,
                    bb_config=bb_config,
                )

                # Check that graph for isomorphism with others graphs.
                passed_iso = True
                for tc, bc in run_topology_codes:
                    test_graph = get_bb_topology_code_graph(
                        topology_code=tc, bb_config=bc
                    )

                    if rx.is_isomorphic(
                        current_graph,
                        test_graph,
                        node_matcher=lambda x, y: x.split("-")[1]
                        == y.split("-")[1],
                    ):
                        passed_iso = False
                        break

                if not passed_iso:
                    continue
                run_topology_codes.append((stk_topology_code, bb_config))

                # Do the construction.
                constructed_molecule = cgx.scram.try_except_construction(
                    iterator=iterator,
                    topology_code=stk_topology_code,
                    building_block_configuration=bb_config,
                    vertex_positions={
                        nidx: np.array(stk_positions[nidx]) * 10
                        for nidx in stk_topology_code.get_nx_graph().nodes
                    },
                )

                constructed_molecule.write(structure_dir / f"{name}_unopt.mol")
                stko_graph = stko.Network.init_from_molecule(
                    constructed_molecule
                )
                gg = nx.kamada_kawai_layout(stko_graph.get_graph(), dim=3)
                constructed_molecule = (
                    constructed_molecule.with_position_matrix(
                        np.array([gg[i] for i in gg]) * 10
                    )
                )
                constructed_molecule.write(structure_dir / f"{name}_graph.mol")


if __name__ == "__main__":
    main()
