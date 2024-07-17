"""Script to generate and optimise CG models."""

import logging
import mchammer as mch
import pathlib
import matplotlib.pyplot as plt
from rdkit import RDLogger
import stko
import itertools as it
import argparse
import numpy as np
from min_utilities import (
    binder_bead,
    abead_d,
    abead_c,
    cbead_d,
    cbead_c,
    ebead_c,
    ebead_d,
    tetra_bead,
    forcefield_lf_ls1,
    forcefield_lf_ls9,
    SixBead,
    eb_str,
    forcefield_la_st52,
    forcefield_la_st5,
    optimise_cage,
)
import cgexplore
from topologies import TopologyIterator
from openmm import openmm, OpenMMException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(database_path, name, forcefield, iterator, topology_code):
    database = cgexplore.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "num_components" not in properties:
        energy_decomp = {}
        for component in properties["energy_decomposition"]:
            component_tup = properties["energy_decomposition"][component]
            if component == "total energy":
                energy_decomp[f"{component}_{component_tup[1]}"] = float(
                    component_tup[0]
                )
            else:
                just_name = component.split("'")[1]
                key = f"{just_name}_{component_tup[1]}"
                value = float(component_tup[0])
                if key in energy_decomp:
                    energy_decomp[key] += value
                else:
                    energy_decomp[key] = value
        fin_energy = energy_decomp["total energy_kJ/mol"]
        if (
            sum(energy_decomp[i] for i in energy_decomp if "total energy" not in i)
            != fin_energy
        ):
            msg = (
                "energy decompisition does not sum to total energy for"
                f" {name}: {energy_decomp}"
            )
            raise RuntimeError(msg)

        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        k_dict = {}
        v_dict = {}

        for bt in ff_targets["bonds"]:
            cp = (bt.type1, bt.type2)
            k_dict["_".join(cp)] = bt.bond_k.value_in_unit(
                openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(openmm.unit.angstrom)

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(openmm.unit.degrees)
            except TypeError:
                # Handle different angle types.
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
                v_dict["_".join(cp)] = (at.n, at.b)

        for at in ff_targets["torsions"]:
            cp = at.search_string
            k_dict["_".join(cp)] = at.torsion_k.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            v_dict["_".join(cp)] = at.phi0.value_in_unit(openmm.unit.degrees)
        for at in ff_targets["nonbondeds"]:
            v_dict[at.bead_class] = at.sigma.value_in_unit(openmm.unit.angstrom)
            k_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "k_dict": k_dict,
            "v_dict": v_dict,
        }

        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy / iterator.get_num_building_blocks(),
                "pair": name.split("_")[0] + "_" + name.split("_")[1],
                "num_components": num_components,
                "multiplier": name.split("_")[2],
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    parser.add_argument(
        "--atomise",
        action="store_true",
        help="set to build atomistic structures",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "minmc_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "minmc_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "minmc_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "minmc_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "minmc_run.db"

    # Define bead libraries.
    present_beads = (cbead_d, abead_d, cbead_c, abead_c, binder_bead, tetra_bead)
    cgexplore.molecular.BeadLibrary(beads=present_beads)

    pairs = {
        "lf_ls1": {
            "forcefield": forcefield_lf_ls1,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 3),
        },
        "lf_ls9": {
            "forcefield": forcefield_lf_ls9,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 3),
        },
        "la_st5": {
            "forcefield": forcefield_la_st5,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "forcefield": forcefield_la_st52,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 4),
        },
    }

    for pair in pairs:
        forcefield = pairs[pair]["forcefield"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]
        # Prepare ligands.
        for i, precursor in enumerate((converging, diverging, tetra)):
            name = f"{precursor.get_name()}_f{forcefield.get_identifier()}"
            building_block = cgexplore.utilities.optimise_ligand(
                molecule=precursor.get_building_block(),
                name=name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            building_block.write(str(ligand_dir / f"{name}_optl.mol"))
            if i == 0:
                converging_bb = building_block.clone()
            elif i == 1:
                diverging_bb = building_block.clone()
            elif i == 2:
                tetra_bb = building_block.clone()

        seeds = (100, 32963, 399, 9)
        current_energy = 1e24
        vmap_str_map = {}
        for multiplier, seed in it.product(pairs[pair]["multipliers"], seeds):
            if not args.run:
                break
            if current_energy < 0.3:
                logging.info(f"breaking before seed {seed}")
                break
            rng = np.random.default_rng(seed=seed)

            # Define a connectivity based on a multiplier.
            iterator = TopologyIterator(
                multiplier=multiplier,
                stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                tetra_bb=tetra_bb,
                converging_bb=converging_bb,
                diverging_bb=diverging_bb,
            )
            logging.info(f"doing: pair {pair}, multi {multiplier}")
            for scramble_step in range(iterator.get_num_scrambles()):
                logging.info(f"doing: step {scramble_step}")
                if scramble_step == 0:
                    new_constructed = iterator.get_topology(
                        input_topology_code=None,
                        generator=rng,
                    )
                    current_energy = 1e24
                    current_topology_code = None
                    current_best = None

                else:
                    new_constructed = iterator.get_topology(
                        input_topology_code=current_topology_code,
                        generator=rng,
                    )

                if new_constructed is None:
                    continue

                if current_energy < 0.3:
                    logging.info(f"breaking at step {scramble_step}")
                    break

                mash_step = 0
                acage = new_constructed.constructed_molecule
                new_topology_code = new_constructed.topology_code
                # Avoid redoing same TGs.
                if new_topology_code.as_string in vmap_str_map:
                    name = vmap_str_map[new_topology_code.as_string]
                    name = name.split("_")
                    logging.info("when you rerun, fix these names!!!")
                    name[3] = str(mash_step)
                    name[4] = str(seed)
                    name = "_".join(name)
                else:
                    name = f"{pair}_{multiplier}_{scramble_step}_{mash_step}_{seed}"
                    vmap_str_map[new_topology_code.as_string] = name

                logging.info(f"building {name}")
                acage.write(str(structure_dir / f"{name}_unopt.mol"))

                num_components = len(
                    stko.Network.init_from_molecule(acage).get_connected_components()
                )
                if num_components != 1:
                    continue
                # Optimise and save.
                try:
                    conformer = optimise_cage(
                        molecule=acage,
                        name=name,
                        output_dir=calculation_dir,
                        forcefield=forcefield,
                        platform=None,
                        database_path=database_path,
                    )
                    if conformer is not None:
                        conformer.molecule.write(
                            str(structure_dir / f"{name}_optc.mol")
                        )

                    analyse_cage(
                        database_path=database_path,
                        name=name,
                        forcefield=forcefield,
                        iterator=iterator,
                        topology_code=new_topology_code,
                    )

                except OpenMMException:
                    continue

                new_energy = (
                    cgexplore.utilities.AtomliteDatabase(database_path)
                    .get_entry(key=name)
                    .properties["energy_per_bb"]
                )

                if mch.test_move(
                    beta=10,
                    curr_pot=current_energy,
                    new_pot=new_energy,
                    generator=rng,
                ):
                    current_energy = new_energy
                    current_topology_code = new_topology_code
                    current_best = name
                    logging.info(
                        "new best %s with E: %s",
                        current_best,
                        round(current_energy, 3),
                    )

                # Scramble the vertex positions.
                for mash_step in range(1, iterator.get_num_mashes() + 1):
                    # Avoid redoing same TGs.
                    if new_topology_code.as_string in vmap_str_map:
                        name = vmap_str_map[new_topology_code.as_string]
                        name = name.split("_")
                        logging.info("when you rerun, fix these names!!!")
                        name[3] = str(mash_step)
                        name[4] = str(seed)
                        name = "_".join(name)
                    else:
                        name = f"{pair}_{multiplier}_{scramble_step}_{mash_step}_{seed}"
                        vmap_str_map[new_topology_code.as_string] = name

                    new_constructed = iterator.get_mashed_topology(
                        topology_code=new_topology_code,
                        generator=rng,
                    )
                    acage = new_constructed.constructed_molecule

                    logging.info(f"building {name}")
                    acage.write(str(structure_dir / f"{name}_unopt.mol"))
                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )
                    if num_components != 1:
                        continue
                    # Optimise and save.
                    try:
                        conformer = optimise_cage(
                            molecule=acage,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        if conformer is not None:
                            conformer.molecule.write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=new_topology_code,
                        )

                    except OpenMMException:
                        continue

                    new_energy = (
                        cgexplore.utilities.AtomliteDatabase(database_path)
                        .get_entry(key=name)
                        .properties["energy_per_bb"]
                    )

                    if mch.test_move(
                        beta=10,
                        curr_pot=current_energy,
                        new_pot=new_energy,
                        generator=rng,
                    ):
                        current_energy = new_energy
                        current_topology_code = new_topology_code
                        current_best = name
                        logging.info(
                            "new best %s with E: %s",
                            current_best,
                            round(current_energy, 3),
                        )

            fig, ax = plt.subplots(figsize=(8, 5))
            energies = {}

            cmap = {
                "1": "tab:blue",
                "2": "tab:orange",
                "3": "tab:green",
                "4": "tab:red",
            }
            for entry in cgexplore.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if pair != entry.properties["pair"]:
                    continue
                multi = entry.properties["multiplier"]
                energy = entry.properties["energy_per_bb"]

                if multi not in energies:
                    energies[multi] = []

                if entry.properties["num_components"] > 1:
                    continue
                energies[multi].append((energy, entry.key))

            with (figure_dir / f"min_{pair}.txt").open("w") as f:
                for multi in energies:
                    if len(energies[multi]) == 0:
                        continue

                    sorted_energies = sorted(energies[multi], key=lambda p: p[0])
                    min_energy = sorted_energies[0]

                    ax.plot(
                        [i[0] for i in sorted_energies],
                        marker="o",
                        c=cmap[multi],
                        markersize=4,
                        # s=40,
                        # alpha=0.3,
                        # ec="none",
                        label=f"{multi}: {round(min_energy[0],2)} @ {min_energy[1]}",
                    )

                    opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
                    f.write(f"{opt_file} ")

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_ylabel(eb_str(), fontsize=16)
            ax.set_yscale("log")
            ax.axhline(y=0.3, c="k", ls="--")
            ax.legend(ncols=1, fontsize=16)
            ax.set_title("by MC", fontsize=16)
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"min_2_{pair}.png",
                dpi=360,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    main()
