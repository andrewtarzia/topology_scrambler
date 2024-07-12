"""Script to generate and optimise CG models."""

import logging
import pathlib
import matplotlib.pyplot as plt
from rdkit import RDLogger
import stko
import argparse
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


def analyse_cage(database_path, name, forcefield, iterator):
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
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "min_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "min_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "min_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "min_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "min_run.db"

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

        for multiplier in pairs[pair]["multipliers"]:
            if args.run:
                # Define a connectivity based on a multiplier.
                iterator = TopologyIterator(
                    multiplier=multiplier,
                    stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                    tetra_bb=tetra_bb,
                    converging_bb=converging_bb,
                    diverging_bb=diverging_bb,
                )
                logging.info(f"doing: pair {pair}, multi {multiplier}")
                count = 0
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    # Initialise positions based on that connectivity.
                    name = f"{pair}_{multiplier}_{idx}"
                    acage.write(str(structure_dir / f"{name}_unopt.mol"))

                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )

                    if num_components != 1:
                        continue

                    # Optimise and save.
                    logging.info(f"building {name}")
                    count += 1

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
                        )

                    except OpenMMException:
                        pass
                logging.info(f"for: pair {pair}, multi {multiplier}, built {count}!")

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
            ax.legend(ncols=2, fontsize=16)
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"min_1_{pair}.png",
                dpi=360,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    main()
