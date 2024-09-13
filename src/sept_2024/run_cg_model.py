"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore
import matplotlib.pyplot as plt
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger

import desymmetrised_scripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901
    database_path: pathlib.Path,
    name: str,
    forcefield: cgexplore.forcefields.ForceField,
    iterator: desymmetrised_scripts.topologies.TopologyIterator,
    topology_code: desymmetrised_scripts.topologies.TopologyCode,
) -> None:
    """Analyse a toy model cage."""
    database = cgexplore.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "topology_code_vmap" not in properties:
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
            sum(
                energy_decomp[i]
                for i in energy_decomp
                if "total energy" not in i
            )
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
                openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(
                openmm.unit.angstrom
            )

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(
                    openmm.unit.degrees
                )
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
            v_dict[at.bead_class] = at.sigma.value_in_unit(
                openmm.unit.angstrom
            )
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
                "energy_per_bb": fin_energy
                / iterator.get_num_building_blocks(),
                "pair": name.split("_")[0] + "_" + name.split("_")[1],
                "num_components": num_components,
                "multiplier": name.split("_")[2],
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
            },
        )


def make_plot(
    pair: str,
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
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
        energies[multi].append((round(energy, 4), entry.key))

    with (figure_dir / f"min_{pair}.txt").open("w") as f:
        for multi in energies:
            if len(energies[multi]) == 0:
                continue

            sorted_energies = sorted(energies[multi], key=lambda p: p[0])
            min_energy = sorted_energies[0]

            ax.plot(
                [i[0] for i in energies[multi]],
                marker="o",
                c=cmap[multi],
                markersize=4,
                label=f"{multi}: {round(min_energy[0],3)} @ {min_energy[1]}",
            )

            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(desymmetrised_scripts.toy.eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.axhline(y=0.3, c="k", ls="--")
    ax.legend(ncols=1, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
    return energies


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
    calculation_dir = wd / "rerun_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "rerun_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "rerun_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "rerun_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "rerun.db"

    ligand_measures = {
        "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4},
        "lf": {"dd": 8.0, "de": 4.3, "dde": 133, "eg": 1.4, "gb": 1.4},
        "ls1": {"ba": 2.8, "aa": 4.9, "bac": 150, "bacab": 180},
        "ls9": {"ba": 2.8, "aa": 5.5, "bac": 165, "bacab": 180},
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
        "c12": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 120},
        "c13": {"ba": 2.8, "aa": 3.4, "bac": 100, "bacab": 180},
        "c14": {"ba": 2.8, "aa": 3.4, "bac": 110, "bacab": 180},
        "c15": {"ba": 2.8, "aa": 3.4, "bac": 120, "bacab": 180},
    }

    pairs = {
        "la_c1": {
            "converging_name": "la",
            "diverging_name": "c1",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_c12": {
            "converging_name": "la",
            "diverging_name": "c12",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_c13": {
            "converging_name": "la",
            "diverging_name": "c13",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_c14": {
            "converging_name": "la",
            "diverging_name": "c14",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_c15": {
            "converging_name": "la",
            "diverging_name": "c15",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "lf_ls1": {
            "converging_name": "lf",
            "diverging_name": "ls1",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        },
        "lf_ls9": {
            "converging_name": "lf",
            "diverging_name": "ls9",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        },
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": desymmetrised_scripts.toy.SixBead(
                bead=desymmetrised_scripts.toy.cbead_c,
                abead1=desymmetrised_scripts.toy.abead_c,
                abead2=desymmetrised_scripts.toy.ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=desymmetrised_scripts.toy.cbead_d,
                abead1=desymmetrised_scripts.toy.abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=desymmetrised_scripts.toy.tetra_bead,
                abead1=desymmetrised_scripts.toy.binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
    }

    for pair in pairs:
        converging_name = pairs[pair]["converging_name"]
        diverging_name = pairs[pair]["diverging_name"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]

        forcefield = desymmetrised_scripts.toy.precursors_to_forcefield(
            pair=pair,
            diverging=diverging,
            converging=converging,
            conv_meas=ligand_measures[converging_name],
            dive_meas=ligand_measures[diverging_name],
        )

        converging_bb = desymmetrised_scripts.toy.prepare_building_block(
            precursor=converging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        diverging_bb = desymmetrised_scripts.toy.prepare_building_block(
            precursor=diverging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        tetra_bb = desymmetrised_scripts.toy.prepare_building_block(
            precursor=tetra,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )

        for multiplier in pairs[pair]["multipliers"]:
            if args.run:
                # Define a connectivity based on a multiplier.
                iterator = desymmetrised_scripts.topologies.TopologyIterator(
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
                        conformer = desymmetrised_scripts.toy.optimise_cage(
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

                        desymmetrised_scripts.toy.save_vertex_positions(
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
                filename=f"rerun_1_{pair}.png",
            )


if __name__ == "__main__":
    main()
