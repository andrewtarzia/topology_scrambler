"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
from copy import deepcopy

import cgexplore
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger

from min_utilities import (
    FiveBead,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    ebead_d,
    optimise_cage,
    prepare_building_block,
    save_vertex_positions,
    tetra_bead,
)
from topologies import TopologyIterator
from utilities import extract_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(database_path, name, forcefield, iterator, topology_code):
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


def make_plot(pair, database_path, structure_dir, figure_dir, filename):
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
                # s=40,
                # alpha=0.3,
                # ec="none",
                label=f"{multi}: {round(min_energy[0],3)} @ {min_energy[1]}",
            )

            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.axhline(y=0.3, c="k", ls="--")
    ax.legend(ncols=1, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        filename,
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


def main() -> None:  # noqa: PLR0915
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

    atomistic_calculation_dir = wd / "calculations"
    atomistic_ligand_dir = wd / "ligands"

    database_path = data_dir / "rerun.db"

    # Define bead libraries.
    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        ebead_d,
        binder_bead,
        tetra_bead,
    )
    cgexplore.molecular.BeadLibrary(present_beads)

    pairs = {
        "la_c1": {
            "converging_name": "la",
            "diverging_name": "c1",
            "converging_confid": 3,
            "diverging_confid": 1,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": FiveBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "lf_ls1": {
            "converging_name": "lf",
            "diverging_name": "ls1",
            "converging_confid": None,
            "diverging_confid": None,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": FiveBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        },
        "lf_ls9": {
            "converging_name": "lf",
            "diverging_name": "ls9",
            "converging_confid": None,
            "diverging_confid": None,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": FiveBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3, 4),
        },
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "converging_confid": None,
            "diverging_confid": None,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": FiveBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": FiveBead(
                bead=cbead_d,
                abead1=abead_d,
                abead2=ebead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "converging_confid": None,
            "diverging_confid": None,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": FiveBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": FiveBead(
                bead=cbead_d,
                abead1=abead_d,
                abead2=ebead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
    }

    for pair in pairs:
        converging_name = pairs[pair]["converging_name"]
        diverging_name = pairs[pair]["diverging_name"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        converging_confid = pairs[pair]["converging_confid"]
        diverging_confid = pairs[pair]["diverging_confid"]
        if converging_confid is None or diverging_confid is None:
            raise RuntimeError
        tetra = pairs[pair]["tetra"]

        constant_definer_dict = {
            # Bonds.
            "mb": ("bond", 1.0, 1e5),
            "ab": ("bond", 1.0, 1e5),
            "gb": ("bond", 0.7, 1e5),
            "fb": ("bond", 0.7, 1e5),
            "eg": ("bond", 0.7, 1e5),
            "af": ("bond", 0.7, 1e5),
            # Angles.
            "bmb": ("pyramid", 90, 1e2),
            "mba": ("angle", 180, 1e2),
            "mbg": ("angle", 180, 1e2),
            "mbf": ("angle", 180, 1e2),
            "aca": ("angle", 180, 1e2),
            "ede": ("angle", 180, 1e2),
            # Torsions.
            # Nonbondeds.
            "m": ("nb", 10.0, 1.0),
            "d": ("nb", 10.0, 1.0),
            "e": ("nb", 10.0, 1.0),
            "a": ("nb", 10.0, 1.0),
            "b": ("nb", 10.0, 1.0),
            "c": ("nb", 10.0, 1.0),
            "g": ("nb", 10.0, 1.0),
            "f": ("nb", 10.0, 1.0),
        }
        definer_dict = deepcopy(constant_definer_dict)

        conf_data = extract_ensemble(
            molecule=stk.BuildingBlock.init_from_file(
                path=atomistic_ligand_dir / f"{converging_name}_optl.mol",
                functional_groups=(
                    stko.functional_groups.ThreeSiteFactory(
                        "[#6]~[#7X2]~[#6]"
                    ),
                ),
            ),
            crest_run=atomistic_calculation_dir / f"{converging_name}_crest",
        )[converging_confid]

        if isinstance(converging, FiveBead):
            bead1, bead2, bead3 = list(converging.get_bead_set())

            # This angle is constant.
            deg = 120
            # Therefore, from trapezoid, so is this:
            egg = (360 - deg * 2) / 2
            definer_dict[f"{bead3}{bead2}{bead1}"] = ("angle", deg, 1e2)

            bge = sum(conf_data["binder_angles"]) / 2
            # Scale to cg, divide by 2.
            bb = conf_data["binder_distance"] / 2
            gb = definer_dict["gb"][1]
            if bge / 2 < 90:  # noqa: PLR2004
                gg = bb + (2 * gb * np.cos(np.radians(bge / 2)))
            else:
                theta = bge - 90 - egg
                gg = bb - (2 * gb * np.sin(np.radians(theta)))

            eg = definer_dict["eg"][1]
            ee = gg - (2 * eg * np.sin(np.radians(deg - 90)))
            ed = ee / 2

            definer_dict[f"{bead2}{bead1}"] = ("bond", ed, 1e5)
            definer_dict[f"b{bead3}{bead2}"] = ("angle", bge, 1e2)

            # Define torsion as restricted.
            tname = f"{bead3}{bead2}{bead1}{bead2}{bead3}"
            definer_dict[tname] = ("tors", "0134", 180, 50, 1)

        elif isinstance(converging, cgexplore.molecular.TwoC1Arm):
            bead1, bead2 = list(converging.get_bead_set())
            if bead1 != "c" or bead2 != "a":
                raise NotImplementedError
            # Define bac by the average of the binder angles.
            bac = sum(conf_data["binder_angles"]) / 2
            definer_dict[f"b{bead2}{bead1}"] = ("angle", bac, 1e2)

            # Defined by binder-binder distance /2 .
            bb = conf_data["binder_distance"] / 2
            ba = definer_dict["ab"][1]
            aa = bb - (2 * ba * np.sin(np.radians(bac - 90)))
            ac = aa / 2
            definer_dict[f"{bead2}{bead1}"] = ("bond", ac, 1e5)

            # Define torsion as restricted.
            tname = f"b{bead2}{bead1}{bead2}b"
            definer_dict[tname] = ("tors", "0134", 180, 50, 1)

        conf_data = extract_ensemble(
            molecule=stk.BuildingBlock.init_from_file(
                path=atomistic_ligand_dir / f"{diverging_name}_optl.mol",
                functional_groups=(
                    stko.functional_groups.ThreeSiteFactory(
                        "[#6]~[#7X2]~[#6]"
                    ),
                ),
            ),
            crest_run=atomistic_calculation_dir / f"{diverging_name}_crest",
        )[diverging_confid]

        if isinstance(diverging, FiveBead):
            bead1, bead2, bead3 = list(diverging.get_bead_set())

            # This angle is constant.
            deg = 120
            # Therefore, from trapezoid, so is this:
            egg = (360 - deg * 2) / 2
            definer_dict[f"{bead3}{bead2}{bead1}"] = ("angle", deg, 1e2)

            bge = sum(conf_data["binder_angles"]) / 2
            # Scale to cg, divide by 2.
            bb = conf_data["binder_distance"] / 2
            gb = definer_dict["gb"][1]
            if bge / 2 < 90:  # noqa: PLR2004
                gg = bb + (2 * gb * np.cos(np.radians(bge / 2)))
            else:
                theta = bge - 90 - egg
                gg = bb - (2 * gb * np.sin(np.radians(theta)))

            eg = definer_dict["eg"][1]
            ee = gg - (2 * eg * np.sin(np.radians(deg - 90)))
            ed = ee / 2

            definer_dict[f"{bead2}{bead1}"] = ("bond", ed, 1e5)
            definer_dict[f"b{bead3}{bead2}"] = ("angle", bge, 1e2)

            # Define torsion as restricted.
            tname = f"{bead3}{bead2}{bead1}{bead2}{bead3}"
            definer_dict[tname] = ("tors", "0134", 180, 50, 1)

        elif isinstance(diverging, cgexplore.molecular.TwoC1Arm):
            bead1, bead2 = list(diverging.get_bead_set())
            if bead1 != "c" or bead2 != "a":
                raise NotImplementedError
            # Define bac by the average of the binder angles.
            bac = sum(conf_data["binder_angles"]) / 2
            definer_dict[f"b{bead2}{bead1}"] = ("angle", bac, 1e2)

            # Defined by binder-binder distance /2 .
            bb = conf_data["binder_distance"] / 2
            aa = bb - (2 * np.sin(np.radians(bac - 90)))
            ac = aa / 2
            definer_dict[f"{bead2}{bead1}"] = ("bond", ac, 1e5)

            # Define torsion as restricted.
            tname = f"b{bead2}{bead1}{bead2}b"
            definer_dict[tname] = ("tors", "0134", 180, 50, 1)

        forcefield = cgexplore.systems_optimisation.get_forcefield_from_dict(
            identifier=f"{pair}ff",
            prefix=f"{pair}ff",
            vdw_bond_cutoff=2,
            present_beads=present_beads,
            definer_dict=definer_dict,
        )

        converging_bb = prepare_building_block(
            precursor=converging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        diverging_bb = prepare_building_block(
            precursor=diverging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        tetra_bb = prepare_building_block(
            precursor=tetra,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )

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
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                count = 0
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
                    break

            _ = make_plot(
                database_path=database_path,
                pair=pair,
                structure_dir=structure_dir,
                figure_dir=figure_dir,
                filename=figure_dir / f"rerun_1_{pair}.png",
            )
            raise SystemExit


if __name__ == "__main__":
    main()
