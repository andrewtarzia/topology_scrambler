"""Script to generate and optimise CG models."""

import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from rdkit import RDLogger
import stko
import argparse
import warnings

from min_utilities import (
    binder_bead,
    abead_d,
    cbead_d,
    eb_str,
    tetra_bead,
    optimise_cage,
    get_forcefield,
)
import cgexplore
from topologies import HomolepticTopologyIterator
from openmm import openmm, OpenMMException
import mchammer as mch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


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
                "ligand": name.split("_")[0],
                "num_components": num_components,
                "multiplier": name.split("_")[1],
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
            },
        )


def get_validation_forcefield(
    bac_angle: float,
    identifier: str,
) -> cgexplore.forcefields.ForceField:  # noqa: C901
    """Get forcefield."""

    present_beads = (cbead_d, abead_d, binder_bead, tetra_bead)
    definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        "ab": ("bond", 1.0, 1e5),
        "ac": ("bond", 1.5, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "bac": ("angle", bac_angle, 1e2),
        # Torsions.
        "bacab": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }
    return get_forcefield(
        identifier=identifier,
        prefix="min_val",
        present_beads=present_beads,
        vdw_bond_cutoff=2,
        definer_dict=definer_dict,
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


def make_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
):
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
        "6": "tab:purple",
        "8": "tab:pink",
        "10": "tab:cyan",
        "12": "tab:brown",
    }
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 5))
    energies = {}
    bacs = {}
    energies2 = {}
    bacs2 = {}
    for entry in cgexplore.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        bite_angle = (bac_angle - 90) * 2

        if len(entry.key.split("_")) == 3:
            if multi not in energies:
                energies[multi] = []

            if bite_angle not in bacs:
                bacs[bite_angle] = []

            if entry.properties["num_components"] > 1:
                continue

            energies[multi].append((bite_angle, energy))
            bacs[bite_angle].append((multi, energy, entry.key))
        elif len(entry.key.split("_")) == 5:
            if multi not in energies2:
                energies2[multi] = []

            if bite_angle not in bacs2:
                bacs2[bite_angle] = []

            if entry.properties["num_components"] > 1:
                continue

            energies2[multi].append((bite_angle, energy))
            bacs2[bite_angle].append((multi, energy, entry.key))

    for multi in sorted([int(i) for i in energies]):
        idx = str(multi)
        min_energy = min(energies[idx], key=lambda p: p[1])

        axs[0].scatter(
            [i[0] for i in energies[idx]],
            [i[1] for i in energies[idx]],
            marker="o",
            c=cmap[idx],
            s=20,
            alpha=0.2,
            ec="none",
            label=(
                f"M{idx}: {round(min_energy[1],2)} @ {min_energy[0]} "
                f"({len(energies[idx])})"
            ),
        )

        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

        axs[0].plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=cmap[idx],
            ls="--",
            alpha=0.4,
        )

    with (figure_dir / "val_opt.txt").open("w") as f:
        bac_line = []
        for bac_angle in sorted(bacs):
            min_energy = min(bacs[bac_angle], key=lambda p: p[1])
            opt_file = structure_dir / f"{min_energy[2]}_optc.mol"
            f.write(f"{opt_file} ")
            bac_line.append((bac_angle, min_energy[1]))

            axs[0].scatter(
                bac_angle,
                min_energy[1],
                c=cmap[min_energy[0]],
                marker="o",
                s=60,
                ec="k",
            )

    axs[0].plot(
        [i[0] for i in bac_line],
        [i[1] for i in bac_line],
        c="k",
        zorder=-1,
    )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("target bite angle [deg]", fontsize=16)
    axs[0].set_ylabel(eb_str(), fontsize=16)
    axs[0].set_yscale("log")
    axs[0].axhline(y=0.3, c="k", ls="--")
    leg = axs[0].legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    for multi in sorted([int(i) for i in energies2]):
        idx = str(multi)
        min_energy = min(energies2[idx], key=lambda p: p[1])

        axs[1].scatter(
            [i[0] for i in energies2[idx]],
            [i[1] for i in energies2[idx]],
            marker="o",
            c=cmap[idx],
            s=20,
            alpha=1,
            ec="none",
            label=(
                f"M{idx}: {round(min_energy[1],2)} @ {min_energy[0]} "
                f"({len(energies2[idx])})"
            ),
        )

        bac_line = []
        for bac_angle in sorted(bacs2):
            rel_energies = [i[1] for i in energies2[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

        axs[1].plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=cmap[idx],
            ls="--",
            alpha=0.4,
        )

    bac_line2 = []
    for bac_angle in sorted(bacs2):
        min_energy = min(bacs2[bac_angle], key=lambda p: p[1])
        opt_file = structure_dir / f"{min_energy[2]}_optc.mol"
        bac_line2.append((bac_angle, min_energy[1]))

        axs[1].scatter(
            bac_angle,
            min_energy[1],
            c=cmap[min_energy[0]],
            marker="o",
            s=60,
            ec="k",
        )

    axs[1].plot(
        [i[0] for i in bac_line2],
        [i[1] for i in bac_line2],
        c="k",
        zorder=-1,
    )

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("target bite angle [deg]", fontsize=16)
    axs[1].set_ylabel(eb_str(), fontsize=16)
    axs[1].set_yscale("log")
    axs[1].axhline(y=0.3, c="k", ls="--")
    leg = axs[1].legend(ncols=1, fontsize=12)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "val_1.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main():
    raise SystemExit(
        "**paused this for now, I know it can work, but it is a matter of "
        "thinking about the algorithm. Can revisit when we take this further**"
    )
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "val_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "val_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "val_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "val_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "val_run.db"

    ligands = {
        str(i): {
            "forcefield": get_validation_forcefield(
                bac_angle=bac_angle, identifier=str(i)
            ),
            "stoichiometry_L_M": (2, 1),
            "ditopic": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 3, 4, 6, 8, 10, 12),
        }
        for i, bac_angle in enumerate(range(85, 181, 5))
    }

    if args.run:
        for lig in ligands:
            forcefield = ligands[lig]["forcefield"]
            ditopic = ligands[lig]["ditopic"]
            tetra = ligands[lig]["tetra"]
            # Prepare ligands.
            for i, precursor in enumerate((ditopic, tetra)):
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
                    ditopic_bb = building_block.clone()
                elif i == 1:
                    tetra_bb = building_block.clone()

            for multiplier in ligands[lig]["multipliers"]:
                # if lig not in ("9", "10", "11"):
                #     continue
                # if multiplier not in (6,):
                #     continue

                # if lig not in ("12", "13", "14", "19"):
                #     continue
                # if multiplier not in (12,):
                #     continue

                if lig not in ("5",):
                    continue
                if multiplier not in (3,):
                    continue

                # if lig not in ("7",):
                #     continue
                # if multiplier not in (4,):
                #     continue

                vmap_str_map = {}
                cut_energy = 0.3
                current_energy = 1e24
                seeds = (100, 32963, 399, 9)
                for seed in seeds:
                    if current_energy < cut_energy:
                        logging.info(f"breaking before seed {seed}")
                        break
                    rng = np.random.default_rng(seed=seed)

                    # Define a connectivity based on a multiplier.
                    iterator = HomolepticTopologyIterator(
                        multiplier=multiplier,
                        stoichiometry=ligands[lig]["stoichiometry_L_M"],
                        tetra_bb=tetra_bb,
                        ditopic_bb=ditopic_bb,
                    )
                    logging.info(f"doing: ligand {lig}, multi {multiplier}")
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
                            logging.info(f"fail for {scramble_step}")
                            continue

                        if current_energy < cut_energy:
                            logging.info(f"breaking at step {scramble_step}")
                            break

                        mash_step = 0
                        acage = new_constructed.constructed_molecule
                        new_topology_code = new_constructed.topology_code
                        # Avoid redoing same TGs.
                        if new_topology_code.as_string in vmap_str_map:
                            name = vmap_str_map[new_topology_code.as_string]
                            name = name.split("_")
                            name[3] = str(mash_step)
                            name[4] = str(seed)
                            name = "_".join(name)
                        else:
                            name = (
                                f"{lig}_{multiplier}_{scramble_step}_{mash_step}_{seed}"
                            )
                            vmap_str_map[new_topology_code.as_string] = name

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
                            beta=iterator.get_beta(),
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
                                name[3] = str(mash_step)
                                name[4] = str(seed)
                                name = "_".join(name)
                            else:
                                name = f"{lig}_{multiplier}_{scramble_step}_{mash_step}_{seed}"
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
                                beta=iterator.get_beta(),
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

                raise SystemExit("figure out hwo to atomisse")
                continue

                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    # Initialise positions based on that connectivity.
                    name = f"{lig}_{multiplier}_{idx}"
                    logging.info(f"building {name}")
                    mid = stko.molecule_analysis.GeometryAnalyser().get_min_centroid_distance(
                        acage
                    )
                    if mid < 16:
                        continue
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
                        )

                    except OpenMMException:
                        pass

    make_plot(
        figure_dir=figure_dir,
        structure_dir=structure_dir,
        database_path=database_path,
    )


if __name__ == "__main__":
    main()
