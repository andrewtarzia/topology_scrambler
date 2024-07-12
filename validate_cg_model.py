"""Script to generate and optimise CG models."""

import logging
import pathlib
import matplotlib.pyplot as plt
from rdkit import RDLogger
import stko

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
                "ligand": name.split("_")[0],
                "num_components": num_components,
                "multiplier": name.split("_")[1],
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
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = {}
    bacs = {}
    for entry in cgexplore.utilities.AtomliteDatabase(database_path).get_entries():
        multi = entry.properties["multiplier"]
        if multi not in energies:
            energies[multi] = []

        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        bite_angle = (bac_angle - 90) * 2
        if bite_angle not in bacs:
            bacs[bite_angle] = []
        if entry.properties["num_components"] > 1:
            continue
        energies[multi].append((bite_angle, energy))
        bacs[bite_angle].append((multi, energy, entry.key))

    for multi in energies:
        min_energy = min(energies[multi], key=lambda p: p[1])
        ax.scatter(
            [i[0] for i in energies[multi]],
            [i[1] for i in energies[multi]],
            marker="o",
            c=cmap[multi],
            s=20,
            alpha=0.1,
            ec="none",
            label=f"{multi}: {round(min_energy[1],2)} @ {min_energy[0]}",
        )

        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[multi] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

        ax.plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c=cmap[multi],
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
            ax.scatter(
                bac_angle,
                min_energy[1],
                c=cmap[min_energy[0]],
                marker="o",
                s=60,
                ec="k",
            )
    ax.plot(
        [i[0] for i in bac_line],
        [i[1] for i in bac_line],
        c="k",
        zorder=-1,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("target bite angle [deg]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.axhline(y=0.3, c="k", ls="--")
    ax.legend(ncols=2, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "val_1.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main():
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

                # Define a connectivity based on a multiplier.
                iterator = HomolepticTopologyIterator(
                    multiplier=multiplier,
                    stoichiometry=ligands[lig]["stoichiometry_L_M"],
                    tetra_bb=tetra_bb,
                    ditopic_bb=ditopic_bb,
                )
                logging.info(f"doing: ligand {lig}, multi {multiplier}")
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    # Initialise positions based on that connectivity.
                    name = f"{lig}_{multiplier}_{idx}"
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
