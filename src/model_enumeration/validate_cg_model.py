"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
import warnings
from collections import defaultdict

import cgexplore
import matplotlib.pyplot as plt
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from utilities import abead_d, binder_bead, cbead_d, eb_str, tetra_bead

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

multi_cmap = {
    "1": "#ff0000",
    "2": "#fe800a",
    "3": "#fffe04",
    "4": "#7cff17",
    "5": "#07ff1a",
    "6": "#00ff7e",
    "7": "#0cfeff",
    "8": "#107eff",
    "9": "#0014ff",
    "10": "#7f00ff",
    "11": "#fe00fd",
    "12": "#ff0080",
}


def analyse_cage(  # noqa: C901
    database_path: pathlib.Path,
    name: str,
    forcefield: cgexplore.forcefields.ForceField,
    iterator: scram.topologies.TopologyIterator,
    topology_code: scram.topologies.TopologyCode,
) -> None:
    """Analyse toy model cage."""
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
) -> cgexplore.forcefields.ForceField:
    """Get forcefield."""
    present_beads = (
        cbead_d,
        abead_d,
        binder_bead,
        tetra_bead,
    )
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
    return cgexplore.systems_optimisation.get_forcefield_from_dict(
        identifier=identifier,
        prefix="min_val",
        present_beads=present_beads,
        vdw_bond_cutoff=2,
        definer_dict=definer_dict,
        verbose=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    parser.add_argument(
        "--targetted",
        action="store_true",
        help="set to do non-naive tests",
    )
    return parser.parse_args()


def make_plot(
    figure_dir: pathlib.Path,
    database_path: pathlib.Path,
    filename: str,
) -> None:
    """Plot energies."""
    energies = defaultdict(list)
    bacs = defaultdict(list)
    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]
        bac_angle = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]

        if entry.properties["num_components"] > 1:
            continue

        energies[multi].append((bac_angle, energy, entry.key))
        bacs[bac_angle].append((multi, energy, entry.key))

    fig, axs = plt.subplots(
        nrows=len(energies),
        sharey=True,
        sharex=True,
        figsize=(8, 10),
    )
    flat_axs = [axs] if len(energies) == 1 else axs.flatten()

    for i, (ax, multi) in enumerate(
        zip(flat_axs, sorted([int(i) for i in energies]), strict=True)
    ):
        axx = ax.twinx()

        idx = str(multi)
        min_energy = min(energies[idx], key=lambda p: p[1])

        ax.scatter(
            [i[0] for i in energies[idx]],
            [i[1] for i in energies[idx]],
            marker="o",
            c="tab:blue",
            s=20,
            alpha=0.4,
            ec="none",
            label=(
                f"M{idx}: {round(min_energy[1],2)} @ {min_energy[0]} "
                f"({min_energy[2]})"
            ),
            zorder=2,
        )

        countsx = {}
        bac_line = []
        for bac_angle in sorted(bacs):
            rel_energies = [i[1] for i in energies[idx] if i[0] == bac_angle]
            if len(rel_energies) == 0:
                continue
            min_energy = min(rel_energies)
            bac_line.append((bac_angle, min_energy))

            countsx[bac_angle] = len(rel_energies)

        ax.plot(
            [i[0] for i in bac_line],
            [i[1] for i in bac_line],
            c="tab:blue",
            ls="--",
            alpha=1.0,
            zorder=2,
        )

        ax.tick_params(axis="both", which="major", labelsize=16)

        ax.set_yscale("log")
        ax.axhline(y=0.3, c="k", ls="--")

        axx.plot(
            list(countsx),
            [countsx[i] for i in countsx],
            c="tab:red",
            marker="D",
            zorder=2,
            markersize=3.0,
        )
        axx.tick_params(
            axis="both",
            which="major",
            labelsize=16,
            labelcolor="tab:red",
        )
        if i == 4:  # noqa: PLR2004
            ax.set_ylabel(eb_str(), fontsize=16)
            axx.set_ylabel("num. structures", fontsize=16, color="tab:red")
        axx.set_yticks([max([countsx[i] for i in countsx])])
        # axx.set_ylim(0, None)  # noqa: ERA001

        leg = ax.legend(ncols=1, fontsize=12)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

    ax.set_xlabel("target $bac$ angle [deg]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "validation_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "validation_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "validation_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "validation_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "validation_run.db"

    if args.targetted:
        ligands = {
            str(bac_angle): {
                "forcefield": get_validation_forcefield(
                    bac_angle=bac_angle,
                    identifier=str(bac_angle),
                ),
                "stoichiometry_L_M": (2, 1),
                "ditopic": cgexplore.molecular.TwoC1Arm(
                    bead=cbead_d,
                    abead1=abead_d,
                ),
                "tetra": cgexplore.molecular.FourC1Arm(
                    bead=tetra_bead,
                    abead1=binder_bead,
                ),
                "multipliers": (multi,),
            }
            for bac_angle, multi in zip(
                (110, 120, 135, 150), (3, 4, 6, 12), strict=True
            )
        }
    else:
        ligands = {
            str(bac_angle): {
                "forcefield": get_validation_forcefield(
                    bac_angle=bac_angle,
                    identifier=str(i),
                ),
                "stoichiometry_L_M": (2, 1),
                "ditopic": cgexplore.molecular.TwoC1Arm(
                    bead=cbead_d,
                    abead1=abead_d,
                ),
                "tetra": cgexplore.molecular.FourC1Arm(
                    bead=tetra_bead,
                    abead1=binder_bead,
                ),
                "multipliers": (1, 2, 3, 4, 6, 8, 10, 12),
            }
            for i, bac_angle in enumerate(range(40, 181, 5))
        }

    if args.run:
        for lig in ligands:
            forcefield = ligands[lig]["forcefield"]
            ditopic = ligands[lig]["ditopic"]
            tetra = ligands[lig]["tetra"]

            ditopic_bb = scram.toy.prepare_building_block(
                precursor=ditopic,
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

            for multiplier in ligands[lig]["multipliers"]:
                iterator = scram.topologies.HomolepticTopologyIterator(
                    multiplier=multiplier,
                    stoichiometry=ligands[lig]["stoichiometry_L_M"],
                    tetra_bb=tetra_bb,
                    ditopic_bb=ditopic_bb,
                )
                count = 0
                logging.info("doing: ligand %s, multi %s", lig, multiplier)
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    name = f"{lig}_{multiplier}_{idx}"
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

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=constructed.topology_code,
                        )

                    except (OpenMMException, ValueError):
                        pass
                    count += 1

                    maxc = 100 if args.targetted else 40
                    if count == maxc:
                        break

                make_plot(
                    database_path=database_path,
                    figure_dir=figure_dir,
                    filename="validation_1.png",
                )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="validation_1.png",
    )


if __name__ == "__main__":
    main()
