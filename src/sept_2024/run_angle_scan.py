"""Script to generate and optimise CG models."""

import itertools as it
import logging
import pathlib

import cgexplore
import matplotlib as mpl
import matplotlib.pyplot as plt
import stk
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from utilities import (
    SixBead,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901
    database_path: pathlib.Path,
    name: str,
    forcefield: cgexplore.forcefields.ForceField,
    num_building_blocks: int,
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

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy / num_building_blocks,
            },
        )


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vmin = 0
    vmax = 1.0
    min_energy = float("inf")
    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        x = entry.properties["forcefield_dict"]["v_dict"]["a_c"]
        y = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        c = entry.properties["energy_per_bb"]
        logging.info("%s: x:%s, y:%s, e:%s", entry.key, x, y, c)
        min_energy = min(c, min_energy)

        ax.scatter(
            x,
            y,
            c=c,
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$a$-$c$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel("$b$-$a$-$c$  [$^\\circ$]", fontsize=16)

    ax.axhline(y=90, c="k", ls="--", alpha=0.5)
    ax.axhline(y=120, c="k", ls="--", alpha=0.5)
    ax.axvline(x=3.4 / 4, c="k", ls="--", alpha=0.5)
    ax.axvline(x=5.0 / 4, c="k", ls="--", alpha=0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(eb_str(), fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "scan_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "scan_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "scan_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "scan_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "scan.db"

    aa_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    bac_range = [90, 100, 105, 110, 115, 120, 125, 130, 135, 140, 150]

    pair = "la_st5"
    converging = SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c)
    converging_name = "la"
    diverging = cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    logging.info("building %s structures", len(aa_range) * len(bac_range))
    for (i, aa), (j, bac) in it.product(
        enumerate(aa_range), enumerate(bac_range)
    ):
        ligand_measures = {
            "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4},
            "st5": {"ba": 2.8, "aa": aa, "bac": bac, "bacab": 180},
        }

        forcefield = precursors_to_forcefield(
            pair=f"{pair}",
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

        name = f"scan_{i}-{j}"
        logging.info("building %s", name)

        cage = stk.ConstructedMolecule(
            stk.cage.M3L6(
                building_blocks={
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                },
                vertex_positions=None,
            )
        )
        cage.write(structure_dir / f"{name}_unopt.mol")

        try:
            conformer = scram.toy.optimise_cage(
                molecule=cage,
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
                num_building_blocks=9,
            )

        except OpenMMException:
            pass

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_1.png",
    )


if __name__ == "__main__":
    main()
