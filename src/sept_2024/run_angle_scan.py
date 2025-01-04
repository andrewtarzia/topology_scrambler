"""Script to generate and optimise CG models."""

import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import stk
from openmm import OpenMMException
from rdkit import RDLogger
from utilities import (
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
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

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield.get_forcefield_dictionary(),
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
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin = 0
    vmax = 0.6
    min_energy = float("inf")
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
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

    cg_scale = 2
    ax.scatter(
        [3.4 / (2 * cg_scale), 5.0 / (2 * cg_scale), 3.9 / (2 * cg_scale)],
        [90, 110, 120],
        c="tab:red",
        alpha=1.0,
        edgecolor="k",
        s=100,
        marker="X",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$a$-$c$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel("$b$-$a$-$c$  [$^\\circ$]", fontsize=16)

    ax.axhline(y=90, c="k", ls="--", alpha=0.5)
    ax.axhline(y=120, c="k", ls="--", alpha=0.5)
    ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
    ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)

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


def make_ac_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(3, 5))
    bac = 120
    xs = []
    energies = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        x = entry.properties["forcefield_dict"]["v_dict"]["a_c"]
        y = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        if y != bac:
            continue

        xs.append(x)
        energies.append(entry.properties["energy_per_bb"])
    ax.plot(
        xs,
        energies,
        c="tab:blue",
        alpha=1.0,
        mec="k",
        markersize=8,
        marker="o",
    )

    cg_scale = 2

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$a$-$c$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)

    ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
    ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
    ax.set_ylim(0, 1)

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


def make_bac_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(3, 5))
    ac = 1.0
    xs = []
    energies = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        x = entry.properties["forcefield_dict"]["v_dict"]["a_c"]
        y = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        if x != ac:
            continue

        xs.append(y)
        energies.append(entry.properties["energy_per_bb"])
    ax.plot(
        xs,
        energies,
        c="tab:blue",
        alpha=1.0,
        mec="k",
        markersize=8,
        marker="o",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("$b$-$a$-$c$  [$^\\circ$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)

    ax.axvline(x=90, c="k", ls="--", alpha=0.5)
    ax.axvline(x=120, c="k", ls="--", alpha=0.5)
    ax.set_ylim(0, 1)

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
    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
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
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "la"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

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

        converging_bb = cgx.utilities.optimise_ligand(
            molecule=converging.get_building_block(),
            name=f"{converging.get_name()}_f{forcefield.get_identifier()}",
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        )
        converging_bb.write(
            str(
                ligand_dir
                / f"{converging.get_name()}_f{forcefield.get_identifier()}"
                "_optl.mol"
            )
        )
        converging_bb = converging_bb.clone()

        tetra_bb = cgx.utilities.optimise_ligand(
            molecule=tetra.get_building_block(),
            name=f"{tetra.get_name()}_f{forcefield.get_identifier()}",
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        )
        tetra_bb.write(
            str(
                ligand_dir
                / f"{tetra.get_name()}_f{forcefield.get_identifier()}"
                "_optl.mol"
            )
        )
        tetra_bb = tetra_bb.clone()

        diverging_bb = cgx.utilities.optimise_ligand(
            molecule=diverging.get_building_block(),
            name=f"{diverging.get_name()}_f{forcefield.get_identifier()}",
            output_dir=calculation_dir,
            forcefield=forcefield,
            platform=None,
        )
        diverging_bb.write(
            str(
                ligand_dir
                / f"{diverging.get_name()}_f{forcefield.get_identifier()}"
                "_optl.mol"
            )
        )
        diverging_bb = diverging_bb.clone()

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
            conformer = cgx.scram.optimise_cage(
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
    make_ac_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_2.png",
    )
    make_bac_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_3.png",
    )


if __name__ == "__main__":
    main()
