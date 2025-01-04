"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import stk
from openmm import OpenMMException
from rdkit import RDLogger
from ufo_utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)
from utilities import eb_str

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
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "study" not in properties:
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
                "multiplier": name.split("_")[1],
                "study": name.split("_")[0],
            },
        )


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(16, 5))
    vmin = 0
    vmax = 4.0
    min_energy = {"3": float("inf"), "4": float("inf")}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        x = entry.properties["forcefield_dict"]["v_dict"]["d_d_e"]
        y = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        c = entry.properties["energy_per_bb"]
        multi = entry.properties["multiplier"]
        logging.info("%s: x:%s, y:%s, e:%s", entry.key, x, y, c)

        min_energy[multi] = min(c, min_energy[multi])
        ax = ax3 if multi == "3" else ax4

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

    for multi in (3, 4):
        ax = ax3 if multi == 3 else ax4  # noqa: PLR2004
        ax.set_title(
            f"$M$={multi}: min. {eb_str(no_unit=True)}="
            f"{round(min_energy[str(multi)], 2)}",
            fontsize=16,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("$d$-$d$-$e$  [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$b$-$a$-$c$  [$^\\circ$]", fontsize=16)

        ax.axhline(y=150, c="k", ls="--", alpha=0.5)
        ax.axhline(y=170, c="k", ls="--", alpha=0.5)
        ax.axvline(x=133, c="k", ls="--", alpha=0.5)

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


def make_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(16, 5))
    vmin = 0
    vmax = 4.0
    min_energy = {"3": float("inf"), "4": float("inf")}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        x = entry.properties["forcefield_dict"]["v_dict"]["a_c"]
        y = entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
        c = entry.properties["energy_per_bb"]
        multi = entry.properties["multiplier"]
        logging.info("%s: x:%s, y:%s, e:%s", entry.key, x, y, c)

        min_energy[multi] = min(c, min_energy[multi])
        ax = ax3 if multi == "3" else ax4

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

    for multi in (3, 4):
        ax = ax3 if multi == 3 else ax4  # noqa: PLR2004
        ax.set_title(
            f"$M$={multi}: min. {eb_str(no_unit=True)}="
            f"{round(min_energy[str(multi)], 2)}",
            fontsize=16,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(r"$a$-$c$  [$\mathrm{\AA}$]", fontsize=16)
        ax.set_ylabel("$b$-$a$-$c$  [$^\\circ$]", fontsize=16)

        ax.axhline(y=150, c="k", ls="--", alpha=0.5)
        ax.axhline(y=170, c="k", ls="--", alpha=0.5)
        ax.axvline(x=5 / 4, c="k", ls="--", alpha=0.5)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "ufo_scan_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "ufo_scan_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "ufo_scan_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "ufo_scan_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "ufo_scan.db"

    bac_range = range(90, 180, 5)
    dde_range = range(90, 180, 5)

    l2s = ("ls1",)
    pairs = {}
    for l2 in l2s:
        name = f"lf_{l2}"
        pairs[name] = {
            "converging_name": "lf",
            "diverging_name": l2,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (3, 4),
        }

    if args.run:
        for (i, dde), (j, bac) in it.product(
            enumerate(dde_range), enumerate(bac_range)
        ):
            ligand_measures = {
                # From prep.
                "lf": {"dd": 8.0, "de": 4.3, "dde": dde, "eg": 1.4, "gb": 1.4},
                # From optl.
                "ls1": {"ba": 2.8, "aa": 4.9, "bac": bac, "bacab": 180},
            }

            for pair in pairs:
                converging_name = pairs[pair]["converging_name"]
                diverging_name = pairs[pair]["diverging_name"]

                converging = pairs[pair]["converging"]
                diverging = pairs[pair]["diverging"]
                tetra = pairs[pair]["tetra"]

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    diverging=diverging,
                    converging=converging,
                    conv_meas=ligand_measures[converging_name],
                    dive_meas=ligand_measures[diverging_name],
                )

                converging_name = (
                    f"{converging.get_name()}_f{forcefield.get_identifier()}"
                )
                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=converging.get_building_block(),
                    name=converging_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb.write(
                    str(ligand_dir / f"{converging_name}_optl.mol")
                )
                converging_bb = converging_bb.clone()

                tetra_name = (
                    f"{tetra.get_name()}_f{forcefield.get_identifier()}"
                )
                tetra_bb = cgx.utilities.optimise_ligand(
                    molecule=tetra.get_building_block(),
                    name=tetra_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                tetra_bb.write(str(ligand_dir / f"{tetra_name}_optl.mol"))
                tetra_bb = tetra_bb.clone()

                diverging_name = (
                    f"{diverging.get_name()}_f{forcefield.get_identifier()}"
                )
                diverging_bb = cgx.utilities.optimise_ligand(
                    molecule=diverging.get_building_block(),
                    name=diverging_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                diverging_bb.write(
                    str(ligand_dir / f"{diverging_name}_optl.mol")
                )
                diverging_bb = diverging_bb.clone()

                for multiplier in pairs[pair]["multipliers"]:
                    logging.info("doing: pair %s, multi %s", pair, multiplier)

                    name = f"ufoscan_{multiplier}_{i}-{j}"
                    if multiplier == 3:  # noqa: PLR2004
                        cage = stk.ConstructedMolecule(
                            stk.cage.M3L6(
                                building_blocks={
                                    tetra_bb: (0, 1, 2),
                                    converging_bb: (3, 5, 7),
                                    diverging_bb: (4, 6, 8),
                                },
                                vertex_positions=None,
                            )
                        )

                    elif multiplier == 4:  # noqa: PLR2004
                        cage = stk.ConstructedMolecule(
                            stk.cage.M4L8(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    converging_bb: (4, 6, 8, 10),
                                    diverging_bb: (5, 7, 9, 11),
                                },
                                vertex_positions=None,
                            )
                        )

                    cage.write(structure_dir / f"{name}_unopt.mol")

                    # Optimise and save.
                    logging.info("building %s", name)

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
                            num_building_blocks=9 if multiplier == 3 else 12,  # noqa: PLR2004
                        )

                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufoscan_1.png",
    )

    # Now here, I want to scan bac with ac length.
    database_path = data_dir / "ufo_scan2.db"
    bac_range = range(90, 180, 5)
    aa_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    l2s = ("ls1",)
    pairs = {}
    for l2 in l2s:
        name = f"lf_{l2}"
        pairs[name] = {
            "converging_name": "lf",
            "diverging_name": l2,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (3, 4),
        }

    if args.run:
        for (i, aa), (j, bac) in it.product(
            enumerate(aa_range), enumerate(bac_range)
        ):
            ligand_measures = {
                # From prep.
                "lf": {"dd": 8.0, "de": 4.3, "dde": 133, "eg": 1.4, "gb": 1.4},
                # From optl.
                "ls1": {"ba": 2.8, "aa": aa, "bac": bac, "bacab": 180},
            }

            for pair in pairs:
                converging_name = pairs[pair]["converging_name"]
                diverging_name = pairs[pair]["diverging_name"]

                converging = pairs[pair]["converging"]
                diverging = pairs[pair]["diverging"]
                tetra = pairs[pair]["tetra"]

                forcefield = precursors_to_forcefield(
                    pair=f"s2{pair}",
                    diverging=diverging,
                    converging=converging,
                    conv_meas=ligand_measures[converging_name],
                    dive_meas=ligand_measures[diverging_name],
                )

                converging_name = (
                    f"{converging.get_name()}_f{forcefield.get_identifier()}"
                )
                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=converging.get_building_block(),
                    name=converging_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb.write(
                    str(ligand_dir / f"{converging_name}_optl.mol")
                )
                converging_bb = converging_bb.clone()

                tetra_name = (
                    f"{tetra.get_name()}_f{forcefield.get_identifier()}"
                )
                tetra_bb = cgx.utilities.optimise_ligand(
                    molecule=tetra.get_building_block(),
                    name=tetra_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                tetra_bb.write(str(ligand_dir / f"{tetra_name}_optl.mol"))
                tetra_bb = tetra_bb.clone()

                diverging_name = (
                    f"{diverging.get_name()}_f{forcefield.get_identifier()}"
                )
                diverging_bb = cgx.utilities.optimise_ligand(
                    molecule=diverging.get_building_block(),
                    name=diverging_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                diverging_bb.write(
                    str(ligand_dir / f"{diverging_name}_optl.mol")
                )
                diverging_bb = diverging_bb.clone()

                for multiplier in pairs[pair]["multipliers"]:
                    logging.info("doing: pair %s, multi %s", pair, multiplier)

                    name = f"ufoscan2_{multiplier}_{i}-{j}"
                    if multiplier == 3:  # noqa: PLR2004
                        cage = stk.ConstructedMolecule(
                            stk.cage.M3L6(
                                building_blocks={
                                    tetra_bb: (0, 1, 2),
                                    converging_bb: (3, 5, 7),
                                    diverging_bb: (4, 6, 8),
                                },
                                vertex_positions=None,
                            )
                        )

                    elif multiplier == 4:  # noqa: PLR2004
                        cage = stk.ConstructedMolecule(
                            stk.cage.M4L8(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    converging_bb: (4, 6, 8, 10),
                                    diverging_bb: (5, 7, 9, 11),
                                },
                                vertex_positions=None,
                            )
                        )

                    cage.write(structure_dir / f"{name}_unopt.mol")

                    # Optimise and save.
                    logging.info("building %s", name)

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
                            num_building_blocks=9 if multiplier == 3 else 12,  # noqa: PLR2004
                        )

                    except OpenMMException:
                        logging.info("failed optimisation of %s", name)

    make_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="ufoscan_2.png",
    )


if __name__ == "__main__":
    main()
