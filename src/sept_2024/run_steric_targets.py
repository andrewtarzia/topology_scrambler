"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
from collections import defaultdict

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from openmm import openmm
from rdkit import RDLogger
from utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    inner_bead,
    isomer_energy,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")

cmap = {
    "3P6": "tab:blue",
    "3P6s": "tab:orange",
    "4P8": "tab:green",
    "4P8s": "tab:red",
    "4P82": "tab:pink",
    "4P82s": "tab:cyan",
}


def geom_distributions(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    fileprefix: str,
) -> None:
    """Plot geometry distributions."""
    skips = (
        "Pb_Pd",
        "Fe_Ga",
        "Fe_Ni",
        "Ni_Ni",
        "Ag_Ba",
        "Ga_Pb",
        "Ba_Pb",
        "Ir_Ni",
        "Pd_Pb_Ga",
        "Pb_Pd_Pb",
        # "Pb_Ga_Fe",
        "Pd_Pb_Ba",
        # "Pb_Ba_Ag",
        # "Ga_Fe_Ni",
        # "Fe_Ni_Ni",
        # "Ba_Ag_Ba",
        # "Fe_Ni_Ir",
        # "Ni_Ir_Ni",
        # "Pb_Ba_Ag_Ba_Pb",
        # "Fe_Ni_Ni_Fe",
        # "Fe_Ni_Ir_Ni_Fe",
    )

    geom_dict = defaultdict(lambda: defaultdict(list))
    database = cgx.utilities.AtomliteDatabase(database_path)
    for entry in database.get_entries():
        for label, gd_data in entry.properties["bond_data"].items():
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)
        for label, gd_data in entry.properties["angle_data"].items():
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)
        for label, gd_data in entry.properties["dihedral_data"].items():
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)

    for label, tstr_dict in geom_dict.items():
        if label in skips:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        lbl_length = len(label.split("_"))

        if lbl_length == 2:  # noqa: PLR2004
            xwidth = 0.02
            xmin = 0.4
            xmax = 5

        if lbl_length == 3:  # noqa: PLR2004
            xwidth = 2
            xmin = 0
            xmax = 180

        if lbl_length == 4:  # noqa: PLR2004
            xwidth = 0.5
            xmin = -10
            xmax = 10

        if lbl_length == 5:  # noqa: PLR2004
            xwidth = 0.5
            xmin = -10
            xmax = 10

        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)

        for tstr in cmap:
            xdata = tstr_dict[tstr]
            if len(xdata) == 0:
                continue

            ax.hist(
                x=xdata,
                bins=xbins,
                density=True,
                histtype="stepfilled",
                stacked=True,
                facecolor=cmap[tstr],
                linewidth=1.0,
                edgecolor="k",
                label=tstr,
                alpha=0.6,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel("frequency", fontsize=16)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"{fileprefix}_{label}.png",
            dpi=360,
            bbox_inches="tight",
        )
        fig.savefig(
            figure_dir / f"{fileprefix}_{label}.pdf",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

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
                "tstr": name.split("_")[1],
                "attempt": name.split("_")[-1],
            },
        )

    properties = database.get_entry(key=name).properties
    if "bond_data" not in properties:
        # Always want to extract target torions if present.
        g_measure = cgx.analysis.GeomMeasure(
            target_torsions=(
                cgx.terms.TargetTorsion(
                    search_string=("b", "a", "c", "a", "b"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    torsion_n=1,
                ),
                cgx.terms.TargetTorsion(
                    search_string=("e", "d", "i", "d", "e"),
                    search_estring=("Fe", "Ni", "Ir", "Ni", "Fe"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    torsion_n=1,
                ),
                cgx.terms.TargetTorsion(
                    search_string=("e", "d", "d", "e"),
                    search_estring=("Fe", "Ni", "Ni", "Fe"),
                    measured_atom_ids=[0, 1, 2, 3],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    torsion_n=1,
                ),
            )
        )
        bond_data = g_measure.calculate_bonds(final_molecule)
        bond_data = {"_".join(i): bond_data[i] for i in bond_data}
        angle_data = g_measure.calculate_angles(final_molecule)
        angle_data = {"_".join(i): angle_data[i] for i in angle_data}
        dihedral_data = g_measure.calculate_torsions(
            molecule=final_molecule,
            absolute=False,
            as_search_string=True,
        )

        database.add_properties(
            key=name,
            property_dict={
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
            },
        )

    properties = database.get_entry(key=name).properties
    if "min_ii_dist" not in properties:
        tstr = properties["tstr"]
        ii_dists = (
            stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
                molecule=final_molecule,
                metal_atom_nos=(77,),
            )
        )

        if "s" not in tstr:
            min_value = -1
            max_value = -1
        else:
            min_value = min(ii_dists.values())
            max_value = max(ii_dists.values())

        database.add_properties(
            key=name,
            property_dict={"min_ii_dist": min_value, "max_ii_dist": max_value},
        )


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas = defaultdict(lambda: defaultdict(list))
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if "s" in tstr:
            x = entry.properties["forcefield_dict"]["v_dict"]["i"]
        else:
            x = -0.2
        y = entry.properties["energy_per_bb"]

        datas[tstr][x].append(y)

    for tstr in cmap:
        ax.plot(
            list(datas[tstr]),
            [min(datas[tstr][i]) for i in datas[tstr]],
            alpha=1.0,
            marker="o",
            mec="k",
            markersize=10,
            label=f"{tstr}",
            c=cmap[tstr],
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.axhline(y=isomer_energy(), c="k", ls="--")
    ax.legend(fontsize=16)

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


def ii_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas = defaultdict(lambda: defaultdict(list))
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if "s" not in tstr:
            continue
        x = entry.properties["forcefield_dict"]["v_dict"]["i"]
        if "min_ii_dist" not in entry.properties:
            continue
        y = entry.properties["min_ii_dist"]

        datas[tstr][x].append(y)

    for tstr in cmap:
        if "s" not in tstr:
            continue
        ax.plot(
            list(datas[tstr]),
            [min(datas[tstr][i]) for i in datas[tstr]],
            alpha=1.0,
            marker="o",
            mec="k",
            markersize=10,
            label=f"{tstr}",
            c=cmap[tstr],
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel(r"min. $r_{i-i}$ [$\AA$]", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim(0, None)

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


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    calculation_dir = wd / "tsteric_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "tsteric_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "tsteric_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "tsteric_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "tsteric.db"

    pair = "la_st5"
    convergings = cgx.molecular.StericSixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
        ibead=inner_bead,
        sbead=steric_bead,
    )
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "la"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    e_range = [10]
    s_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    new_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbg": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "egb": ("angle", 120, 1e2),
        "deg": ("angle", 180, 1e2),
        # Torsions.
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "g": ("nb", 10.0, 1.0),
        "i": ("nb", 10.0, 1.0),
    }

    if args.run:
        # Without sterics.
        topologies = (
            ("3P6", stk.cage.M3L6, (2, 1)),
            ("4P8", cgx.topologies.CGM4L8, (1, 1)),
            ("4P82", cgx.topologies.M4L82, (1, 1)),
            ("3P6s", stk.cage.M3L6, (2, 1)),
            ("4P8s", cgx.topologies.CGM4L8, (1, 1)),
            ("4P82s", cgx.topologies.M4L82, (1, 1)),
        )

        ligand_measures = {
            "la": {"dd": 7.0, "de": 1.5, "eg": 1.4, "gb": 1.4, "dde": 170},
            "st5": {"ba": 2.8, "aa": 5.0, "bac": 120, "bacab": 180},
        }
        forcefield = precursors_to_forcefield(
            pair=pair,
            diverging=diverging,
            converging=converging,
            conv_meas=ligand_measures["la"],
            dive_meas=ligand_measures["st5"],
            new_definer_dict=new_definer_dict,
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
        converging_bb.write(str(ligand_dir / f"{converging_name}_optl.mol"))
        converging_bb = converging_bb.clone()

        tetra_name = f"{tetra.get_name()}_f{forcefield.get_identifier()}"
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
        diverging_bb.write(str(ligand_dir / f"{diverging_name}_optl.mol"))
        diverging_bb = diverging_bb.clone()

        for tstr, tfunction, _ in topologies:
            if "s" in tstr:
                continue
            for attempt, scale in enumerate(
                (1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
            ):
                name = f"ts_{tstr}_0_0_{attempt}"
                logging.info("building %s", name)

                if tstr == "3P6":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2),
                                converging_bb: (3, 4, 5, 6),
                                diverging_bb: (7, 8),
                            },
                            vertex_positions=None,
                            scale_multiplier=scale,
                        )
                    )
                    num_bbs = 9
                    ff = forcefield

                elif tstr == "4P8":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2, 3),
                                converging_bb: (4, 6, 8, 10),
                                diverging_bb: (5, 7, 9, 11),
                            },
                            vertex_positions=None,
                            scale_multiplier=scale,
                        )
                    )
                    num_bbs = 12
                    ff = forcefield

                elif tstr == "4P82":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2, 3),
                                converging_bb: (5, 6, 7, 8),
                                diverging_bb: (4, 9, 10, 11),
                            },
                            vertex_positions=None,
                            scale_multiplier=scale,
                        )
                    )
                    num_bbs = 12
                    ff = forcefield

                cage.write(structure_dir / f"{name}_unopt.mol")

                conformer = cgx.scram.optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=ff,
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
                    forcefield=ff,
                    num_building_blocks=num_bbs,
                )

        # With sterics.
        for (i, ivdw_e_value), (j, ivdw_s_value) in it.product(
            enumerate(e_range), enumerate(s_range)
        ):
            ligand_measures = {
                "la": {
                    "dd": 7.0,
                    "de": 1.5,
                    "ide": 170,
                    "eg": 1.4,
                    "gb": 1.4,
                    "ivdw_e": ivdw_e_value,
                    "ivdw_s": ivdw_s_value,
                },
                "st5": {"ba": 2.8, "aa": 5.0, "bac": 120, "bacab": 180},
            }

            forcefields = precursors_to_forcefield(
                pair=pair,
                diverging=diverging,
                converging=convergings,
                conv_meas=ligand_measures["la"],
                dive_meas=ligand_measures["st5"],
                new_definer_dict=new_definer_dict,
            )

            convergings_name = (
                f"{convergings.get_name()}_f{forcefields.get_identifier()}"
            )
            convergings_bb = cgx.utilities.optimise_ligand(
                molecule=convergings.get_building_block(),
                name=convergings_name,
                output_dir=calculation_dir,
                forcefield=forcefields,
                platform=None,
            )
            convergings_bb.write(
                str(ligand_dir / f"{convergings_name}_optl.mol")
            )
            convergings_bb = convergings_bb.clone()

            tetra_name = f"{tetra.get_name()}_f{forcefield.get_identifier()}"
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
            diverging_bb.write(str(ligand_dir / f"{diverging_name}_optl.mol"))
            diverging_bb = diverging_bb.clone()

            for tstr, tfunction, _ in topologies:
                if "s" not in tstr:
                    continue
                for attempt, scale in enumerate(
                    (1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
                ):
                    name = f"ts_{tstr}_{i}_{j}_{attempt}"
                    logging.info("building %s", name)

                    if tstr == "3P6s":
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2),
                                    convergings_bb: (3, 4, 5, 6),
                                    diverging_bb: (7, 8),
                                },
                                vertex_positions=None,
                                scale_multiplier=scale,
                            )
                        )
                        num_bbs = 9
                        ff = forcefields

                    elif tstr == "4P8s":
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    convergings_bb: (4, 6, 8, 10),
                                    diverging_bb: (5, 7, 9, 11),
                                },
                                vertex_positions=None,
                                scale_multiplier=scale,
                            )
                        )
                        num_bbs = 12
                        ff = forcefields

                    elif tstr == "4P82s":
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    convergings_bb: (5, 6, 7, 8),
                                    diverging_bb: (4, 9, 10, 11),
                                },
                                vertex_positions=None,
                                scale_multiplier=scale,
                            )
                        )
                        num_bbs = 12
                        ff = forcefields

                    cage.write(structure_dir / f"{name}_unopt.mol")

                    conformer = cgx.scram.optimise_cage(
                        molecule=cage,
                        name=name,
                        output_dir=calculation_dir,
                        forcefield=ff,
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
                        forcefield=ff,
                        num_building_blocks=num_bbs,
                    )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_1.png",
    )

    ii_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_2.png",
    )

    geom_distributions(
        database_path=database_path,
        figure_dir=figure_dir,
        fileprefix="sterics_3",
    )


if __name__ == "__main__":
    main()
