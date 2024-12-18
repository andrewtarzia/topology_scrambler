"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import stko
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    isomer_energy,
    precursors_to_forcefield,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901, PLR0912
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: cgx.scram.TopologyIterator,
    topology_code: cgx.scram.TopologyCode,
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

        splits = name.split("_")
        if len(splits) == 4:  # noqa: PLR2004
            multiplier = name.split("_")[2]
            pairname = name.split("_")[0] + "_" + name.split("_")[1]
        elif len(splits) == 5:  # noqa: PLR2004
            multiplier = name.split("_")[3]
            pairname = (
                name.split("_")[0]
                + "_"
                + name.split("_")[1]
                + "_"
                + name.split("_")[2]
            )

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy
                / iterator.get_num_building_blocks(),
                "pair": pairname,
                "num_components": num_components,
                "multiplier": multiplier,
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

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

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
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.axhline(y=isomer_energy(), c="k", ls="--")
    ax.legend(ncols=1, fontsize=16)
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


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}

    xs = ["1", "2", "3", "4"]
    ys = ["la_st5", "la_st52", "la_c1", "la_c12", "la_st5_11", "la_st52_11"]
    ys.reverse()

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if "lf_ls" in pair or "_11" in pair:
            continue

        vstr = entry.key.split("_")[-1]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), vstr))

    vmin = 0
    vmax = 1
    for pair, multi in energies:
        sorted_energies = sorted(energies[(pair, multi)], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = ys.index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )
        ax.text(
            x=x,
            y=y,
            s=min_energy[1],
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="w" if min_energy[0] < 0.5 else "k",  # noqa: PLR2004
            fontsize=16,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels(ys)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"4:2:3 {eb_str()}", fontsize=16)

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


def make_summary_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    systems = {
        ("la_st5", "1"): {"name": "st5-s-1", "data": []},
        ("la_st5", "2"): {"name": "st5-s-2", "data": []},
        ("la_st5", "4"): {"name": "st5-s-4", "data": []},
        ("la_st52", "1"): {"name": "st5-l-1", "data": []},
        ("la_st52", "2"): {"name": "st5-l-2", "data": []},
        ("la_st52", "4"): {"name": "st5-l-4", "data": []},
        ("la_c1", "1"): {"name": "st1-t0-1", "data": []},
        ("la_c1", "2"): {"name": "st1-t0-2", "data": []},
        ("la_c1", "4"): {"name": "st1-t0-4", "data": []},
        ("la_c12", "1"): {"name": "st1-t60-1", "data": []},
        ("la_c12", "2"): {"name": "st1-t60-2", "data": []},
        ("la_c12", "4"): {"name": "st1-t60-4", "data": []},
        # ("la_c13", "1"): {"name": "st1-3-1", "data": []},
        # ("la_c13", "2"): {"name": "st1-3-2", "data": []},
        # ("la_c13", "4"): {"name": "st1-3-4", "data": []},
        # ("la_c14", "1"): {"name": "st1-4-1", "data": []},
        # ("la_c14", "2"): {"name": "st1-4-2", "data": []},
        # ("la_c14", "4"): {"name": "st1-4-4", "data": []},
        # ("la_c15", "1"): {"name": "st1-5-1", "data": []},
        # ("la_c15", "2"): {"name": "st1-5-2", "data": []},
        # ("la_c15", "4"): {"name": "st1-5-4", "data": []},
        ("la_st5_11", "1"): {"name": "st5-s,1:1-1", "data": []},
        ("la_st5_11", "2"): {"name": "st5-s,1:1-2", "data": []},
        ("la_st5_11", "3"): {"name": "st5-s,1:1-3", "data": []},
        ("la_st52_11", "1"): {"name": "st5-l,1:1-1", "data": []},
        ("la_st52_11", "2"): {"name": "st5-l,1:1-2", "data": []},
        ("la_st52_11", "3"): {"name": "st5-l,1:1-3", "data": []},
    }
    count_423 = 0
    count_111 = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if (pair, multi) not in systems:
            continue
        energy = entry.properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        systems[(pair, multi)]["data"].append(energy)

        if "_11" in pair:
            count_111 += 1
        else:
            count_423 += 1

    logging.info("structures built, 4:2:3 %s, 1:1:1 %s", count_423, count_111)
    rng = np.random.default_rng(seed=2)

    for i, (pair, multi) in enumerate(systems):
        if len(systems[(pair, multi)]["data"]) == 0:
            continue
        min_energy = min(systems[(pair, multi)]["data"])

        ax.scatter(
            [
                i + (2 * rng.random() - 1) * 0.3
                for j in range(len(systems[(pair, multi)]["data"]))
            ],
            systems[(pair, multi)]["data"],
            c="tab:blue",
            alpha=0.1,
            edgecolor="none",
            s=30,
            marker="o",
            zorder=1,
        )
        ax.scatter(
            i,
            min_energy,
            c="tab:orange",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
            zorder=2,
        )

    ax.axvline(x=2 + 0.5, c="gray")
    ax.axvline(x=5 + 0.5, c="gray")
    ax.axvline(x=8 + 0.5, c="gray")
    ax.axvline(x=11 + 0.5, c="gray")
    ax.axvline(x=14 + 0.5, c="gray")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=90)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.axhline(y=isomer_energy(), c="k", ls="--")

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


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
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
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
        "c12": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 120},
    }

    pairs = {
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
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
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
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
            "multipliers": (1, 2, 4),
        },
        "la_c1": {
            "converging_name": "la",
            "diverging_name": "c1",
            "stoichiometry_L_L_M": (4, 2, 3),
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
            "multipliers": (1, 2, 4),
        },
        "la_c12": {
            "converging_name": "la",
            "diverging_name": "c12",
            "stoichiometry_L_L_M": (4, 2, 3),
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
            "multipliers": (1, 2, 4),
        },
        "la_st5_11": {
            "converging_name": "la",
            "diverging_name": "st5",
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
            "multipliers": (1, 2, 3),
        },
        "la_st52_11": {
            "converging_name": "la",
            "diverging_name": "st52",
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
            "multipliers": (1, 2, 3),
        },
    }

    if args.run:
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

            for multiplier in pairs[pair]["multipliers"]:
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
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
                        conformer = cgx.scram.optimise_cage(
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

                    except OpenMMException:
                        pass

                make_plot(
                    database_path=database_path,
                    pair=pair,
                    structure_dir=structure_dir,
                    figure_dir=figure_dir,
                    filename=f"rerun_1_{pair}.png",
                )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rerun_3.png",
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rerun_4.png",
    )
    for pair in pairs:
        make_plot(
            database_path=database_path,
            pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"rerun_1_{pair}.png",
        )


if __name__ == "__main__":
    main()
