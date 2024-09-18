"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore
import matplotlib as mpl
import matplotlib.pyplot as plt
import stko
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
    save_vertex_positions,
    tetra_bead,
)

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901, PLR0912
    database_path: pathlib.Path,
    name: str,
    forcefield: cgexplore.forcefields.ForceField,
    iterator: scram.topologies.TopologyIterator,
    topology_code: scram.topologies.TopologyCode,
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

    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
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
    ax.axhline(y=0.3, c="k", ls="--")
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

    return energies


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}

    xs = ["1", "2", "4"]
    ys = ["la_st5", "la_st52", "la_c1", "la_c12", "la_c13", "la_c14", "la_c15"]

    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
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
    raise SystemExit


def make_parity_plot(  # noqa: PLR0912, C901, PLR0915
    database_path: pathlib.Path,
    steric_database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies = {}
    steric_energies = {}

    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if "lf_ls" in pair:
            continue
        vstr = entry.key.split("_")[-1]
        energy = entry.properties["energy_per_bb"]

        if pair not in energies:
            energies[pair] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[pair].append((round(energy, 4), vstr, multi))

    for entry in cgexplore.utilities.AtomliteDatabase(
        steric_database_path
    ).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if "lf_ls" in pair:
            continue
        vstr = entry.key.split("_")[-1]
        energy = entry.properties["energy_per_bb"]

        if pair not in steric_energies:
            steric_energies[pair] = []

        if entry.properties["num_components"] > 1:
            continue
        steric_energies[pair].append((round(energy, 4), vstr, multi))

    for pair in energies:
        if "_11" in pair:
            continue
        sorted_energies = sorted(energies[pair], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        pair11 = pair + "_11"
        sorted_energies11 = sorted(energies[pair11], key=lambda p: p[0])
        min_energy11 = sorted_energies11[0]
        ec = "k" if min_energy[0] < 0.3 else "none"  # noqa: PLR2004

        ax.scatter(
            min_energy[0],
            min_energy11[0],
            c="tab:blue",
            ec=ec,
            s=60,
        )
        ax.text(
            x=6,
            y=min_energy11[0],
            s=(
                f"{pair}: {min_energy[2]}|{min_energy[1]} vs. "
                f"{min_energy11[2]}|{min_energy11[1]}"
            ),
        )

    for pair in steric_energies:
        if "_11" in pair:
            continue

        sorted_energies = sorted(steric_energies[pair], key=lambda p: p[0])
        min_energy = sorted_energies[0]

        pair11 = pair + "_11"
        sorted_energies11 = sorted(steric_energies[pair11], key=lambda p: p[0])
        min_energy11 = sorted_energies11[0]
        ec = "k" if min_energy[0] < 0.3 else "none"  # noqa: PLR2004

        ax.scatter(
            min_energy[0],
            min_energy11[0],
            c="tab:orange",
            ec=ec,
            s=60,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"4:2:3 {eb_str()}", fontsize=16)
    ax.set_ylabel(f"1:1:1 {eb_str()}", fontsize=16)

    ax.set_xlim(0.0, 10)
    ax.set_ylim(0.0, 10)

    ax.plot((0, 10), (0, 10), c="k")

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
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
    steric_database_path = wd / "steric_data" / "steric.db"

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
            "converging": SixBead(
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
        "la_st5_11": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_st52_11": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c1_11": {
            "converging_name": "la",
            "diverging_name": "c1",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c12_11": {
            "converging_name": "la",
            "diverging_name": "c12",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c13_11": {
            "converging_name": "la",
            "diverging_name": "c13",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c14_11": {
            "converging_name": "la",
            "diverging_name": "c14",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c15_11": {
            "converging_name": "la",
            "diverging_name": "c15",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
        "la_c12": {
            "converging_name": "la",
            "diverging_name": "c12",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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
        "la_c13": {
            "converging_name": "la",
            "diverging_name": "c13",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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
        "la_c14": {
            "converging_name": "la",
            "diverging_name": "c14",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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
        "la_c15": {
            "converging_name": "la",
            "diverging_name": "c15",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(
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
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(
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

            for multiplier in pairs[pair]["multipliers"]:
                # Define a connectivity based on a multiplier.
                iterator = scram.topologies.TopologyIterator(
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

        _ = make_plot(
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
    for pair in pairs:
        make_plot(
            database_path=database_path,
            pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"rerun_1_{pair}.png",
        )
    make_parity_plot(
        database_path=database_path,
        steric_database_path=steric_database_path,
        figure_dir=figure_dir,
        filename="rerun_2.png",
    )


if __name__ == "__main__":
    main()
