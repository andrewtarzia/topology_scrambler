"""Perform crest analysis on ligand."""

import logging
import pathlib
from collections import Counter

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def plot_xy(
    xproperty: str,
    ensemble: dict[str:dict],
    min_energy: float,
    figure_dir: pathlib.Path,
    ligand_name: str,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if xproperty in ("binder_angles",):
        ax.scatter(
            [ensemble[i][xproperty][0] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )
        ax.scatter(
            [ensemble[i][xproperty][1] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    elif xproperty in ("torsion_state",):
        xs = [Counter(ensemble[i][xproperty]) for i in ensemble]

        xs = [i.get("b", 0) for i in xs]

        ax.scatter(
            xs,
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    else:
        ax.scatter(
            [ensemble[i][xproperty] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xproperty, fontsize=16)
    ax.set_ylabel("relative energy [kJ/mol]", fontsize=16)
    if xproperty == "binder_adjacent_torsion":
        ax.set_xlim(-180, 180)

    if xproperty == "binder_angles":
        ax.set_xlim(0, 180)

    if xproperty == "binder_binder_angle":
        ax.set_xlim(0, 180)

    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"xy_{xproperty}_{ligand_name}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_distance_angle(
    ensembles: dict[str, dict[str, dict]],  # type: ignore[type-arg]
    figure_dir: pathlib.Path,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ligand, ensemble in ensembles.items():
        min_energy = min([ensemble[i]["energy"] for i in ensemble])

        is_kept = [
            i
            for i in ensemble
            if (ensemble[i]["energy"] - min_energy) * 2625.5 < 20  # noqa: PLR2004
        ]

        xs = [ensemble[i]["binder_angles"][0] for i in is_kept] + [
            ensemble[i]["binder_angles"][0] for i in is_kept
        ]
        ys = [ensemble[i]["binder_distance"] for i in is_kept] + [
            ensemble[i]["binder_distance"] for i in is_kept
        ]

        ax.scatter(
            xs,
            ys,
            marker="o",
            edgecolor="k",
            s=40,
            label=ligand,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("binder angle [deg]", fontsize=16)
    ax.set_ylabel("N-N distance [AA]", fontsize=16)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "dist_vs_angles_1.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    ligand_dir = wd / "ufo_aa_ligands"
    ligand_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "ufo_aa"
    figure_dir.mkdir(exist_ok=True)
    calculation_dir = wd / "ufo_aa_calculations"
    calculation_dir.mkdir(exist_ok=True)
    crest_path = pathlib.Path("/home/atarzia/software/crest_301/crest")
    xtb_path = pathlib.Path(
        "/home/atarzia/miniforge3/envs/meproduction/bin/xtb"
    )

    ligands = {
        "lf": {
            "smiles": (
                "C1=C(C2=CC=C(C3C=CC4C(=O)C5C=CC(C6=CC=C(C7=CC=CN=C7)C=C6)=CC"
                "=5C=4C=3)C=C2)C=NC=C1"
            ),
        },
        "ls1": {"smiles": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3"},
        "ls2": {"smiles": "C1=CC(=NC(=C1)C2=CC=NC=C2)C3=CC=NC=C3"},
        "ls3": {"smiles": "C1=CC(=C(C(=C1)C2=CC=NC=C2)N)C3=CC=NC=C3"},
        "ls4": {"smiles": "N1=CC=C(C2=C(OC)C(C3=CC=NC=C3)=CC=C2)C=C1"},
        "ls5": {"smiles": "C1=CC(=C(C(=C1)C2=CC=NC=C2)O)C3=CC=NC=C3"},
        "ls7": {
            "smiles": "C1(C(C2C=CN=CC=2)=CC=CC=1C1C=CN=CC=1)OC(=O)C(C)(C)C"
        },
        "ls8": {
            "smiles": "C1(C(C2C=CN=CC=2)=CC=CC=1C1C=CN=CC=1)OC(=O)C1C=CC=CC=1"
        },
        "ls9": {"smiles": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3"},
        "ls10": {"smiles": "C1=CN=CC=C1C2=CC=C([Se]2)C3=CC=NC=C3"},
    }

    ensembles = {}
    for ligand, ldict in ligands.items():
        logging.info("doing %s", ligand)
        molecule = stk.BuildingBlock(ldict["smiles"])

        molecule.write(ligand_dir / f"{ligand}_unopt.mol")

        ensemble = cgx.atomistic.run_conformer_analysis(
            ligand_name=ligand,
            molecule=molecule,
            ligand_dir=ligand_dir,
            calculation_dir=calculation_dir,
            functional_group_factories=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
            crest_path=crest_path,
            xtb_path=xtb_path,
        )
        ensembles[ligand] = ensemble

    plot_distance_angle(
        ensembles=ensembles,
        figure_dir=figure_dir,
    )

    for ligand, ensemble in ensembles.items():
        min_energy = min([ensemble[i]["energy"] for i in ensemble])

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 5))
        xwidth = 0.5
        relative_energies_kjmol = [
            (ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble
        ]
        xmin = min(relative_energies_kjmol)
        xmax = 20
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ax.hist(
            x=relative_energies_kjmol,
            bins=xbins,
            density=False,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("relative energy [kJ/mol]", fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"dist_energy_{ligand}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()

        for x_property in (
            "binder_angles",
            "binder_binder_angle",
            "binder_distance",
            "binder_adjacent_torsion",
            "adjacent_distance",
        ):
            plot_xy(
                xproperty=x_property,
                ensemble=ensemble,
                min_energy=min_energy,
                figure_dir=figure_dir,
                ligand_name=ligand,
            )

        _ = cgx.atomistic.get_ditopic_aligned_bb(
            path=ligand_dir / f"{ligand}_prep.mol",
            optl_path=ligand_dir / f"{ligand}_optl.mol",
        )


if __name__ == "__main__":
    main()
