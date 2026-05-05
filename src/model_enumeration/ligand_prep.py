"""Perform crest analysis on ligand."""

import logging
import os
import pathlib
from collections import Counter

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"

import bbprep
import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def has_alkynes(lsmiles: str) -> bool:
    """Check if a molecule has alkynes."""
    mol = stk.BuildingBlock(
        lsmiles,
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts="[*][C]#[C][*]",
                bonders=(),
                deleters=(),
            )
        ),
    )
    return mol.get_num_functional_groups() > 0


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

    ax.legend(ncols=4, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "dist_vs_angles_1.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")
    run_prefix = "ligand_param"
    crest_path = pathlib.Path("/home/tarziaa/software/crest_301/crest")
    xtb_path = pathlib.Path(
        "/home/tarziaa/miniforge3/envs/meproduction/bin/xtb"
    )
    calculation_dir = wd / f"{run_prefix}_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ligand_dir = wd / f"{run_prefix}_ligands"
    ligand_dir.mkdir(exist_ok=True)
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / f"{run_prefix}"
    figure_dir.mkdir(exist_ok=True)

    ligands = {
        # cu18.
        "cu18_1": "C1=CC(=NC(=C1)C2=CC=C(C=C2)C(=O)O)C3=CC=C(C=C3)C(=O)O",
        "cu18_90": "C1=CC2=C(C=C1C(=O)O)C3=C(N2)C=CC(=C3)C(=O)O",
        # four_comp.
        "four_comp_lin": "C1=CN=CC=C1C2=CC=NC=C2",
        "four_comp_mxy": (
            "C1=CC(=CC(=C1)CN2C=CC(=N2)C3=CC=NC=C3)CN4C=CC(=N4)C5=CC=NC=C5"
        ),
        "four_comp_pxy": (
            "C1=CC(=CC=C1CN2C=C(C=N2)C3=CC=NC=C3)CN4C=C(C=N4)C5=CC=NC=C5"
        ),
        # tri_di.
        "tri_di_l1": "C(=CC(=O)O)C(=O)O",
        "tri_di_l2": "C1=CC(=CC=C1C(=O)O)C(=O)O",
        "tri_di_l5": "C1=CC2=C(C=CC(=C2)C(=O)O)C=C1C(=O)O",
        "tri_di_l6": "C1=C(SC(=C1)C=CC(=O)O)C=CC(=O)O",
        "tri_di_l9": "C1=CC(=CC=C1C2=CC=C(C=C2)C(=O)O)C3=CC=C(C=C3)C(=O)O",
    }

    ensembles = {}
    have_alkynes = set()
    for ligand, lsmiles in ligands.items():
        stk.BuildingBlock(lsmiles).write(ligand_dir / f"{ligand}_unopt.mol")
        if "tri_di" in ligand or "four_comp" in ligand or "cu18" in ligand:
            new_dir = calculation_dir / f"{ligand}_confs"
            new_dir.mkdir(exist_ok=True)
            logging.info("building confs for %s", ligand)

            input_molecule = stk.BuildingBlock(lsmiles)
            input_molecule.write(new_dir / f"{ligand}_input.mol")
            generator = bbprep.generators.ETKDG(num_confs=100)
            ensemble = generator.generate_conformers(input_molecule)
            for conformer in ensemble.yield_conformers():
                conformer.molecule.write(
                    new_dir / f"{ligand}_{conformer.conformer_id}.mol"
                )

        else:
            logging.info("doing %s", ligand)
            molecule = stk.BuildingBlock(lsmiles)

            molecule.write(ligand_dir / f"{ligand}_unopt.mol")

            ensemble = cgx.atomistic.run_conformer_analysis(
                ligand_name=ligand,
                molecule=molecule,
                ligand_dir=ligand_dir,
                calculation_dir=calculation_dir,
                functional_group_factories=(
                    stko.functional_groups.ThreeSiteFactory(
                        "[#6]~[#7X2]~[#6]"
                    ),
                ),
                crest_path=crest_path,
                xtb_path=xtb_path,
            )
            # Only CREST done ligands pass to the plotting below.
            ensembles[ligand] = ensemble
        if has_alkynes(lsmiles):
            have_alkynes.add(ligand)

    logging.info("ligands %s have alkynes", have_alkynes)
    plot_distance_angle(ensembles=ensembles, figure_dir=figure_dir)

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
