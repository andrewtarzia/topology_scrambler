"""Perform crest analysis on ligand."""

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from utilities import plot_xy

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """Run script."""
    raise SystemExit("update directories to be new for non sept")
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    ligand_dir = wd / "ligands"
    ligand_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    calculation_dir = wd / "calculations"
    calculation_dir.mkdir(exist_ok=True)
    crest_path = pathlib.Path("/home/atarzia/software/crest_301/crest")
    xtb_path = pathlib.Path("/home/atarzia/miniforge3/envs/tscram/bin/xtb")

    forcefield_info_file = figure_dir / "auto_ff_information.txt"
    if forcefield_info_file.exists():
        forcefield_info_file.unlink()

    ligands = {
        "ls1": {"smiles": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3"},
        "lf": {
            "smiles": (
                "C1=C(C2=CC=C(C3C=CC4C(=O)C5C=CC(C6=CC=C(C7=CC=CN=C7)C=C6)=CC"
                "=5C=4C=3)C=C2)C=NC=C1"
            ),
        },
        "ls9": {"smiles": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3"},
    }

    for ligand in ligands:
        logging.info("doing %s", ligand)
        if "smiles" in ligands[ligand]:
            molecule = stk.BuildingBlock(ligands[ligand]["smiles"])
        elif "input" in ligands[ligand]:
            molecule = stk.BuildingBlock.init_from_file(
                ligands[ligand]["input"]
            )

        molecule.write(ligand_dir / f"{ligand}_unopt.mol")

        ensemble = scram.run_conformer_analysis(
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

        _ = scram.atomistic.get_ligand_bb(
            path=wd / "ligands" / f"{ligand}_prep.mol",
            optl_path=wd / "ligands" / f"{ligand}_optl.mol",
        )


if __name__ == "__main__":
    main()
