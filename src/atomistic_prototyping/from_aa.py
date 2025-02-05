"""Perform analysis on ligand conformations."""

import logging
import pathlib
from collections import Counter

import bbprep
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
    ensemble_data: dict[str, dict],
    min_energy: float,
    figure_dir: pathlib.Path,
    ligand_name: str,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if xproperty in ("binder_angles",):
        ax.scatter(
            [ensemble_data[i][xproperty][0] for i in ensemble_data],
            [(ensemble_data[i]["energy"] - min_energy) for i in ensemble_data],
            edgecolor="k",
            s=80,
        )
        ax.scatter(
            [ensemble_data[i][xproperty][1] for i in ensemble_data],
            [(ensemble_data[i]["energy"] - min_energy) for i in ensemble_data],
            edgecolor="k",
            marker="D",
            s=80,
        )
    elif xproperty in ("torsion_state",):
        xs = [Counter(ensemble_data[i][xproperty]) for i in ensemble_data]

        xs = [i.get("b", 0) for i in xs]

        ax.scatter(
            xs,
            [(ensemble_data[i]["energy"] - min_energy) for i in ensemble_data],
            edgecolor="k",
            marker="D",
            s=80,
        )
    else:
        ax.scatter(
            [ensemble_data[i][xproperty] for i in ensemble_data],
            [(ensemble_data[i]["energy"] - min_energy) for i in ensemble_data],
            edgecolor="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xproperty, fontsize=16)
    ax.set_ylabel("relative MMFF energy [kcal mol-1]", fontsize=16)
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


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/unreacted/")
    ligand_dir = wd / "aa_ligands"
    ligand_dir.mkdir(exist_ok=True)

    ligands = {
        "ltp": {
            "smiles": "O=C1N(c2cccnc2)C(=O)c3cc4CN5CN(Cc6cc7C(=O)N(c8cccnc8)C"
            "(=O)c9cccc(c56)c79)c4c%10cccc1c3%10"
        },
        "alcltp": {
            "smiles": "O=C1NC(=O)c2cc3CN4CN(Cc5cc6C(=O)NC(=O)c7cccc(c45)c67)"
            "c3c8cccc1c28"
        },
    }

    for ligand in ligands:
        logging.info("doing %s", ligand)
        ensemble_dir = ligand_dir / f"{ligand}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        figure_dir = wd / "figures" / f"{ligand}_ensemble"
        figure_dir.mkdir(exist_ok=True)

        molecule = stk.BuildingBlock(
            smiles=ligands[ligand]["smiles"],
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        )
        molecule.write(ligand_dir / f"{ligand}_unopt.mol")

        calculator = bbprep.EnergyCalculator(
            name="MMFFEnergy",
            function=stko.MMFFEnergy().get_energy,
        )

        optimiser = bbprep.Optimiser(
            name="MMFF",
            function=stko.MMFF().optimize,
        )

        generator = bbprep.generators.ETKDG(num_confs=400)
        ensemble = generator.generate_conformers(molecule)
        logging.info("built %s", ensemble)

        # Optimise ensemble.
        opt_ensemble = ensemble.optimise_conformers(optimiser=optimiser)

        # Get lowest energy conformer.
        lowest_energy_conformer = opt_ensemble.get_lowest_energy_conformer(
            calculator=calculator
        )
        logging.info(
            "lowest E: %s",
            calculator.function(lowest_energy_conformer.molecule),
        )
        lowest_energy_conformer.molecule.write(
            ligand_dir / f"{ligand}_lowe.mol"
        )

        if ligand == "alcltp":
            continue
        # Get the prepared conformer.
        process = bbprep.DitopicFitter(ensemble=opt_ensemble)
        min_molecule = process.get_minimum()
        min_molecule.molecule.write(ligand_dir / f"{ligand}_prep.mol")

        calc = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        ensemble_data = {}
        for opt_conf in opt_ensemble.yield_conformers():
            opt_conf.molecule.write(
                ensemble_dir / f"conf_{opt_conf.conformer_id}.mol"
            )

            adjacent_centroids = calc.get_adjacent_centroids(opt_conf.molecule)
            adjacent_distance = np.linalg.norm(
                adjacent_centroids[0] - adjacent_centroids[1]
            )

            ensemble_data[opt_conf.conformer_id] = {
                "energy": float(calculator.function(opt_conf.molecule)),
                "binder_angles": calc.get_binder_angles(opt_conf.molecule),
                "binder_binder_angle": calc.get_binder_binder_angle(
                    opt_conf.molecule
                ),
                "binder_distance": calc.get_binder_distance(opt_conf.molecule),
                "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(
                    opt_conf.molecule
                ),
                "adjacent_distance": adjacent_distance,
                "binder_com_angle": calc.get_binder_centroid_angle(
                    opt_conf.molecule
                ),
            }

        min_energy = min([ensemble_data[i]["energy"] for i in ensemble_data])

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 5))
        xwidth = 0.5
        relative_energies_kjmol = [
            (ensemble_data[i]["energy"] - min_energy) for i in ensemble_data
        ]
        xmin = min(relative_energies_kjmol)
        xmax = max(relative_energies_kjmol)
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
        ax.set_xlabel("relative MMFF energy [kcal mol-1]", fontsize=16)
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
                ensemble_data=ensemble_data,
                min_energy=min_energy,
                figure_dir=figure_dir,
                ligand_name=ligand,
            )


if __name__ == "__main__":
    main()
