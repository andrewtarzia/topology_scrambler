"""Perform crest analysis on ligand."""

import logging
import pathlib
from collections import Counter

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


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    ligand_dir = wd / "mgen_aa_ligands"
    ligand_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_aa"
    figure_dir.mkdir(exist_ok=True)
    calculation_dir = wd / "mgen_aa_calculations"
    calculation_dir.mkdir(exist_ok=True)
    crest_path = pathlib.Path("/home/atarzia/software/crest_301/crest")
    xtb_path = pathlib.Path(
        "/home/atarzia/miniforge3/envs/meproduction/bin/xtb"
    )

    ligands = {
        # Case study 2.
        "lf": (
            "C1=C(C2=CC=C(C3C=CC4C(=O)C5C=CC(C6=CC=C(C7=CC=CN=C7)C=C6)=CC"
            "=5C=4C=3)C=C2)C=NC=C1"
        ),
        "ls2": "C1=CC(=NC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "ls3": "C1=CC(=C(C(=C1)C2=CC=NC=C2)N)C3=CC=NC=C3",
        "ls4": "N1=CC=C(C2=C(OC)C(C3=CC=NC=C3)=CC=C2)C=C1",
        "ls5": "C1=CC(=C(C(=C1)C2=CC=NC=C2)O)C3=CC=NC=C3",
        "ls7": "C1(C(C2C=CN=CC=2)=CC=CC=1C1C=CN=CC=1)OC(=O)C(C)(C)C",
        "ls8": "C1(C(C2C=CN=CC=2)=CC=CC=1C1C=CN=CC=1)OC(=O)C1C=CC=CC=1",
        "ls10": "C1=CN=CC=C1C2=CC=C([Se]2)C3=CC=NC=C3",
        # Case study 1.
        # Diverging-tarzia_2024.
        "l1": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "l2": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "l3": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        # Converging-tarzia_2024.
        "la": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "lb": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "lc": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "ld": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e10": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=C1"
        ),
        "e11": ("C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1"),
        "e12": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
        "e16": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC=C1"
        ),
        "e17": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
        ),
        # Case study 3.
        "cs3_l1": "C1=CC(=CC(=C1)C(=O)O)/C=C/C2=CC(=CC=C2)C(=O)O",
        "cs3_l2": "C1=CC(=CC(=C1)N=NC2=CC=CC(=C2)C(=O)O)C(=O)O",
        "cs3_l1p": "C1=CC(=CC(=C1)C(=O)O)C(=O)O",
        "cs3_l6p": "C1=CC(=CC2=C1C=CC(=C2)C(=O)O)C(=O)O",
        # Case study 4.
        "cs4_1": "C1=CC(=NC(=C1)C2=CC=C(C=C2)C(=O)O)C3=CC=C(C=C3)C(=O)O",
        "cs4_90": "C1=CC2=C(C=C1C(=O)O)C3=C(N2)C=CC(=C3)C(=O)O",
        # Case study 5.
        "cs5_lin": "C1=CN=CC=C1C2=CC=NC=C2",
        "cs5_mxy": (
            "C1=CC(=CC(=C1)CN2C=CC(=N2)C3=CC=NC=C3)CN4C=CC(=N4)C5=CC=NC=C5"
        ),
        "cs5_pxy": (
            "C1=CC(=CC=C1CN2C=C(C=N2)C3=CC=NC=C3)CN4C=C(C=N4)C5=CC=NC=C5"
        ),
        # Case study 6, only non sterics.
        "cs6_l1": "C(=CC(=O)O)C(=O)O",
        "cs6_l2": "C1=CC(=CC=C1C(=O)O)C(=O)O",
        "cs6_l5": "C1=CC2=C(C=CC(=C2)C(=O)O)C=C1C(=O)O",
        "cs6_l6": "C1=C(SC(=C1)C=CC(=O)O)C=CC(=O)O",
        "cs6_l9": "C1=CC(=CC=C1C2=CC=C(C=C2)C(=O)O)C3=CC=C(C=C3)C(=O)O",
        "cs6_cc31": "C1=C(C=C(C=C1C=O)C=O)C=O",
        "cs6_cc32": "C1CCC(C(C1)N)N",
    }

    ensembles = {}
    for ligand, lsmiles in ligands.items():
        if "cs3" in ligand or "cs4" in ligand or "cs6" in ligand:
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
