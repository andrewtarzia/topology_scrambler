"""Script to plot geometrical distributions."""

import argparse
import logging
import pathlib

import cgexplore as cgx
import matplotlib.pyplot as plt
from ds_utilities import EnvVariables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def geom_distributions(
    prefix: str,
    database: cgx.utilities.AtomliteDatabase,
    figure_output: pathlib.Path,
) -> None:
    """Plot geometry distributions."""
    comparisons = {
        "Pb_Ba_Ag_Ba_Pb": {
            "xlabel": r"$bacab$ [$^\circ$]",
            "measure": "dihedral_data",
            "ylim": (0, 180),
        },
        "Pb_Ba_O_Ba_Pb": {
            "xlabel": r"$baoab$ [$^\circ$]",
            "measure": "dihedral_data",
            "ylim": (0, 180),
        },
        "Pb_Pd_Pb": {
            "xlabel": r"$mbm$ [$^\circ$]",
            "measure": "angle_data",
            "ylim": (0, 180),
        },
        "Pb_Ba_Ag": {
            "xlabel": r"$bac$ [$^\circ$]",
            "measure": "angle_data",
            "ylim": (90, 180),
        },
        "Pb_Ba_O": {
            "xlabel": r"$bao$ [$^\circ$]",
            "measure": "angle_data",
            "ylim": (90, 180),
        },
        "Ba_Ag_Ba": {
            "xlabel": r"$aca$ [$^\circ$]",
            "measure": "angle_data",
            "ylim": (150, 180),
        },
        "Ba_O_Ba": {
            "xlabel": r"$aoa$ [$^\circ$]",
            "measure": "angle_data",
            "ylim": (150, 180),
        },
        "Pb_Pd": {
            "xlabel": r"$mb$ [$\mathrm{\AA}$]",
            "measure": "bond_data",
            "ylim": (0, 2),
        },
        "Ag_Ba": {
            "xlabel": r"$ac$ [$\mathrm{\AA}$]",
            "measure": "bond_data",
            "ylim": (0, 2),
        },
        "Ba_O": {
            "xlabel": r"$ao$ [$\mathrm{\AA}$]",
            "measure": "bond_data",
            "ylim": (0, 2),
        },
        "Ba_Pb": {
            "xlabel": r"$ba$ [$\mathrm{\AA}$]",
            "measure": "bond_data",
            "ylim": (0, 2),
        },
    }

    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(16, 10))
    flat_axs = axs.flatten()
    geom_dict = {}
    for entry in database.get_entries():
        if "dihedral_states" not in entry.properties:
            continue

        for label in comparisons:
            if label in entry.properties[comparisons[label]["measure"]]:
                gd_data = entry.properties[comparisons[label]["measure"]][
                    label
                ]
                if label not in geom_dict:
                    geom_dict[label] = []

                geom_dict[label].extend(gd_data)

    for label, ax in zip(comparisons, flat_axs, strict=False):
        try:
            xdata = geom_dict[label]
        except KeyError:
            continue

        parts = ax.violinplot(
            xdata,
            vert=True,
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("#086788")
            pc.set_edgecolor("none")
            pc.set_alpha(1.0)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(comparisons[label]["xlabel"], fontsize=16)
        ax.set_ylim(comparisons[label]["ylim"])
        ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(
        figure_output / f"gd_{prefix}.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("database_path", help="database to analyse")
    return parser.parse_args()


def main() -> None:
    """Run script."""
    raise SystemExit("rerun")
    args = _parse_args()
    raise SystemExit("Change paths")
    database = cgx.utilities.AtomliteDatabase(db_file=args.database_path)
    logging.info("there are %s collected data", database.get_num_entries())
    prefix = (
        pathlib.Path(args.database_path).absolute().name.replace(".db", "")
    )

    geom_distributions(
        prefix=prefix,
        database=database,
        figure_output=EnvVariables.cg_figures,
    )


if __name__ == "__main__":
    main()
