"""Script to plot property, input maps."""

import itertools as it
import json
import logging
import pathlib

import bbprep
import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import stk
import stko

from model_enumeration.utilities import (
    convert_topo,
    eb_str,
    isomer_energy,
    multi_cmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_xtal_data(  # noqa: C901, PLR0912, PLR0915
    figure_output: pathlib.Path,
    xtal_dir: pathlib.Path,
) -> dict:
    """Load and plot xtal data."""
    xtal_data_output = xtal_dir / "xtal_analysis.json"

    expected_structures = {
        "EVUXUR": {
            "topology": "4P82",
            "num_bcn": 16,
            "colour": "tab:red",
            "marker": "o",
        },
        "EVUYAY": {
            "topology": "6P122",
            "num_bcn": 24,
            "colour": "tab:red",
            "marker": "D",
        },
        "EVUYEC": {
            "topology": "4P82",
            "num_bcn": 16,
            "colour": "tab:red",
            "marker": "X",
        },
        "EVUYIG": {
            "topology": "4P82",
            "num_bcn": 16,
            "colour": "tab:red",
            "marker": "P",
        },
    }

    if xtal_data_output.exists():
        with xtal_data_output.open("r") as f:
            xtal_data = json.load(f)

    else:
        xtal_data = {}
        for xtal in expected_structures:
            molecule = stk.BuildingBlock.init_from_file(
                str(xtal_dir / f"{xtal}_single_molecule.mol")
            )
            organic_linkers = stko.molecule_analysis.DecomposeMOC().decompose(
                molecule=molecule,
                metal_atom_nos=(46,),
            )
            torsion_data = []
            crosser_internal_data = []
            outer_internal_data = []
            internal_data = []
            tta = stko.molecule_analysis.DitopicThreeSiteAnalyser()
            for i in organic_linkers:
                bb = stk.BuildingBlock.init_from_molecule(
                    molecule=i,
                    functional_groups=(
                        stko.functional_groups.ThreeSiteFactory(
                            "[#6]~[#7X2]~[#6]"
                        ),
                    ),
                )
                if bb.get_num_functional_groups() == 0:
                    continue
                bb = bbprep.FurthestFGs().modify(
                    building_block=bb,
                    desired_functional_groups=2,
                )

                torsion_data.append(abs(tta.get_binder_adjacent_torsion(bb)))
                int_angle = tta.get_binder_angles(bb)
                internal_data.append(
                    dict(zip(("NN_BCN1", "NN_BCN2"), int_angle, strict=True))
                )

            int_data = [i["NN_BCN1"] for i in internal_data] + [
                i["NN_BCN2"] for i in internal_data
            ]
            if len(int_data) != expected_structures[xtal]["num_bcn"]:
                raise RuntimeError

            cutoff = sum(int_data) / len(int_data)
            for i in internal_data:
                bcn1 = i["NN_BCN1"]
                bcn2 = i["NN_BCN2"]
                if bcn1 < cutoff:
                    if bcn2 > cutoff:
                        raise RuntimeError
                    outer_internal_data.append(bcn1)
                    outer_internal_data.append(bcn2)
                else:
                    if bcn2 < cutoff:
                        raise RuntimeError
                    crosser_internal_data.append(bcn1)
                    crosser_internal_data.append(bcn2)
            if (
                len(crosser_internal_data)
                != expected_structures[xtal]["num_bcn"] / 2
            ):
                raise RuntimeError
            if (
                len(outer_internal_data)
                != expected_structures[xtal]["num_bcn"] / 2
            ):
                raise RuntimeError
            xtal_data[xtal] = {
                "crosser_internal": crosser_internal_data,
                "outer_internal": outer_internal_data,
                "torsion": torsion_data,
                "topology": expected_structures[xtal]["topology"],
                "colour": expected_structures[xtal]["colour"],
                "marker": expected_structures[xtal]["marker"],
            }
        with xtal_data_output.open("w") as f:
            json.dump(xtal_data, f, indent=4)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for i, xtal in enumerate(xtal_data):
        axs[0].scatter(
            [i + 1 for j in range(len(xtal_data[xtal]["crosser_internal"]))],
            xtal_data[xtal]["crosser_internal"],
            c="tab:blue",
            s=160,
            edgecolor="k",
            label="across",
        )
        axs[0].scatter(
            [i + 1 for j in range(len(xtal_data[xtal]["outer_internal"]))],
            xtal_data[xtal]["outer_internal"],
            c="tab:orange",
            s=160,
            edgecolor="k",
            label="outer",
        )

        axs[1].scatter(
            [i + 1 for j in range(len(xtal_data[xtal]["torsion"]))],
            xtal_data[xtal]["torsion"],
            c="tab:green",
            s=160,
            edgecolor="k",
        )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_ylabel("binder angles [$^\\circ$]", fontsize=16)
    axs[0].set_ylim(None, 180)
    axs[0].set_xticks([i + 1 for i, _ in enumerate(xtal_data)])
    axs[0].set_xticklabels(list(xtal_data), rotation=45)
    axs[0].legend(fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_ylabel("|binder torsion| [$^\\circ$]", fontsize=16)
    axs[1].set_ylim(0, None)
    axs[1].set_xticks([i + 1 for i, _ in enumerate(xtal_data)])
    axs[1].set_xticklabels(list(xtal_data), rotation=45)

    fig.tight_layout()
    fig.savefig(
        figure_output / "li2023_xtal_distributions.png",
        dpi=720,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_output / "li2023_xtal_distributions.pdf",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    return xtal_data


def li2023_bite_angle(
    databases: tuple[pathlib.Path, ...],
    figure_output: pathlib.Path,
    xtal_dir: pathlib.Path,
) -> None:
    """Make a plot."""
    logging.info("running li2023_bite_angle")

    vmax = 1

    xtal_data = load_xtal_data(figure_output, xtal_dir)

    fig, axs = plt.subplots(
        ncols=len(databases),
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )
    for ax, database_path in zip(axs, databases, strict=True):
        database = cgx.utilities.AtomliteDatabase(database_path)
        tstr = database_path.name.strip(".db").split("_")[1]

        target_x = "$.forcefield_dict.v_dict.b_a_c"
        target_y = "$.forcefield_dict.v_dict.b_a_o"
        ax.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
        ax.set_ylabel("$bao$ [$^\\circ$]", fontsize=16)

        df_properties = [
            "$.energy_per_bb",
            "$.forcefield_dict.v_dict.b_a_c",
            "$.forcefield_dict.v_dict.b_a_o",
            "$.bb_dict_idx",
        ]

        vmin = 0
        vmax = 1
        dataframe = database.get_property_df(
            properties=df_properties,
            allow_missing=False,
        )
        logging.info("%s dataframe size: %s", tstr, len(dataframe))

        for xangle, yangle in it.product(
            set(dataframe[target_x]),
            set(dataframe[target_y]),
        ):
            pdata = dataframe.filter(pl.col(target_x) == xangle)
            pdata = pdata.filter(pl.col(target_y) == yangle)

            if len(pdata) == 0:
                continue

            min_energy = min(pdata["$.energy_per_bb"])

            ax.scatter(
                xangle,
                yangle,
                c=min_energy,
                alpha=1.0,
                edgecolor="k",
                s=160,
                marker="s",
                vmin=vmin,
                vmax=vmax,
                cmap="Blues_r",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(convert_topo(tstr), fontsize=16)

        # Add xtal data, always assuming smaller internal angles are
        # the outer ligands.
        for xtal in xtal_data:
            xd = xtal_data[xtal]
            ys = xd["crosser_internal"]
            xs = xd["outer_internal"]
            col = xd["colour"]
            m = xd["marker"]
            xtstr = xd["topology"]
            ax.scatter(
                xs,
                ys,
                c=col,
                alpha=1.0,
                marker=m,
                edgecolor="k",
                s=60,
                label=f"{xtal}:{convert_topo(xtstr)}",
                zorder=2,
            )
        ax.legend(fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"min {eb_str()}", fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_output / "li2023_ba.png", dpi=720, bbox_inches="tight")
    fig.savefig(figure_output / "li2023_ba.pdf", dpi=720, bbox_inches="tight")
    plt.close()


def li2023_ss(  # noqa: C901, PLR0912, PLR0915
    databases: tuple[pathlib.Path, ...],
    figure_output: pathlib.Path,
    xtal_dir: pathlib.Path,
) -> None:
    """Make a plot."""
    logging.info("running li2023_ss")

    xtal_data = load_xtal_data(figure_output, xtal_dir)
    df_properties = [
        "$.energy_per_bb",
        "$.forcefield_dict.v_dict.b_a_c",
        "$.forcefield_dict.v_dict.b_a_o",
        "$.bb_dict_idx",
    ]

    dataframe1 = cgx.utilities.AtomliteDatabase(databases[0]).get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    tstr1 = databases[0].name.strip(".db").split("_")[1]
    dataframe2 = cgx.utilities.AtomliteDatabase(databases[1]).get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    tstr2 = databases[1].name.strip(".db").split("_")[1]

    target_x = "$.forcefield_dict.v_dict.b_a_c"
    target_y = "$.forcefield_dict.v_dict.b_a_o"

    logging.info("%s dataframe size: %s", tstr1, len(dataframe1))
    logging.info("%s dataframe size: %s", tstr2, len(dataframe2))

    xs = set(list(dataframe1[target_x]) + list(dataframe2[target_x]))
    ys = set(list(dataframe1[target_y]) + list(dataframe2[target_y]))

    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

    labels = set()
    for xangle, yangle in it.product(xs, ys):
        pdata1 = dataframe1.filter(pl.col(target_x) == xangle)
        pdata1 = pdata1.filter(pl.col(target_y) == yangle)

        pdata2 = dataframe2.filter(pl.col(target_x) == xangle)
        pdata2 = pdata2.filter(pl.col(target_y) == yangle)

        if len(pdata1) != 0:
            pdata1 = pdata1.filter(
                pl.col("$.energy_per_bb") <= isomer_energy()
            )

        if len(pdata2) != 0:
            pdata2 = pdata2.filter(
                pl.col("$.energy_per_bb") <= isomer_energy()
            )

        if len(pdata1) == 0 and len(pdata2) == 0:
            colour = "white"
            alpha = 0.2
            string = None

        elif len(pdata1) > 1 and len(pdata2) > 1:
            colour = "tab:blue"
            alpha = 0.2
            string = "both > 1"

        elif len(pdata1) == 1 and len(pdata2) == 1:
            colour = "tab:orange"
            alpha = 1
            string = "both = 1"

        elif len(pdata1) == 1:
            colour = multi_cmap["4"]
            alpha = 1
            string = f"{convert_topo('4P82')} = 1"

        elif len(pdata2) == 1:
            colour = multi_cmap["6"]
            alpha = 1
            string = f"{convert_topo('6P122')} = 1"

        elif len(pdata1) > 1:
            colour = multi_cmap["4"]
            alpha = 0.2
            string = f"{convert_topo('4P82')} > 1"

        elif len(pdata2) > 1:
            colour = multi_cmap["6"]
            alpha = 0.2
            string = f"{convert_topo('6P122')} > 1"

        label = string if string not in labels and string is not None else None
        labels.add(label)
        ax.scatter(
            xangle,
            yangle,
            c=colour,
            alpha=alpha,
            edgecolor="none",
            s=200,
            marker="s",
            label=label,
        )

        # Print file name to visualise.
        if xangle == 95 and yangle == 165:  # noqa: PLR2004
            logging.info(
                "x:%s, y:%s, key: %s", xangle, yangle, pdata2["key"].item()
            )
        elif xangle == 100 and yangle == 145:  # noqa: PLR2004
            logging.info(
                "x:%s, y:%s, key: %s", xangle, yangle, pdata1["key"].item()
            )
        elif xangle == 100 and yangle == 140:  # noqa: PLR2004
            logging.info(
                "x:%s, y:%s, key: %s", xangle, yangle, pdata2["key"].item()
            )

    # Add xtal data, always assuming smaller internal angles are
    # the outer ligands.
    for xtal in xtal_data:
        xd = xtal_data[xtal]
        ys = xd["crosser_internal"]
        xs = xd["outer_internal"]
        col = xd["colour"]
        m = xd["marker"]
        xtstr = xd["topology"]
        ax.scatter(
            xs,
            ys,
            c=col,
            alpha=1.0,
            marker=m,
            edgecolor="k",
            s=60,
            label=f"{xtal}: {convert_topo(xtstr)}",
            zorder=2,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
    ax.set_ylabel("$bao$ [$^\\circ$]", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_output / "li2023_ss.png", dpi=720, bbox_inches="tight")
    fig.savefig(figure_output / "li2023_ss.pdf", dpi=720, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures" / "li2023"
    figure_dir.mkdir(exist_ok=True)
    data_dir = wd / "angle_data"
    xtal_dir = wd / "xtals"
    databases = (data_dir / "hr_4P82.db", data_dir / "hr_6P122.db")

    li2023_ss(databases, figure_dir, xtal_dir)
    li2023_bite_angle(databases, figure_dir, xtal_dir)


if __name__ == "__main__":
    main()
