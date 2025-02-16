"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
from collections import defaultdict

import atomlite
import cgexplore as cgx
import matplotlib.pyplot as plt
import polars as pl
import stk
from matplotlib.lines import Line2D

from model_enumeration.utilities import (
    convert_topo,
    eb_str,
    isomer_energy,
    pore_str,
    topology_cmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def make_plot(
    fit_db: cgx.utilities.AtomliteDatabase,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = {
        "hr": ("tab:blue", "scan"),
        "st": ("tab:purple", "torsion"),
        "dval": ("tab:orange", "mn2ln-nodoubles"),
        "ival": ("tab:green", "mn2ln"),
        "mgen": ("tab:red", "prediction"),
    }

    option_xs = defaultdict(list)
    option_ys = defaultdict(list)

    for entry in fit_db.get_entries():
        db_name = entry.properties["db_name"]
        energy = entry.properties["energy_per_bb"]
        min_distance = entry.properties["min_distance"]

        if "hr_" in db_name:
            prefix = "hr"
        elif "st1_" in db_name or "st3_" in db_name:
            prefix = "st"

        elif "dvalidation" in db_name:
            prefix = "dval"
        elif "ivalidation" in db_name:
            prefix = "ival"
        elif "mgen_" in db_name:
            prefix = "mgen"
        else:
            msg = "Unknown prefix %s", db_name
            raise ValueError(msg)

        option_xs[prefix].append(min_distance)
        option_ys[prefix].append(energy)

    legend_elements = []
    for option, (c, labl) in cmap.items():
        if option in ("st",):
            continue
        ax.scatter(
            option_xs[option],
            option_ys[option],
            marker="o",
            c=c,
            s=50,
            ec="k" if labl != "scan" else "none",
            alpha=1,
            rasterized=True,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=labl,
                markerfacecolor=c,
                markersize=8,
                markeredgecolor="k" if labl != "scan" else "none",
                alpha=1,
            )
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(pore_str(), fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_xlim(0, 10)

    ax.legend(handles=legend_elements, ncols=1, fontsize=16)
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


def make_opt_plot(
    fit_db: cgx.utilities.AtomliteDatabase,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise stage of the optimisation produces the low-E conformer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    lowe_sources = {}  # Produces low energy structures.
    for entry in fit_db.get_entries():
        # Skip those built from templates.
        if "temp" in entry.properties["source"]:
            continue
        if entry.properties["source"] not in lowe_sources:
            lowe_sources[entry.properties["source"]] = 0
        lowe_sources[entry.properties["source"]] += 1

    ax.bar(
        sorted(lowe_sources),
        [lowe_sources[i] for i in sorted(lowe_sources)],
        color="#086788",
        edgecolor="k",
        lw=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xticks(range(len(lowe_sources)))
    ax.set_xticklabels(sorted(lowe_sources), rotation=45)
    ax.set_yscale("log")

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


def make_section2_plot(  # noqa: PLR0915
    fit_db: cgx.utilities.AtomliteDatabase,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax, ax2) = plt.subplots(
        ncols=2,
        sharey=True,
        sharex=True,
        figsize=(10, 5),
    )

    target_x = "$.forcefield_dict.v_dict.b_a_c"
    target_y = "$.forcefield_dict.v_dict.b_a_o"

    df_properties = [
        "$.energy_per_bb",
        "$.db_name",
        "$.tstr",
        "$.forcefield_dict.v_dict.b_a_c",
        "$.forcefield_dict.v_dict.b_a_o",
    ]

    dataframe = fit_db.get_property_df(
        properties=df_properties,
        allow_missing=False,
    )
    dataframe = dataframe.filter(pl.col("$.db_name").str.contains("hr_"))
    logging.info("dataframe size: %s", len(dataframe))

    excluded_tstrs = ("4P6", "4P62")

    labels = set()
    to_visualise = []
    for xangle, yangle in it.product(
        set(dataframe[target_x]),
        set(dataframe[target_y]),
    ):
        pdata = dataframe.filter(pl.col(target_x) == xangle)
        pdata = pdata.filter(pl.col(target_y) == yangle)

        if len(pdata) == 0:
            continue

        pdata = pdata.filter(pl.col("$.energy_per_bb") <= isomer_energy())

        if len(pdata) > 10:  # noqa: PLR2004
            colours = ["white"]
            kinetic_colours = ["white"]

        elif len(pdata) == 1:
            found_tstr = pdata["$.tstr"].item()
            if found_tstr in excluded_tstrs:
                colours = ["white"]
                kinetic_colours = ["white"]
            else:
                colours = [topology_cmap[found_tstr]]
                kinetic_colours = [topology_cmap[found_tstr]]
                labels.add(found_tstr)
                to_visualise.append(pdata["key"].item())

        else:
            found_names = list(pdata["key"])
            found_tstrs = list(pdata["$.tstr"])

            colours = []
            stoichs = []
            for tstr in found_tstrs:
                colours.append(topology_cmap[tstr])
                labels.add(tstr)
                stoich = cgx.topologies.stoich_map(tstr)
                stoichs.append(stoich)
                if tstr in excluded_tstrs:
                    continue

            min_stoich = min(stoichs)
            kinetic_colours = list(
                {
                    i
                    for i, stoich in zip(colours, stoichs, strict=False)
                    if stoich == min_stoich
                }
            )
            to_visualise.extend(
                {
                    i
                    for i, stoich in zip(found_names, stoichs, strict=False)
                    if stoich == min_stoich
                }
            )

        cgx.utilities.draw_pie(
            colours=colours,
            xpos=xangle,
            ypos=yangle,
            size=150,
            ax=ax,
        )
        cgx.utilities.draw_pie(
            colours=kinetic_colours,
            xpos=xangle,
            ypos=yangle,
            size=150,
            ax=ax2,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=convert_topo(labl),
            markerfacecolor=topology_cmap[labl],
            markersize=8,
            markeredgecolor="k",
            alpha=1,
        )
        for labl in sorted(labels)
    ]

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
    ax.set_ylabel("$bao$ [$^\\circ$]", fontsize=16)
    ax.legend(handles=legend_elements, ncols=1, fontsize=16)
    ax.plot((90, 180), (90, 180), color="black", linestyle="--")
    ax.set_xlim(87, 143)
    ax.set_ylim(97, 180)

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_xlabel("$bac$ [$^\\circ$]", fontsize=16)
    ax2.set_ylabel("$bao$ [$^\\circ$]", fontsize=16)
    ax2.plot((90, 180), (90, 180), color="black", linestyle="--")

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

    logging.info("visualising %s structures", len(to_visualise))
    draw_cages(
        to_visualise=to_visualise,
        fit_db=fit_db,
        figure_dir=figure_dir,
    )


def draw_cages(
    to_visualise: list[str],
    fit_db: cgx.utilities.AtomliteDatabase,
    figure_dir: pathlib.Path,
) -> None:
    """Draw structures to image."""
    struct_figure_output = figure_dir / "structures"
    struct_figure_output.mkdir(parents=True, exist_ok=True)

    for sname in to_visualise:
        x = fit_db.get_entry(sname).properties["forcefield_dict"]["v_dict"][
            "b_a_c"
        ]
        y = fit_db.get_entry(sname).properties["forcefield_dict"]["v_dict"][
            "b_a_o"
        ]

        mol_file = struct_figure_output / f"{sname}_{x}_{y}.mol"
        fit_db.get_molecule(sname).write(mol_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    figure_dir = wd / "figures" / "all_database_analysis"
    figure_dir.mkdir(exist_ok=True)

    fit_db_path = figure_dir / "fit_structures.db"
    fit_db = cgx.utilities.AtomliteDatabase(fit_db_path)

    database_paths = {
        # Angle hunter.
        "hr_3P6": wd / "angle_data" / "hr_3P6.db",
        "hr_4P6": wd / "angle_data" / "hr_4P6.db",
        "hr_4P62": wd / "angle_data" / "hr_4P62.db",
        "hr_4P82": wd / "angle_data" / "hr_4P82.db",
        "hr_4P8": wd / "angle_data" / "hr_4P8.db",
        "hr_6P12": wd / "angle_data" / "hr_6P12.db",
        "hr_6P122": wd / "angle_data" / "hr_6P122.db",
        "hr_8P16": wd / "angle_data" / "hr_8P16.db",
        # Environment hunter.
        "st1_2P4": wd / "envi_data" / "st1_2P4.db",
        "st1_3P6": wd / "envi_data" / "st1_3P6.db",
        "st1_4P82": wd / "envi_data" / "st1_4P82.db",
        "st1_4P8": wd / "envi_data" / "st1_4P8.db",
        "st1_6P122": wd / "envi_data" / "st1_6P122.db",
        "st1_6P12": wd / "envi_data" / "st1_6P12.db",
        "st1_8P162": wd / "envi_data" / "st1_8P162.db",
        "st1_8P16": wd / "envi_data" / "st1_8P16.db",
        "st3_2P3": wd / "envi_data" / "st3_2P3.db",
        "st3_4P62": wd / "envi_data" / "st3_4P62.db",
        "st3_4P6": wd / "envi_data" / "st3_4P6.db",
        "st3_6P9": wd / "envi_data" / "st3_6P9.db",
        "st3_8P12": wd / "envi_data" / "st3_8P12.db",
        # Model enumeration.
        "dvalidation_run": wd / "dvalidation_data" / "dvalidation_run.db",
        "ivalidation_run": wd / "ivalidation_data" / "ivalidation_run.db",
        "mgen": wd / "mgen_data" / "mgen.db",
    }

    if args.run:
        for db_name, db_path in database_paths.items():
            total_fit_structures = 0
            if not db_path.exists():
                msg = f"database {db_name} not found."
                raise FileNotFoundError(msg)
            db = cgx.utilities.AtomliteDatabase(db_path)
            num_entries = db.get_num_entries()
            logging.info(
                "processing database %s with %s entries", db_name, num_entries
            )

            for entry in db.get_entries():
                if fit_db.has_molecule(entry.key):
                    continue
                if "energy_per_bb" not in entry.properties:
                    continue

                energy = entry.properties["energy_per_bb"]
                if energy < isomer_energy():
                    total_fit_structures += 1
                    if "min_distance" not in entry.properties:
                        db.add_properties(
                            key=entry.key,
                            property_dict={
                                "min_distance": (
                                    cgx.analysis.GeomMeasure().calculate_min_distance(
                                        db.get_molecule(key=entry.key)
                                    )["min_distance"]
                                ),
                            },
                        )
                        # Reload.
                        fit_entry = db.get_entry(entry.key)
                    else:
                        fit_entry = entry

                    fit_db.add_molecule(
                        molecule=db.get_molecule(key=fit_entry.key),
                        key=fit_entry.key,
                    )
                    fit_db.add_properties(
                        key=fit_entry.key,
                        property_dict=fit_entry.properties,
                    )
                    fit_db.add_properties(
                        key=fit_entry.key,
                        property_dict={"db_name": db_name},
                    )

            logging.info(
                "there are %s FIT structures of %s in %s",
                total_fit_structures,
                num_entries,
                db_name,
            )

    logging.info(
        "there are %s FIT structuresin total",
        fit_db.get_num_entries(),
    )

    make_section2_plot(
        fit_db=fit_db,
        figure_dir=figure_dir,
        filename="alldb_3_section2.png",
    )
    make_opt_plot(
        fit_db=fit_db,
        figure_dir=figure_dir,
        filename="alldb_1.png",
    )

    make_plot(
        fit_db=fit_db,
        figure_dir=figure_dir,
        filename="alldb_2.png",
    )

    raise SystemExit(
        "Do a specific outcome just for the angle scans for section 2 of paper"
    )

    # Write to chemiscope.
    properties_to_get = {
        "E_b / kjmol-1": {
            "path": ["energy_per_bb"],
            "function": None,
        },
        "prefix": {
            "path": None,
            "function": None,
        },
        "pore_radius / AA": {
            "path": ["min_distance"],
            "function": None,
        },
        "fitness ": {
            "path": ["all_fitness"],
            "function": None,
        },
    }

    prefixes = {"at", "kt", "st", "lt", "tt", "dval", "ival", "ufo"}
    structures = []
    properties = {}
    for entry in fit_db.get_entries():
        db_name = entry.properties["db_name"]
        structures.append(
            stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            )
        )
        for prop in properties_to_get:
            if prop == "prefix":
                if db_name[:2] in prefixes:
                    value = db_name[:2]
                elif db_name[:3] in prefixes:
                    value = db_name[:3]
                else:
                    value = db_name[:4]
            else:
                value = cgx.utilities.extract_property(
                    path=properties_to_get[prop]["path"],
                    properties=entry.properties,
                )
            if prop not in properties:
                properties[prop] = []
            properties[prop].append(value)

    cgx.utilities.write_chemiscope_json(
        json_file=wd / "figures" / "fit_structures.json.gz",
        structures=structures,
        properties=properties,
        bonds_as_shapes=True,
        meta_dict={
            "name": "model-enumeration: all data",
            "description": ("Minimal models"),
            "authors": ["Andrew Tarzia"],
            "references": ["TBD"],
        },
        x_axis_dict={"property": "pore_radius / AA"},
        y_axis_dict={"property": "E_b / kjmol-1"},
        z_axis_dict={"property": ""},
        color_dict={"property": "prefix"},
        bond_hex_colour="#919294",
    )


if __name__ == "__main__":
    main()
