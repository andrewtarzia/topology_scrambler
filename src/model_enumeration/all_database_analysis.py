"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
from collections import defaultdict

import atomlite
import cgexplore as cgx
import matplotlib.pyplot as plt
import stk
from matplotlib.lines import Line2D
from utilities import eb_str, isomer_energy, pore_str

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
        "at": ("tab:blue", "scan"),
        "kt": ("tab:blue", "scan"),
        "lt": ("tab:blue", "scan"),
        "tt": ("tab:blue", "scan"),
        "st": ("tab:purple", "torsion"),
        "dval": ("tab:orange", "nodoubles"),
        "ival": ("tab:green", "graph"),
        "ufo": ("tab:red", "ufo"),
    }

    option_xs = defaultdict(list)
    option_ys = defaultdict(list)

    for entry in fit_db.get_entries():
        db_name = entry.properties["db_name"]
        energy = entry.properties["energy_per_bb"]
        min_distance = entry.properties["min_distance"]

        if db_name[:2] in cmap:
            prefix = db_name[:2]
        elif db_name[:3] in cmap:
            prefix = db_name[:3]
        else:
            prefix = db_name[:4]
        option_xs[prefix].append(min_distance)
        option_ys[prefix].append(energy)

    legend_elements = []
    for option, (c, labl) in cmap.items():
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

        if option in ("kt", "lt", "tt"):
            continue
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
    fig, ax = plt.subplots(figsize=(5, 5))

    stages = ("opt1", "nx0", "nx1", "nx2", "nx3", "shifted", "smd")

    lowe_sources = {i: 0 for i in stages}  # Produces low energy structures.
    for entry in fit_db.get_entries():
        # Skip those built from templates.
        if "temp" in entry.properties["source"]:
            continue
        lowe_sources[entry.properties["source"]] += 1

    ax.bar(
        stages,
        [lowe_sources[i] for i in stages],
        color="#086788",
        edgecolor="k",
        lw=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)  # , color=color)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    ax.set_xlabel("stage", fontsize=16)
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


def compute_fitness(entry: atomlite.Entry, db_name: str) -> int:  # noqa: C901, PLR0911, PLR0912
    """Return fitness value. 0 for not, 1 for fit."""
    try:
        if entry.properties["energy_per_bb"] > isomer_energy():
            return 0
    except KeyError:
        return 0

    if db_name in ("dvalidation_run", "ivalidation_run", "ufo"):
        # Only energy for this one.
        return 1

    if db_name[:2] == "at" or db_name[:2] == "tt":
        angle_diff = abs(
            entry.properties["forcefield_dict"]["v_dict"]["b_a_c"]
            - entry.properties["forcefield_dict"]["v_dict"]["b_a_o"]
        )
        if angle_diff > 10:  # noqa: PLR2004
            return 1
        return 0

    if db_name[:2] == "kt":
        if "b_m_b" in entry.properties["forcefield_dict"]["v_dict"]:
            target = 90
            angle_str = "b_m_b"
        elif "b_n_b" in entry.properties["forcefield_dict"]["v_dict"]:
            target = 120
            angle_str = "b_n_b"
        angle_diff = abs(
            target - entry.properties["forcefield_dict"]["v_dict"][angle_str]
        )
        if angle_diff > 10:  # noqa: PLR2004
            return 1
        return 0

    if db_name[:2] == "lt":
        if "b_m_b" in entry.properties["forcefield_dict"]["v_dict"]:
            angle_str1 = "b_m_b"
            angle_str2 = "b_y_b"
        elif "b_n_b" in entry.properties["forcefield_dict"]["v_dict"]:
            angle_str1 = "b_n_b"
            angle_str2 = "b_x_b"
        angle_diff = abs(
            entry.properties["forcefield_dict"]["v_dict"][angle_str1]
            - entry.properties["forcefield_dict"]["v_dict"][angle_str2]
        )

        if angle_diff > 10:  # noqa: PLR2004
            return 1
        return 0

    if db_name[:2] == "st":
        num_states = entry.properties["dihedral_num_states"]

        if num_states == 3:  # noqa: PLR2004
            return 1
        return 0

    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912
    """Run script."""
    args = _parse_args()
    raise SystemExit("Change paths")
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    raise SystemExit("rerun")
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    fit_db_path = figure_dir / "fit_structures.db"
    fit_db = cgx.utilities.AtomliteDatabase(fit_db_path)
    database_paths = {
        # Angle hunter.
        "at1r1_3P6": wd / "desymm_finder" / "outputdata" / "at1r1_3P6.db",
        "at1r1_4P82": wd / "desymm_finder" / "outputdata" / "at1r1_4P82.db",
        "at1r1_4P8": wd / "desymm_finder" / "outputdata" / "at1r1_4P8.db",
        "at1r1_6P12": wd / "desymm_finder" / "outputdata" / "at1r1_6P12.db",
        "at1r1_8P16": wd / "desymm_finder" / "outputdata" / "at1r1_8P16.db",
        "at2r1_3P6": wd / "desymm_finder" / "outputdata" / "at2r1_3P6.db",
        "at2r1_6P12": wd / "desymm_finder" / "outputdata" / "at2r1_6P12.db",
        # "at3r1_3P6":
        # wd / "desymm_finder" / "outputdata" / "at3r1_3P6.db",  # noqa: ERA001
        # "at3r1_6P12":
        # wd / "desymm_finder" / "outputdata" / "at3r1_6P12.db",# noqa: ERA001
        "at4r1_6P122": wd / "desymm_finder" / "outputdata" / "at4r1_6P122.db",
        "at5r1_2P3": wd / "desymm_finder" / "outputdata" / "at5r1_2P3.db",
        "at5r1_4P62": wd / "desymm_finder" / "outputdata" / "at5r1_4P62.db",
        "at5r1_4P6": wd / "desymm_finder" / "outputdata" / "at5r1_4P6.db",
        # Removed due to noise.
        # "at5r1_6P9":
        # wd / "desymm_finder" / "outputdata" / "at5r1_6P9.db",# noqa: ERA001
        # "at5r1_8P12":
        # wd / "desymm_finder" / "outputdata" / "at5r1_8P12.db",# noqa: ERA001
        "at6r1_4P6": wd / "desymm_finder" / "outputdata" / "at6r1_4P6.db",
        "at7r1_4P6": wd / "desymm_finder" / "outputdata" / "at7r1_4P6.db",
        "at8r1_4P62": wd / "desymm_finder" / "outputdata" / "at8r1_4P62.db",
        # # Angle hunter large-small.
        "kt1r1_6P12": wd / "desymm_finder" / "outputdata" / "kt1r1_6P12.db",
        "kt2r1_4P6": wd / "desymm_finder" / "outputdata" / "kt2r1_4P6.db",
        # # Angle hunter large.
        "lt1r1_6P12": wd / "desymm_finder" / "outputdata" / "lt1r1_6P12.db",
        "lt2r1_4P6": wd / "desymm_finder" / "outputdata" / "lt2r1_4P6.db",
        # Environment hunter.
        "st1_2P4": wd / "desymm_finder" / "outputdata" / "st1_2P4.db",
        "st1_3P6": wd / "desymm_finder" / "outputdata" / "st1_3P6.db",
        "st1_4P82": wd / "desymm_finder" / "outputdata" / "st1_4P82.db",
        "st1_4P8": wd / "desymm_finder" / "outputdata" / "st1_4P8.db",
        "st1_6P122": wd / "desymm_finder" / "outputdata" / "st1_6P122.db",
        "st1_6P12": wd / "desymm_finder" / "outputdata" / "st1_6P12.db",
        "st1_8P162": wd / "desymm_finder" / "outputdata" / "st1_8P162.db",
        "st1_8P16": wd / "desymm_finder" / "outputdata" / "st1_8P16.db",
        "st3_2P3": wd / "desymm_finder" / "outputdata" / "st3_2P3.db",
        "st3_4P62": wd / "desymm_finder" / "outputdata" / "st3_4P62.db",
        "st3_4P6": wd / "desymm_finder" / "outputdata" / "st3_4P6.db",
        "st3_6P9": wd / "desymm_finder" / "outputdata" / "st3_6P9.db",
        "st3_8P12": wd / "desymm_finder" / "outputdata" / "st3_8P12.db",
        # Torsion hunter.
        "tt1r1_4P6": wd / "desymm_finder" / "outputdata" / "tt1r1_4P6.db",
        "tt1r1_8P16": wd / "desymm_finder" / "outputdata" / "tt1r1_8P16.db",
        # # Model enumeration.
        "dvalidation_run": wd / "dvalidation_data" / "dvalidation_run.db",
        "ivalidation_run": wd / "ivalidation_data" / "ivalidation_run.db",
        "ufo": wd / "ufo_data" / "ufo.db",
    }

    if args.run:
        for db_name, db_path in database_paths.items():
            total_fit_structures = 0
            db = cgx.utilities.AtomliteDatabase(db_path)
            num_entries = db.get_num_entries()

            for entry in db.get_entries():
                if fit_db.has_molecule(entry.key):
                    continue

                fitness = compute_fitness(entry, db_name)

                if fitness == 1:
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
