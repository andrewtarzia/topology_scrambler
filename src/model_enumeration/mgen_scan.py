"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import warnings
from collections import abc

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import stk
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)
from model_enumeration.utilities import eb_str, multi_cmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    database.add_properties(
        key=name,
        property_dict={
            "forcefield_dict": forcefield.get_forcefield_dictionary(),
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=properties["energy_decomposition"],
                number_building_blocks=num_building_blocks,
            ),
        },
    )

    g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
    bond_data = g_measure.calculate_bonds(final_molecule)
    bond_data = {str("_".join(i)): bond_data[i] for i in bond_data}
    angle_data = g_measure.calculate_angles(final_molecule)
    angle_data = {str("_".join(i)): angle_data[i] for i in angle_data}
    dihedral_data = g_measure.calculate_torsions(
        molecule=final_molecule,
        absolute=True,
    )
    database.add_properties(
        key=name,
        property_dict={
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
        },
    )

    ligands = stko.molecule_analysis.DecomposeMOC().decompose(
        molecule=final_molecule,
        metal_atom_nos=(46,),
    )

    # Get the bg angles.
    c_binder_binder_angles = []
    d_binder_binder_angles = []
    for lig in ligands:
        if lig.get_num_atoms() == 8:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ga]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = True
        elif lig.get_num_atoms() == 5:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ba]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = False

        if as_building_block.get_num_functional_groups() != 2:  # noqa: PLR2004
            raise RuntimeError

        vectors = [
            as_building_block.get_centroid(atom_ids=fg.get_bonder_ids())
            - as_building_block.get_centroid(atom_ids=fg.get_deleter_ids())
            for fg in as_building_block.get_functional_groups()
        ]
        normed = [i / np.linalg.norm(i) for i in vectors]
        angle = np.degrees(
            stko.vector_angle(vector1=normed[0], vector2=normed[1])
        )
        if converging:
            c_binder_binder_angles.append(angle)
        else:
            d_binder_binder_angles.append(angle)

    database.add_properties(
        key=name,
        property_dict={
            "converging_binder_binder_angles": c_binder_binder_angles,
            "diverging_binder_binder_angles": d_binder_binder_angles,
        },
    )


def make_energy_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    combos: dict[str, dict[str, str | abc.Iterable]],
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(
        ncols=len(combos),
        nrows=2,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )

    row_plot = (
        {
            "ffx": "b_a_c",
            "aay": "diverging_binder_binder_angles",
            "xlim": (None, None),
            "ylim": (None, None),
            "ylbl": eb_str(),
            "xlbl": "rigid angle  [$^\\circ$]",
            "obs_source": "bba",
        },
        {
            "ffx": "d_d_e",
            "aay": "converging_binder_binder_angles",
            "xlim": (None, None),
            "ylim": (None, None),
            "ylbl": eb_str(),
            "xlbl": "twistable angle  [$^\\circ$]",
            "obs_source": "bba",
        },
    )
    for axrow, rowd in zip(axs, row_plot, strict=True):
        for combo, ax in zip(combos, axrow, strict=True):
            xs = []
            ys = []

            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                ys.append(float(entry.properties["energy_per_bb"]))
                if rowd["obs_source"] == "ff":
                    try:
                        xs.append(entry.properties["angle_data"][rowd["aay"]])
                    except KeyError:
                        xs.append(entry.properties["bond_data"][rowd["aay"]])
                elif rowd["obs_source"] == "bba":
                    xs.append(np.mean(entry.properties[rowd["aay"]]))

            ax.scatter(
                xs,
                ys,
                c="tab:blue",
                alpha=1.0,
                edgecolor="k",
                s=60,
                zorder=2,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            if rowd["ffx"] == "b_a_c":
                ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            if combo == "bac-aa":
                ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])

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


def make_geom_plot(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    combos: dict[str, dict[str, str | abc.Iterable]],
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=7, nrows=5, figsize=(16, 16))

    row_plot = (
        {
            "ffx": "b_a_c",
            "aay": "Pb_Ba_Ag",
            "xlim": (90, 180),
            "ylim": (90, 180),
            "xlbl": "target $bac$  [$^\\circ$]",
            "ylbl": "observed $bac$  [$^\\circ$]",
            "obs_source": "ff",
        },
        {
            "ffx": "d_d_e",
            "aay": "Fe_Ni_Ni",
            "xlim": (90, 180),
            "ylim": (90, 180),
            "xlbl": "target $dde$  [$^\\circ$]",
            "ylbl": "observed $dde$  [$^\\circ$]",
            "obs_source": "ff",
        },
        {
            "ffx": "a_c",
            "aay": "Ag_Ba",
            "xlim": (0.5, 3),
            "ylim": (0.5, 3),
            "xlbl": r"target $ac$ [$\mathrm{\AA}$]",
            "ylbl": r"observed $ac$ [$\mathrm{\AA}$]",
            "obs_source": "ff",
        },
        {
            "ffx": "d_d",
            "aay": "Ni_Ni",
            "xlim": (1.5, 6),
            "ylim": (1.5, 6),
            "xlbl": r"target $dd$ [$\mathrm{\AA}$]",
            "ylbl": r"observed $dd$ [$\mathrm{\AA}$]",
            "obs_source": "ff",
        },
        {
            "ffx": "d_e",
            "aay": "Fe_Ni",
            "xlim": (0.5, 3),
            "ylim": (0.5, 3),
            "xlbl": r"target $de$ [$\mathrm{\AA}$]",
            "ylbl": r"observed $de$ [$\mathrm{\AA}$]",
            "obs_source": "ff",
        },
    )
    for axrow, rowd in zip(axs, row_plot, strict=True):
        for combo, ax in zip(combos, axrow, strict=True):
            xs = []
            ys = []

            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                xs.append(
                    float(
                        entry.properties["forcefield_dict"]["v_dict"][
                            rowd["ffx"]
                        ]
                    )
                )
                if rowd["obs_source"] == "ff":
                    try:
                        ys.append(entry.properties["angle_data"][rowd["aay"]])
                    except KeyError:
                        ys.append(entry.properties["bond_data"][rowd["aay"]])
                elif rowd["obs_source"] == "bba":
                    ys.append(entry.properties[rowd["aay"]])

            comp_values = {i: [] for i in sorted(set(xs))}
            for i, j in zip(xs, ys, strict=True):
                comp_values[i].extend(j)

            ax.scatter(
                list(comp_values),
                [np.mean(y) for y in comp_values.values()],
                c="tab:blue",
                alpha=1.0,
                edgecolor="k",
                s=60,
                zorder=2,
            )
            ax.fill_between(
                list(comp_values),
                y1=[np.min(comp_values[i]) for i in comp_values],
                y2=[np.max(comp_values[i]) for i in comp_values],
                alpha=0.6,
                color="tab:blue",
                edgecolor=(0, 0, 0, 2.0),
                lw=0,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            if rowd["ffx"] == "b_a_c":
                ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            if combo == "bac-aa":
                ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])
            ax.set_xticks([])
            ax.set_yticks([])
            if rowd["obs_source"] == "ff":
                ax.plot(rowd["xlim"], rowd["ylim"], c="k", zorder=-1)

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


def make_geom_grid(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    combos: dict[str, dict[str, str | abc.Iterable]],
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(
        ncols=4,
        nrows=2,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
    )
    flat_axs = axs.flatten()

    vmin = 0
    vmax = 0.6
    row_plot = (
        {
            "ffx": "diverging_binder_binder_angles",
            "aay": "converging_binder_binder_angles",
            "xlim": (None, None),
            "ylim": (None, None),
            "xlbl": "observed rigid angle  [$^\\circ$]",
            "ylbl": "observed twistable angle  [$^\\circ$]",
        },
    )
    for axrow, rowd in zip([flat_axs], row_plot, strict=False):
        for combo, ax in zip(combos, axrow, strict=False):
            min_stable_x = float("inf")
            max_stable_x = 0
            min_stable_y = float("inf")
            max_stable_y = 0
            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                xs = entry.properties[rowd["ffx"]]
                ys = entry.properties[rowd["aay"]]
                c = float(entry.properties["energy_per_bb"])
                if c < 0.1:  # noqa: PLR2004
                    zorder = 2
                    min_stable_x = min((min(xs), min_stable_x))
                    max_stable_x = max((max(xs), max_stable_x))
                    min_stable_y = min((min(ys), min_stable_y))
                    max_stable_y = max((max(ys), max_stable_y))

                elif c < 0.3:  # noqa: PLR2004
                    zorder = 1
                else:
                    zorder = 0

                ax.scatter(
                    np.mean(xs),
                    np.mean(ys),
                    c=c,
                    alpha=1.0,
                    edgecolor="k",
                    s=80,
                    zorder=zorder,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="Blues_r",
                )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])
            ax.axhspan(
                ymin=min_stable_y,
                ymax=max_stable_y,
                facecolor="k",
                alpha=0.2,
            )
            ax.axvspan(
                xmin=min_stable_x,
                xmax=max_stable_x,
                facecolor="k",
                alpha=0.2,
            )

        cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])  # type: ignore[call-overload]
        cmap = mpl.cm.Blues_r  # type: ignore[attr-defined]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(eb_str(), fontsize=16)
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


def make_contour_grid(  # noqa: C901, PLR0912, PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    combos: dict[str, dict[str, str | abc.Iterable]],
    as_contour: bool,
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 10))
    cg_scale = 2
    flat_axs = axs.flatten()

    for combo, ax in zip(combos, flat_axs, strict=False):
        if combo == "bac-aa":
            xoption = "a_c"
            xoption2 = "a_a"
            yoption = "b_a_c"
            experimental_xs = (
                (4.9 / (2 * cg_scale), 150),
                (4.7 / (2 * cg_scale), 155),
                (5.0 / (2 * cg_scale), 145),
                (5.0 / (2 * cg_scale), 150),
                (5.3 / (2 * cg_scale), 165),
                (5.4 / (2 * cg_scale), 167),
            )
            xlbl = r"$ac$  [$\mathrm{\AA}$]"
            ylbl = "$bac$  [$^\\circ$]"

        elif combo == "bac-dde":
            xoption = "d_d_e"
            xoption2 = None
            yoption = "b_a_c"
            experimental_xs = (
                (133, 150),
                (133, 155),
                (133, 145),
                (133, 165),
                (133, 167),
            )
            xlbl = "$dde$  [$^\\circ$]"
            ylbl = "$bac$  [$^\\circ$]"

        elif combo == "dd-dde":
            xoption = "d_d_e"
            xoption2 = None
            yoption = "d_d"
            experimental_xs = ((133, 8.0 / cg_scale),)
            xlbl = "$dde$  [$^\\circ$]"
            ylbl = r"$dd$  [$\mathrm{\AA}$]"

        elif combo == "bac-dd":
            xoption = "d_d"
            xoption2 = None
            yoption = "b_a_c"
            experimental_xs = (
                (8.0 / cg_scale, 150),
                (8.0 / cg_scale, 155),
                (8.0 / cg_scale, 145),
                (8.0 / cg_scale, 165),
                (8.0 / cg_scale, 167),
            )
            xlbl = r"$dd$  [$\mathrm{\AA}$]"
            ylbl = "$bac$  [$^\\circ$]"

        elif combo == "de-dde":
            xoption = "d_d_e"
            xoption2 = None
            yoption = "d_e"
            experimental_xs = ((133, 4.3 / cg_scale),)
            xlbl = "$dde$  [$^\\circ$]"
            ylbl = r"$de$  [$\mathrm{\AA}$]"

        elif combo == "de-dd":
            xoption = "d_d"
            xoption2 = None
            yoption = "d_e"
            experimental_xs = ((8.0 / cg_scale, 4.3 / cg_scale),)
            xlbl = r"$dd$  [$\mathrm{\AA}$]"
            ylbl = r"$de$  [$\mathrm{\AA}$]"

        elif combo == "bac-de":
            xoption = "d_e"
            xoption2 = None
            yoption = "b_a_c"
            experimental_xs = (
                (4.3 / cg_scale, 150),
                (4.3 / cg_scale, 155),
                (4.3 / cg_scale, 145),
                (4.3 / cg_scale, 165),
                (4.3 / cg_scale, 167),
            )
            xlbl = r"$de$  [$\mathrm{\AA}$]"
            ylbl = "$bac$  [$^\\circ$]"

        else:
            raise NotImplementedError

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(xlbl, fontsize=16)
        ax.set_ylabel(ylbl, fontsize=16)

        ax.scatter(
            [i[0] for i in experimental_xs],
            [i[1] for i in experimental_xs],
            c="tab:red",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="X",
            zorder=2,
        )

        if xoption2 is None:
            cname = f"{yoption.replace('_', '')}-{xoption.replace('_', '')}_"
        else:
            cname = f"{yoption.replace('_', '')}-{xoption2.replace('_', '')}_"

        frame = cgx.utilities.AtomliteDatabase(database_path).get_property_df(
            properties=[
                "$.energy_per_bb",
                f"$.forcefield_dict.v_dict.{xoption}",
                f"$.forcefield_dict.v_dict.{yoption}",
            ]
        )
        frame = frame.filter(pl.col("key").str.contains(cname))

        if frame.is_empty():
            continue

        if as_contour:
            min_energy_key = frame.filter(
                pl.col("$.energy_per_bb") == frame["$.energy_per_bb"].min()
            )["key"].item(0)
            logging.info("minimum energy key: %s", min_energy_key)

        xs = set(frame[f"$.forcefield_dict.v_dict.{xoption}"])
        ys = set(frame[f"$.forcefield_dict.v_dict.{yoption}"])
        # Plot the underlying grid.
        if as_contour:
            ax.scatter(
                frame[f"$.forcefield_dict.v_dict.{xoption}"],
                frame[f"$.forcefield_dict.v_dict.{yoption}"],
                c="none",
                alpha=0.4,
                edgecolor="k",
                s=50,
                marker="s",
                zorder=2,
            )
        else:
            ax.scatter(
                frame[f"$.forcefield_dict.v_dict.{xoption}"],
                frame[f"$.forcefield_dict.v_dict.{yoption}"],
                c=frame["$.energy_per_bb"],
                alpha=1.0,
                edgecolor="k",
                s=50,
                marker="s",
                zorder=-1,
                vmin=0,
                vmax=0.6,
                cmap="Blues_r",
            )

        frame = frame.sort(pl.col(f"$.forcefield_dict.v_dict.{xoption}")).sort(
            pl.col(f"$.forcefield_dict.v_dict.{yoption}")
        )
        frame = frame.group_by(
            f"$.forcefield_dict.v_dict.{xoption}", maintain_order=True
        ).agg(pl.col("$.energy_per_bb"))

        try:
            zs = np.array(frame["$.energy_per_bb"].to_list()).T
        except ValueError:
            continue
        xs, ys = np.meshgrid(sorted(set(xs)), sorted(set(ys)))

        if as_contour:
            try:
                cs = ax.contourf(
                    xs,
                    ys,
                    zs,
                    levels=[0.0, 0.1, 0.3, 0.6, 1.0],
                    cmap="Blues_r",
                    alpha=0.8,
                    zorder=1,
                )
                if combo == "bac-de":
                    cbar = fig.colorbar(cs)
                    cbar.ax.tick_params(labelsize=16)
                    cbar.ax.set_ylabel(eb_str(), fontsize=16)

            except TypeError:
                continue

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_main_contour_grid(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    as_contour: bool,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(6, 5))

    combo = "bac-dde"
    xoption = "d_d_e"
    yoption = "b_a_c"
    experimental_m3s = (
        (133, 150),
        (133, 155),
        (133, 145),
    )
    experimental_m4s = (
        (133, 165),
        (133, 167),
    )
    experimental_xrds = ((126.9, 166),)
    xlbl = "$dde$  [$^\\circ$]"
    ylbl = "$bac$  [$^\\circ$]"

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xlbl, fontsize=16)
    ax.set_ylabel(ylbl, fontsize=16)

    ax.scatter(
        [i[0] for i in experimental_m3s],
        [i[1] for i in experimental_m3s],
        c=multi_cmap["3"],
        alpha=1.0,
        edgecolor="k",
        s=80,
        marker="X",
        zorder=2,
    )
    ax.scatter(
        [i[0] for i in experimental_m4s],
        [i[1] for i in experimental_m4s],
        c=multi_cmap["4"],
        alpha=1.0,
        edgecolor="k",
        s=80,
        marker="X",
        zorder=2,
    )
    ax.scatter(
        [i[0] for i in experimental_xrds],
        [i[1] for i in experimental_xrds],
        c="cyan",
        alpha=1.0,
        edgecolor="k",
        s=80,
        marker="X",
        zorder=2,
    )

    frame = cgx.utilities.AtomliteDatabase(database_path).get_property_df(
        properties=[
            "$.energy_per_bb",
            f"$.forcefield_dict.v_dict.{xoption}",
            f"$.forcefield_dict.v_dict.{yoption}",
        ]
    )
    frame = frame.filter(pl.col("key").str.contains(combo))

    xs = set(frame[f"$.forcefield_dict.v_dict.{xoption}"])
    ys = set(frame[f"$.forcefield_dict.v_dict.{yoption}"])
    # Plot the underlying grid.
    if as_contour:
        ax.scatter(
            frame[f"$.forcefield_dict.v_dict.{xoption}"],
            frame[f"$.forcefield_dict.v_dict.{yoption}"],
            c="none",
            alpha=0.4,
            edgecolor="k",
            s=50,
            marker="s",
            zorder=2,
        )

    frame = frame.sort(pl.col(f"$.forcefield_dict.v_dict.{xoption}")).sort(
        pl.col(f"$.forcefield_dict.v_dict.{yoption}")
    )
    frame = frame.group_by(
        f"$.forcefield_dict.v_dict.{xoption}", maintain_order=True
    ).agg(pl.col("$.energy_per_bb"))

    zs = np.array(frame["$.energy_per_bb"].to_list()).T
    xs, ys = np.meshgrid(sorted(set(xs)), sorted(set(ys)))

    if as_contour:
        cs = ax.contourf(
            xs,
            ys,
            zs,
            levels=[0.0, 0.1, 0.3, 0.6, 1.0],
            cmap="Blues_r",
            alpha=0.8,
            zorder=1,
        )

        cbar = fig.colorbar(cs)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(eb_str(), fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgenscan_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgenscan_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgenscan_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgenscan_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgenscan"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgenscan.db"

    aa_range = sorted(
        {*list(np.linspace(4.0, 7.0, 5)), 4.9, 4.7, 5.0, 5.3, 5.4}
    )
    bac_range = sorted(
        {
            *list(np.linspace(115.0, 175.0, 10)),
            150.0,
            155.0,
            145.0,
            165.0,
            167.0,
        }
    )

    dde_range = sorted({*list(np.linspace(115.0, 175.0, 10)), 133.0})
    dd_range = sorted({*list(np.linspace(4.0, 10.0, 5)), 8.0})
    de_range = sorted({*list(np.linspace(2.0, 5.0, 5)), 4.3})

    pair = "lf_l2"
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "lf"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "l2"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    combos = {
        "bac-aa": {"yr": bac_range, "xr": aa_range, "yl": "l2", "xl": "l2"},
        "bac-dde": {"yr": bac_range, "xr": dde_range, "yl": "l2", "xl": "lf"},
        "bac-dd": {"yr": bac_range, "xr": dd_range, "yl": "l2", "xl": "lf"},
        "bac-de": {"yr": bac_range, "xr": de_range, "yl": "l2", "xl": "lf"},
        "dd-dde": {"yr": dd_range, "xr": dde_range, "yl": "lf", "xl": "lf"},
        "de-dde": {"yr": de_range, "xr": dde_range, "yl": "lf", "xl": "lf"},
        "de-dd": {"yr": de_range, "xr": dd_range, "yl": "lf", "xl": "lf"},
    }

    if args.run:
        for cname, pair_range_dict in combos.items():
            # Rewrite each time.
            ligand_measures = {
                "lf": {
                    "egb": 120.0,
                    "deg": 180.0,
                    "dd": 8.0,
                    "de": 4.3,
                    "dde": 133.0,
                    "eg": 1.4,
                    "gb": 1.4,
                },
                "l2": {"ba": 2.8, "aa": 4.9, "bac": 150.0, "s": 0.0},
            }

            for (i, xp), (j, yp) in it.product(
                enumerate(pair_range_dict["xr"]),
                enumerate(pair_range_dict["yr"]),
            ):
                ypname, xpname = cname.split("-")
                ligand_measures[pair_range_dict["xl"]][xpname] = xp
                ligand_measures[pair_range_dict["yl"]][ypname] = yp

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    large=converging,
                    small=diverging,
                    large_meas=ligand_measures[converging_name],
                    small_meas=ligand_measures[diverging_name],
                    vdw_bond_cutoff=2,
                )

                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=converging.get_building_block(),
                    name=f"{converging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb = converging_bb.clone()

                tetra_bb = cgx.utilities.optimise_ligand(
                    molecule=tetra.get_building_block(),
                    name=f"{tetra.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                tetra_bb = tetra_bb.clone()

                diverging_bb = cgx.utilities.optimise_ligand(
                    molecule=diverging.get_building_block(),
                    name=f"{diverging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                diverging_bb = diverging_bb.clone()

                name = f"scan_{cname}_{i}-{j}"
                logging.info("building %s", name)

                cage = stk.ConstructedMolecule(
                    cgx.topologies.CGM4L8(
                        building_blocks={
                            tetra_bb: (0, 1, 2, 3),
                            converging_bb: (4, 6, 8, 10),
                            diverging_bb: (5, 7, 9, 11),
                        },
                        vertex_positions=None,
                        scale_multiplier=1.0,
                    )
                )
                cage.write(str(structure_dir / f"{name}_unopt.mol"))

                si, sj = name.split("_")[2].split("-")
                potential_names = []
                for cstr in combos:
                    potential_names.extend(
                        [
                            f"scan_{cstr}_{int(si) - 2}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si)}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si) - 2}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si)}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si) - 2}-{int(sj)}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj)}",
                        ]
                    )

                try:
                    conformer = cgx.scram.optimise_cage(
                        molecule=cage,
                        name=name,
                        output_dir=calculation_dir,
                        forcefield=forcefield,
                        platform=None,
                        database_path=database_path,
                        potential_names=potential_names,
                    )
                    if conformer is not None:
                        conformer.molecule.with_centroid(
                            np.array((0, 0, 0))
                        ).write(str(structure_dir / f"{name}_optc.mol"))

                    analyse_cage(
                        database_path=database_path,
                        name=name,
                        forcefield=forcefield,
                        num_building_blocks=12,
                    )

                except OpenMMException:
                    pass

            # Rescan over the surface for improved energies.
            for (i, xp), (j, yp) in it.product(
                enumerate(pair_range_dict["xr"]),
                enumerate(pair_range_dict["yr"]),
            ):
                ypname, xpname = cname.split("-")
                ligand_measures[pair_range_dict["xl"]][xpname] = xp
                ligand_measures[pair_range_dict["yl"]][ypname] = yp

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    large=converging,
                    small=diverging,
                    large_meas=ligand_measures[converging_name],
                    small_meas=ligand_measures[diverging_name],
                    vdw_bond_cutoff=2,
                )

                name = f"scan_{cname}_{i}-{j}"
                logging.info("rescanning %s", name)

                current_cage = stk.BuildingBlock.init_from_file(
                    structure_dir / f"{name}_optc.mol"
                )

                potential_names = []

                x_indices_of_interest = [
                    pair_range_dict["xr"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - xp) for i in pair_range_dict["xr"]],
                            pair_range_dict["xr"],
                            strict=False,
                        )
                    )
                ][:3]
                y_indices_of_interest = [
                    pair_range_dict["yr"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - yp) for i in pair_range_dict["yr"]],
                            pair_range_dict["yr"],
                            strict=False,
                        )
                    )
                ][:3]

                for cstr, xidx, yidx in it.product(
                    combos, x_indices_of_interest, y_indices_of_interest
                ):
                    potential_names.append(f"scan_{cstr}_{xidx}-{yidx}")

                conformer = cgx.scram.optimise_from_files(
                    molecule=current_cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                    potential_names=potential_names,
                )

                conformer.molecule.with_centroid(np.array((0, 0, 0))).write(
                    str(structure_dir / f"{name}_optc.mol")
                )

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    num_building_blocks=12,
                )

    make_main_contour_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_6.png",
        as_contour=True,
    )

    make_geom_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_2.png",
        combos=combos,
    )
    make_contour_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_3.png",
        combos=combos,
        as_contour=False,
    )
    make_contour_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_1.png",
        combos=combos,
        as_contour=True,
    )
    make_geom_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_4.png",
        combos=combos,
    )
    make_energy_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_5.png",
        combos=combos,
    )


if __name__ == "__main__":
    main()
