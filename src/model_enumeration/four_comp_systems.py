"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
import shutil

import cgexplore as cgx
import matplotlib.pyplot as plt
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    a2bead_d,
    abead_c,
    abead_d,
    analyse_cage,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    e2bead_d,
    ebead_c,
    get_regraphed_molecule,
    get_vertexset_molecule,
    passes_graph_bb_iso,
    steric_bead,
    tetra_bead,
)
from model_enumeration.utilities import (
    eb_str,
    isomer_energy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
attempts = (
    None,
    "regraphed-spring-10",
    "regraphed-kamada-10",
    "set-kamada-10",
    "set-spring-10",
    "set-spectral-10",
)


def study_5_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(5, 5))

    tmap = {
        "mix1": "tab:blue",
        "mix2": "tab:orange",
        "mix3": "tab:green",
        "mix4": "tab:red",
        "mix5": "tab:purple",
    }
    targets = {
        i: {"0": float("inf"), "1": float("inf"), "2": float("inf")}
        for i in tmap
    }

    possible_pos = [-0.3, -0.15, 0, 0.15, 0.3]

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        targets[entry.properties["mix"]][str(entry.properties["sidx"])] = min(
            (
                entry.properties["energy_per_bb"],
                targets[entry.properties["mix"]][
                    str(entry.properties["sidx"])
                ],
            )
        )
        ax.scatter(
            entry.properties["sidx"]
            + possible_pos[int(entry.properties["mix"][-1]) - 1],
            entry.properties["energy_per_bb"],
            c=tmap[entry.properties["mix"]],
            alpha=0.5,
            ec="none",
            s=60,
            marker="o",
        )
        logging.info(
            "E for %s, %s is %s (%s)",
            entry.properties["mix"],
            entry.properties["sidx"],
            round(entry.properties["energy_per_bb"], 2),
            entry.key,
        )

    for xi, (pair, edict) in enumerate(targets.items()):
        ax.bar(
            [
                int(i) + possible_pos[xi]
                for i, ed in edict.items()
                if ed != float("inf")
            ],
            [ed for ed in edict.values() if ed != float("inf")],
            alpha=1.0,
            width=0.1,
            fc=tmap[pair],
            ec="k",
            label=pair,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_yscale("log")

    ax.axhspan(ymin=0, ymax=isomer_energy(), facecolor="k", alpha=0.05)
    ax.legend(fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2:2:1:1", "2:2:0:2", "2:2:2:0"], fontsize=16)

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


def study_5_plot2(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    tmap = {
        "mix1": "tab:blue",
        "mix2": "tab:orange",
        "mix3": "tab:green",
        "mix4": "tab:red",
        "mix5": "tab:purple",
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if (
            entry.properties["sidx"] == 0
            and entry.properties["topology_idx"] == 0
            and entry.properties["bb_config_idx"] == 2  # noqa: PLR2004
        ):
            x = int(entry.properties["mix"][-1]) - 1
            y = entry.properties["energy_per_bb"]

            ax.bar(
                x,
                y,
                alpha=1.0,
                width=0.8,
                fc=tmap[entry.properties["mix"]],
                ec="k",
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"{eb_str()}", fontsize=16)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(tmap)))
    ax.set_xticklabels(list(tmap), fontsize=16)

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


def study_5_plot3(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))

    tmap = {
        "mix1": "tab:blue",
        "mix2": "tab:orange",
        "mix3": "tab:green",
        "mix4": "tab:red",
        "mix5": "tab:purple",
    }

    rxns = {
        "mix1": {"het_eb": 0, "homo_eb": 0},
        "mix2": {"het_eb": 0, "homo_eb": 0},
        "mix3": {"het_eb": 0, "homo_eb": 0},
        "mix4": {"het_eb": 0, "homo_eb": 0},
        "mix5": {"het_eb": 0, "homo_eb": 0},
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        if (
            entry.properties["sidx"] == 0
            and entry.properties["topology_idx"] == 0
            and entry.properties["bb_config_idx"] == 2  # noqa: PLR2004
        ):
            rxns[entry.properties["mix"]]["het_eb"] += entry.properties[
                "energy_per_bb"
            ]

        if (
            entry.properties["sidx"] == 1
            and entry.properties["topology_idx"] == 0
            and entry.properties["bb_config_idx"] == 5  # noqa: PLR2004
        ):
            rxns[entry.properties["mix"]]["homo_eb"] += entry.properties[
                "energy_per_bb"
            ]

        if (
            entry.properties["sidx"] == 2  # noqa: PLR2004
            and entry.properties["topology_idx"] == 0
            and entry.properties["bb_config_idx"] == 0
        ):
            rxns[entry.properties["mix"]]["homo_eb"] += entry.properties[
                "energy_per_bb"
            ]

    for mix in rxns:
        x = int(mix[-1]) - 1
        y = (rxns[mix]["homo_eb"] - rxns[mix]["het_eb"] * 2) / 2
        ax.bar(
            x,
            y,
            alpha=1.0,
            width=0.8,
            fc=tmap[mix],
            ec="k",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"$E_{\mathrm{exchange}}$ / $n$", fontsize=16)
    ax.set_xticks(range(len(tmap)))
    ax.set_xticklabels(list(tmap), fontsize=16)
    ax.axhline(y=0, c="k")

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def case_study_3(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run case study 3 studying Pd(II) heteroleptic systems."""
    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")
    calculation_dir = wd / "mgencs3_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs3_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs3_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs3_data"
    data_dir.mkdir(exist_ok=True)
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgen_cs3"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs3.db"
    raise SystemExit
    stoichiometries = ((2, 2, 1, 1), (2, 2, 0, 2), (2, 2, 2, 0))
    vdw_cutoff = 2
    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        c2bead_d,
        a2bead_d,
        e2bead_d,
        binder_bead,
        tetra_bead,
        steric_bead,
    )

    ligand_measures = {
        "lin": {"ba": 2.8, "aa": 1.5, "bac": 180},
        "lin2": {"ba": 2.8, "aa": 1.5, "bac": 175},
        "mxy": {"be": 7.6, "ee": 5.0, "bed": 90},
        "pxy": {"be": 7.7, "ee": 5.8, "bed": 110},
        "fxy": {"be": 7.7, "ee": 5.8, "bed": 120},
        "tetra": {"mb": 2.0, "bmb": 90},
        "sqp": {"bf": 2.0, "bfb": 90, "fbm": 90},
    }

    mixtures = {
        "mix1": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "mxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
        },
        "mix2": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "pxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
        },
        "mix3": {
            "linear": (
                "lin",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "fxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
        },
        "mix4": {
            "linear": (
                "lin2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "mxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
        },
        "mix5": {
            "linear": (
                "lin2",
                cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            ),
            "tetra": (
                "tetra",
                cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            ),
            "corner": ("sqp", cgx.molecular.TwoC0Arm(bead=e2bead_d)),
            "bent": (
                "pxy",
                cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c),
            ),
        },
    }

    cg_scale = 2
    if run:
        for mix, mdict in mixtures.items():
            cgx.molecular.BeadLibrary(present_beads)
            bent_name, bent = mdict["bent"]
            linear_name, linear = mdict["linear"]
            corner_name, corner = mdict["corner"]
            tetra_name, tetra = mdict["tetra"]
            cs5_definer_dict = {
                # Bent.
                "be": (
                    "bond",
                    ligand_measures[bent_name]["be"] / cg_scale,
                    1e5,
                ),
                "ed": (
                    "bond",
                    ligand_measures[bent_name]["ee"] / 2 / cg_scale,
                    1e5,
                ),
                "bed": (
                    "angle",
                    ligand_measures[bent_name]["bed"],
                    1e2,
                ),
                "ede": ("angle", 180, 1e2),
                # Corner.
                "mb": (
                    "bond",
                    ligand_measures[tetra_name]["mb"] / cg_scale,
                    1e5,
                ),
                "bf": (
                    "bond",
                    ligand_measures[corner_name]["bf"] / cg_scale,
                    1e5,
                ),
                "bmb": ("pyramid", ligand_measures[tetra_name]["bmb"], 1e2),
                "bfb": ("angle", ligand_measures[corner_name]["bfb"], 1e2),
                "fbm": ("angle", ligand_measures[corner_name]["fbm"], 1e2),
                # Linear.
                "ba": (
                    "bond",
                    ligand_measures[linear_name]["ba"] / cg_scale,
                    1e5,
                ),
                "ac": (
                    "bond",
                    ligand_measures[linear_name]["aa"] / 2 / cg_scale,
                    1e5,
                ),
                "bac": (
                    "angle",
                    ligand_measures[linear_name]["bac"],
                    1e2,
                ),
                "aca": ("angle", 180, 1e2),
                # Bonds.
                # Constant.
                "mba": ("angle", 180, 1e2),
                "mbe": ("angle", 180, 1e2),
                # Nonbondeds.
                "m": ("nb", 10.0, 1.0),
                "d": ("nb", 10.0, 1.0),
                "e": ("nb", 10.0, 1.0),
                "a": ("nb", 10.0, 1.0),
                "b": ("nb", 10.0, 1.0),
                "c": ("nb", 10.0, 1.0),
                "f": ("nb", 10.0, 1.0),
            }

            forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=f"{mix}ff",
                prefix=f"{mix}ff",
                vdw_bond_cutoff=vdw_cutoff,
                present_beads=present_beads,
                definer_dict=cs5_definer_dict,
            )

            bent_bb = cgx.utilities.optimise_ligand(
                molecule=bent.get_building_block(),
                name=f"{mix}_{bent.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            bent_bb.write(
                str(ligand_dir / f"{mix}_{bent.get_name()}_optl.mol")
            )

            linear_bb = cgx.utilities.optimise_ligand(
                molecule=linear.get_building_block(),
                name=f"{mix}_{linear.get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            linear_bb.write(
                str(ligand_dir / f"{mix}_{linear.get_name()}_optl.mol")
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra.get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(
                str(ligand_dir / f"{mix}_{tetra.get_name()}_optl.mol")
            )

            corner_bb = corner.get_building_block()
            corner_bb.write(
                str(ligand_dir / f"{mix}_{corner.get_name()}_optl.mol")
            )

            for sidx, stoichiometry in enumerate(stoichiometries):
                logging.info("doing: %s, stoich %s", mix, stoichiometry)
                gtype = f"{stoichiometry[0]}P{stoichiometry[0] * 2}"

                if sum(stoichiometry[1:]) != stoichiometry[0] * 2:
                    raise RuntimeError

                if stoichiometry[2] == 0:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        corner_bb: stoichiometry[1],
                        bent_bb: stoichiometry[3],
                    }
                elif stoichiometry[3] == 0:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        linear_bb: stoichiometry[2],
                        corner_bb: stoichiometry[1],
                    }
                else:
                    b_counts = {
                        tetra_bb: stoichiometry[0],
                        linear_bb: stoichiometry[2],
                        corner_bb: stoichiometry[1],
                        bent_bb: stoichiometry[3],
                    }

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts=b_counts,
                    graph_type=gtype,
                    graph_set="rx",
                )
                logging.info(
                    "graph iteration has %s graphs", iterator.count_graphs()
                )

                possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                    iterator=iterator
                )
                logging.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )

                run_topology_codes = []
                for bb_config, (idx, topology_code) in it.product(
                    possible_bbdicts,
                    enumerate(iterator.yield_graphs()),
                ):
                    # Testing bb-config aware graph check.
                    if not passes_graph_bb_iso(
                        topology_code=topology_code,
                        bb_config=bb_config,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append((topology_code, bb_config))

                    generated_conformers = []
                    for midx, scale in enumerate(attempts):
                        name = f"{mix}_s{sidx}_{idx}_{midx}_b{bb_config.idx}"

                        try:
                            if isinstance(scale, str) and "regraphed" in scale:
                                constructed_molecule = get_regraphed_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=bb_config,
                                )

                            else:
                                constructed_molecule = get_vertexset_molecule(
                                    scale=scale,
                                    topology_code=topology_code,
                                    iterator=iterator,
                                    bb_config=bb_config,
                                )

                        except (ValueError, KeyError):
                            continue

                        constructed_molecule.write(
                            structure_dir / f"{name}_unopt.mol"
                        )
                        # Optimise and save.
                        logging.info("building %s", name)

                        try:
                            potential_names = [
                                f"{mix}_s{sidx}_{idx}_"
                                f"{nmash_idx}_b{bb_config.idx}"
                                for nmash_idx in range(len(attempts))
                            ]
                            if scale is None:
                                conformer = cgx.scram.graph_optimise_cage(
                                    molecule=constructed_molecule,
                                    name=name,
                                    output_dir=calculation_dir,
                                    forcefield=forcefield,
                                    platform=None,
                                    database_path=database_path,
                                )
                                # Copy the file over.
                                wipfinal = (
                                    calculation_dir / f"{name}_wipfinal.mol"
                                )
                                wipfinal_new = (
                                    calculation_dir / f"{name}_final.mol"
                                )
                                if wipfinal.exists():
                                    shutil.copy(wipfinal, wipfinal_new)

                            else:
                                conformer = cgx.scram.optimise_cage(
                                    molecule=constructed_molecule,
                                    name=name,
                                    output_dir=calculation_dir,
                                    forcefield=forcefield,
                                    platform=None,
                                    database_path=database_path,
                                    potential_names=potential_names,
                                )

                            if conformer is not None:
                                num_components = len(
                                    stko.Network.init_from_molecule(
                                        conformer.molecule
                                    ).get_connected_components()
                                )
                                energy_per_bb = (
                                    cgx.utilities.get_energy_per_bb(
                                        energy_decomposition=(
                                            conformer.energy_decomposition
                                        ),
                                        number_building_blocks=(
                                            iterator.get_num_building_blocks()
                                        ),
                                    )
                                )

                                properties = {
                                    "forcefield_dict": (
                                        forcefield.get_forcefield_dictionary()
                                    ),
                                    "energy_per_bb": energy_per_bb,
                                    "linear": linear_name,
                                    "tetra": tetra_name,
                                    "corner": corner_name,
                                    "bent": bent_name,
                                    "mix": mix,
                                    "num_components": num_components,
                                    "num_bbs": (
                                        iterator.get_num_building_blocks()
                                    ),
                                    "sidx": sidx,
                                    "stoichiometry": stoichiometry,
                                    "topology_idx": idx,
                                    "mash_idx": midx,
                                    "topology_code_vmap": tuple(
                                        (int(i[0]), int(i[1]))
                                        for i in topology_code.vertex_map
                                    ),
                                    "bb_config_idx": bb_config.idx,
                                }

                                cgx.utilities.AtomliteDatabase(
                                    database_path
                                ).add_properties(
                                    key=name,
                                    property_dict=properties,
                                )
                                generated_conformers.append(
                                    (
                                        name,
                                        conformer.molecule.with_centroid(
                                            (0, 0, 0)
                                        ),
                                        energy_per_bb,
                                    )
                                )

                        except OpenMMException:
                            logging.info("failed optimisation of %s", name)

                    if len(generated_conformers) == 0:
                        continue
                    min_energy_conformer = sorted(
                        generated_conformers, key=lambda p: p[2]
                    )[0]
                    min_energy_name, min_energy_structure, _ = (
                        min_energy_conformer
                    )

                    min_energy_structure.write(
                        str(structure_dir / f"{min_energy_name}_optc.mol")
                    )

                    analyse_cage(
                        database_path=database_path,
                        name=min_energy_name,
                        forcefield=forcefield,
                    )

    study_5_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )
    study_5_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
    )
    study_5_plot3(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
    )


def main() -> None:
    """Run script."""
    args = _parse_args()
    case_study_3(args.run)


if __name__ == "__main__":
    main()
