"""Script to generate and optimise CG models."""

import itertools as it
import logging
import pathlib

import cgexplore
import matplotlib.pyplot as plt
import stk
from openmm import openmm
from rdkit import RDLogger
from utilities import (
    StericSixBead,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    inner_bead,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(  # noqa: C901
    database_path: pathlib.Path,
    name: str,
    forcefield: cgexplore.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgexplore.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "topology_code_vmap" not in properties:
        energy_decomp = {}
        for component in properties["energy_decomposition"]:
            component_tup = properties["energy_decomposition"][component]
            if component == "total energy":
                energy_decomp[f"{component}_{component_tup[1]}"] = float(
                    component_tup[0]
                )
            else:
                just_name = component.split("'")[1]
                key = f"{just_name}_{component_tup[1]}"
                value = float(component_tup[0])
                if key in energy_decomp:
                    energy_decomp[key] += value
                else:
                    energy_decomp[key] = value
        fin_energy = energy_decomp["total energy_kJ/mol"]
        if (
            sum(
                energy_decomp[i]
                for i in energy_decomp
                if "total energy" not in i
            )
            != fin_energy
        ):
            msg = (
                "energy decompisition does not sum to total energy for"
                f" {name}: {energy_decomp}"
            )
            raise RuntimeError(msg)

        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        k_dict = {}
        v_dict = {}

        for bt in ff_targets["bonds"]:
            cp = (bt.type1, bt.type2)
            k_dict["_".join(cp)] = bt.bond_k.value_in_unit(
                openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(
                openmm.unit.angstrom
            )

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(
                    openmm.unit.degrees
                )
            except TypeError:
                # Handle different angle types.
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
                v_dict["_".join(cp)] = (at.n, at.b)

        for at in ff_targets["torsions"]:
            cp = at.search_string
            k_dict["_".join(cp)] = at.torsion_k.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            v_dict["_".join(cp)] = at.phi0.value_in_unit(openmm.unit.degrees)
        for at in ff_targets["nonbondeds"]:
            v_dict[at.bead_class] = at.sigma.value_in_unit(
                openmm.unit.angstrom
            )
            k_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "k_dict": k_dict,
            "v_dict": v_dict,
        }

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy / num_building_blocks,
                "tstr": name.split("_")[1],
                "attempt": name.split("_")[-1],
            },
        )


def make_plot(
    database_path: pathlib.Path,
    nonsteric_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas = {}
    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        tstr = entry.properties["tstr"]
        a = entry.properties["attempt"]
        x = entry.properties["forcefield_dict"]["v_dict"]["s"]
        x2 = entry.properties["forcefield_dict"]["v_dict"]["i_s"]
        y = entry.properties["energy_per_bb"]

        if (tstr, x2) not in datas:
            datas[(tstr, x2)] = []
        datas[(tstr, x2)].append((x, y, a))

    for tstr, x2 in datas:
        nonsteric_data = cgexplore.utilities.AtomliteDatabase(nonsteric_path)

        nonsteric_energy = nonsteric_data.get_entry(key="scan_6-5").properties[
            "energy_per_bb"
        ]

        datas[(tstr, x2)] = [(-0.2, nonsteric_energy)] + datas[(tstr, x2)]

        ax.plot(
            [i[0] for i in datas[(tstr, x2)]],
            [i[1] for i in datas[(tstr, x2)]],
            alpha=1.0,
            marker="o",
            mec="k",
            markersize=10,
            label=f"{tstr}-$ds$:{round(x2, 1)}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [\AA]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.axhline(y=0.3, c="k", ls="--")
    ax.legend(fontsize=16)
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


def main() -> None:  # noqa: PLR0915
    """Run script."""
    raise SystemExit(
        "This is paused for now, because there are some strangeness in the model:"
        " should go back to the Stericsixbead, but set the inner (i) to eps/sigma=0"
        " And then figure out why the 4P82 and 4P8 optimisation do not go well..."
    )
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "tsteric_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "tsteric_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "tsteric_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "tsteric_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    nonsteric_path = wd / "scan_data" / "scan.db"
    database_path = data_dir / "tsteric.db"

    pair = "la_st5"
    converging = StericSixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
        ibead=inner_bead,
        sbead=steric_bead,
    )
    converging_name = "la"
    diverging = cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    topologies = (
        ("3P6", stk.cage.M3L6, (2, 1)),
        # ("4P8", scram.topologies.CGM4L8, (1, 1)),
        # ("4P82", scram.topologies.M4L82, (1, 1)),
    )

    is_range = [0.5]  # , 1.0, 2.0, 3.0, 4.0]
    s_range = [0.0, 0.1]  # 0.25, 0.5, 0.75]

    new_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbg": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "egb": ("angle", 120, 1e2),
        "deg": ("angle", 180, 1e2),
        # Torsions.
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "g": ("nb", 10.0, 1.0),
        "i": ("nb", 10.0, 1.0),
    }

    logging.info("building %s structures", len(s_range) * len(topologies))
    for (i, s_value), (j, is_value) in it.product(
        enumerate(s_range), enumerate(is_range)
    ):
        ligand_measures = {
            "la": {
                "dd": 7.0,
                "de": 1.5,
                "ide": 170,
                "eg": 1.4,
                "gb": 1.4,
                "s": s_value,
                "se": 0,  # 10.0 if s_value > 0.0 else 0.0,
                "is": is_value,
            },
            "st5": {"ba": 2.8, "aa": 5.0, "bac": 120, "bacab": 180},
        }

        forcefield = precursors_to_forcefield(
            pair=pair,
            diverging=diverging,
            converging=converging,
            conv_meas=ligand_measures[converging_name],
            dive_meas=ligand_measures[diverging_name],
            new_definer_dict=new_definer_dict,
        )

        converging_bb = scram.toy.prepare_building_block(
            precursor=converging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        diverging_bb = scram.toy.prepare_building_block(
            precursor=diverging,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )
        tetra_bb = scram.toy.prepare_building_block(
            precursor=tetra,
            forcefield=forcefield,
            calculation_dir=calculation_dir,
            ligand_dir=ligand_dir,
        )

        for tstr, tfunction, _ in topologies:
            for attempt in range(6):
                name = f"ts_{tstr}_{i}_{j}_{attempt}"
                logging.info("building %s", name)
                if tstr == "3P6":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2),
                                converging_bb: (3, 4, 5, 6),
                                diverging_bb: (7, 8),
                            },
                            vertex_positions=None,
                            scale_multiplier=1.0,
                        )
                    )
                    num_bbs = 9

                if tstr == "4P8":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2, 3),
                                converging_bb: (4, 6, 8, 10),
                                diverging_bb: (5, 7, 9, 11),
                            },
                            vertex_positions=None,
                            scale_multiplier=0.5,
                        )
                    )
                    num_bbs = 12
                    cage.write(structure_dir / f"{name}_unopt.mol")
                    raise SystemExit

                if tstr == "4P82":
                    cage = stk.ConstructedMolecule(
                        tfunction(
                            building_blocks={
                                tetra_bb: (0, 1, 2, 3),
                                converging_bb: (5, 6, 7, 8),
                                diverging_bb: (4, 9, 10, 11),
                            },
                            vertex_positions=None,
                            scale_multiplier=0.5,
                        )
                    )
                    num_bbs = 12
                    cage.write(structure_dir / f"{name}_unopt.mol")
                    raise SystemExit

                cage.write(structure_dir / f"{name}_unopt.mol")

                conformer = scram.toy.optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                )
                if conformer is not None:
                    conformer.molecule.with_centroid((0, 0, 0)).write(
                        str(structure_dir / f"{name}_optc.mol")
                    )

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    num_building_blocks=num_bbs,
                )

    make_plot(
        database_path=database_path,
        nonsteric_path=nonsteric_path,
        figure_dir=figure_dir,
        filename="sterics_1.png",
    )

    raise SystemExit("test what happens if all sigmas are reduced.")
    raise SystemExit("try and find which angle is causing the issue.")
    raise SystemExit("there is something wrong with this force.")


if __name__ == "__main__":
    main()
