"""Script to generate and optimise CG models."""

import logging
import pathlib

import cgexplore
import matplotlib.pyplot as plt
import stk
from openmm import OpenMMException, openmm
from rdkit import RDLogger
from utilities import (
    SixBead,
    StericSixBead,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
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
) -> None:
    """Analyse a toy model cage."""
    database = cgexplore.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties

    pair1, pair2, tstr, bbstr = name.split("_")
    pair = f"{pair1}_{pair2}"
    if "topology" not in properties:
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
                "topology": tstr,
                "bbstr": bbstr,
                "pair": pair,
                "energy_per_bb": fin_energy
                / scram.topologies.stoich_map(tstr),
            },
        )


def make_plot(
    pair: str,
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))
    flat_axs = axs.flatten()
    energies = {}

    pairs = ("la_st50", "la_st5", "la_st52", "la2_st5", "la2_st52")
    axmap = dict(zip(pairs, flat_axs, strict=False))

    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        energy = entry.properties["energy_per_bb"]
        bbstr = entry.properties["bbstr"]
        tstr = entry.properties["topology"]
        pair = entry.properties["pair"]

        if (pair, tstr) not in energies:
            energies[(pair, tstr)] = []

        energies[(pair, tstr)].append((round(energy, 4), bbstr))

    for pair, tstr in energies:
        sorted_energies = sorted(energies[(pair, tstr)], key=lambda p: p[0])
        min_energy = sorted_energies[0]
        ax = axmap[pair]
        ax.plot(
            [i[0] for i in sorted_energies],
            marker="o",
            markersize=4,
            label=(f"{tstr}: {round(min_energy[0],3)}" f" @ {min_energy[1]}"),
        )
        ax.set_title(f"{pair}", fontsize=16)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 50)
        ax.set_xticks([])
        ax.axhline(y=0.3, c="k", ls="--")
        ax.legend(ncols=1, fontsize=16)
        if pair in ("la_st50", "la2_st5"):
            ax.set_ylabel(eb_str(), fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def make_parity_plot(
    pair: str,
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    energies = {}

    for entry in cgexplore.utilities.AtomliteDatabase(
        database_path
    ).get_entries():
        energy = entry.properties["energy_per_bb"]
        bbstr = entry.properties["bbstr"]
        tstr = entry.properties["topology"]
        pair = entry.properties["pair"]

        if (pair, tstr) not in energies:
            energies[(pair, tstr)] = []

        energies[(pair, tstr)].append((round(energy, 4), bbstr))

    for ax, tstr in zip(axs, ("3P6", "4P8", "4P82"), strict=False):
        opt1 = energies[("la_st50", tstr)]
        for p2, string in zip(
            ("la2_st5", "la2_st52", "la_st5", "la_st52"),
            (
                r"$c$|$\sigma=1$",
                r"$o$|$\sigma=1$",
                r"$c$|$\sigma=2$",
                r"$o$|$\sigma=2$",
            ),
            strict=False,
        ):
            opt2 = energies[(p2, tstr)]
            ax.scatter(
                [i[0] for i in opt1],
                [i[0] for i in opt2],
                ec="k",
                s=40,
                label=f"$vs$ {string}",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{eb_str()}", fontsize=16)
        ax.set_ylabel(f"+sterics {eb_str()}", fontsize=16)
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0.01, 100)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot((0, 100), (0, 100), c="k")
        ax.legend(fontsize=16)
        ax.set_title(tstr)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
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

    database_path = data_dir / "tsteric.db"

    ligand_measures = {
        "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4, "s": 2},
        "la2": {
            "dd": 7.0,
            "de": 1.5,
            "dde": 170,
            "eg": 1.4,
            "gb": 1.4,
            "s": 1,
        },
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
    }

    pairs = {
        "la_st50": {
            "converging_name": "la",
            "diverging_name": "st5",
            "converging": SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
        },
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
        },
        "la2_st5": {
            "converging_name": "la2",
            "diverging_name": "st5",
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
        },
        "la2_st52": {
            "converging_name": "la2",
            "diverging_name": "st52",
            "converging": StericSixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
                sbead=steric_bead,
            ),
            "diverging": cgexplore.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgexplore.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
        },
    }

    topologies = (
        ("3P6", stk.cage.M3L6, (2, 1)),
        ("4P8", scram.topologies.CGM4L8, (1, 1)),
        ("4P82", scram.topologies.M4L82, (1, 1)),
    )

    for pair in pairs:
        break
        converging_name = pairs[pair]["converging_name"]
        diverging_name = pairs[pair]["diverging_name"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]

        forcefield = precursors_to_forcefield(
            pair=pair,
            diverging=diverging,
            converging=converging,
            conv_meas=ligand_measures[converging_name],
            dive_meas=ligand_measures[diverging_name],
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

        for tstr, tfunction, ratio in topologies:
            possible_bbdicts = scram.topologies.get_potential_bb_dicts(
                tstr=tstr,
                ratio=ratio,
                bb_type="ditopic",
            )
            logging.info(
                "there are %s possible BB dicts for %s, %s",
                len(possible_bbdicts),
                tstr,
                pair,
            )

            for bbdict in possible_bbdicts:
                bbs = {
                    bb: tuple(bbdict[1][idx])
                    for idx, bb in enumerate(
                        (tetra_bb, converging_bb, diverging_bb)
                    )
                }

                name = f"{pair}_{tstr}_b{bbdict[0]}"

                acage = stk.ConstructedMolecule(tfunction(building_blocks=bbs))
                acage.write(structure_dir / f"{name}_unopt.mol")
                # Optimise and save.
                logging.info("building %s", name)

                try:
                    conformer = scram.toy.optimise_cage(
                        molecule=acage,
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
                    )

                except OpenMMException:
                    pass

    make_plot(
        database_path=database_path,
        pair=pair,
        figure_dir=figure_dir,
        filename="sterics_2.png",
    )

    make_parity_plot(
        database_path=database_path,
        pair=pair,
        figure_dir=figure_dir,
        filename="sterics_3.png",
    )


if __name__ == "__main__":
    main()
