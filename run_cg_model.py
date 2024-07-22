"""Script to generate and optimise CG models."""

import logging
import pathlib
import matplotlib.pyplot as plt
from rdkit import RDLogger
import stko
import stk
import argparse
from min_utilities import (
    binder_bead,
    abead_d,
    save_vertex_positions,
    abead_c,
    cbead_d,
    cbead_c,
    ebead_c,
    ebead_d,
    tetra_bead,
    forcefield_lf_ls1,
    forcefield_lf_ls9,
    SixBead,
    eb_str,
    forcefield_la_st52,
    forcefield_la_st5,
    optimise_cage,
)
import cgexplore
from utilities import atomise, get_ligand_bb
from topologies import (
    TopologyIterator,
    TopologyCode,
    vmap_to_str,
    get_underyling_vertices,
)
from openmm import openmm, OpenMMException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def num_pds(pair, multi):
    opts = {
        "lf_ls1": {"1": 1, "2": 2, "3": 3},
        "lf_ls9": {"1": 1, "2": 2, "3": 3},
        "la_st5": {"1": 3, "2": 6, "4": 12},
        "la_st52": {"1": 3, "2": 6, "4": 12},
    }
    return opts[pair][multi]


def optimiser(pair, multi):
    opts = {
        "lf_ls1": {
            "1": stk.MCHammer(target_bond_length=3),
            "2": stk.MCHammer(),
            "3": stk.MCHammer(),
        },
        "lf_ls9": {
            "1": stk.MCHammer(target_bond_length=3),
            "2": stk.MCHammer(),
            "3": stk.MCHammer(),
        },
        "la_st5": {
            "1": stk.MCHammer(),
            "2": stk.MCHammer(),
            "4": stk.MCHammer(),
        },
        "la_st52": {
            "1": stk.MCHammer(),
            "2": stk.MCHammer(),
            "4": stk.MCHammer(),
        },
    }
    return opts[pair][multi]


def analyse_cage(database_path, name, forcefield, iterator, topology_code):
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
            sum(energy_decomp[i] for i in energy_decomp if "total energy" not in i)
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
                openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(openmm.unit.angstrom)

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(openmm.unit.degrees)
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
            v_dict[at.bead_class] = at.sigma.value_in_unit(openmm.unit.angstrom)
            k_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "k_dict": k_dict,
            "v_dict": v_dict,
        }

        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield_dict,
                "strain_energy": fin_energy,
                "energy_per_bb": fin_energy / iterator.get_num_building_blocks(),
                "pair": name.split("_")[0] + "_" + name.split("_")[1],
                "num_components": num_components,
                "multiplier": name.split("_")[2],
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
            },
        )


def make_plot(pair, database_path, structure_dir, figure_dir, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    energies = {}
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
    }

    for entry in cgexplore.utilities.AtomliteDatabase(database_path).get_entries():
        if pair != entry.properties["pair"]:
            continue
        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]

        if multi not in energies:
            energies[multi] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[multi].append((round(energy, 4), entry.key))

    with (figure_dir / f"min_{pair}.txt").open("w") as f:
        for multi in energies:
            if len(energies[multi]) == 0:
                continue

            sorted_energies = sorted(energies[multi], key=lambda p: p[0])
            min_energy = sorted_energies[0]

            ax.plot(
                [i[0] for i in energies[multi]],
                marker="o",
                c=cmap[multi],
                markersize=4,
                # s=40,
                # alpha=0.3,
                # ec="none",
                label=f"{multi}: {round(min_energy[0],3)} @ {min_energy[1]}",
            )

            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.axhline(y=0.3, c="k", ls="--")
    ax.legend(ncols=1, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
    return energies


def make_aa_plot(pair, atomistic_dir, atomistic_calculation_dir, filename):
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    founds = atomistic_calculation_dir.glob(f"c*{pair}*gulp2")

    energies = {}
    for path in sorted(founds):
        if not path.is_dir():
            continue
        output_file = path / "gulp_opt.ginout"
        with output_file.open("r") as f:
            lines = f.readlines()

        name = path.name
        multi = path.name.split("_")[3]
        for line in lines:
            if "Total lattice energy       =" in line and "kJ/mol" in line:
                energy = float(line.strip().split(" ")[-2]) / num_pds(pair, multi)

        if multi not in energies:
            energies[multi] = []

        energies[multi].append((name.replace("_gulp2", ""), energy))

    for multi in sorted(energies):
        if len(energies[multi]) == 0:
            continue

        sorted_energies = sorted(energies[multi], key=lambda p: p[1])
        min_energy = sorted_energies[0]

        ax.plot(
            [i[1] for i in energies[multi]],
            marker="o",
            c=cmap[multi],
            markersize=4,
            # s=40,
            # alpha=0.3,
            # ec="none",
            label=f"{multi}: {round(min_energy[1],3)} @ {min_energy[0]}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("UFF energy / Pd [kJmol-1]", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(500, 1e5)
    ax.legend(ncols=1, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        filename,
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
    parser.add_argument(
        "--atomise",
        action="store_true",
        help="set to build atomistic structures",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    calculation_dir = wd / "min_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "min_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "min_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "min_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    atomistic_dir = wd / "atomistic"
    atomistic_dir.mkdir(exist_ok=True)
    atomistic_calculation_dir = wd / "atomistic_calculations"
    atomistic_calculation_dir.mkdir(exist_ok=True)

    database_path = data_dir / "min_run.db"

    # Define bead libraries.
    present_beads = (cbead_d, abead_d, cbead_c, abead_c, binder_bead, tetra_bead)
    cgexplore.molecular.BeadLibrary(beads=present_beads)

    pairs = {
        "lf_ls1": {
            "forcefield": forcefield_lf_ls1,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 3),
        },
        "lf_ls9": {
            "forcefield": forcefield_lf_ls9,
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": cgexplore.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 3),
        },
        "la_st5": {
            "forcefield": forcefield_la_st5,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "forcefield": forcefield_la_st52,
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": SixBead(bead=cbead_c, abead1=abead_c, abead2=ebead_c),
            "diverging": SixBead(bead=cbead_d, abead1=abead_d, abead2=ebead_d),
            "tetra": cgexplore.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead),
            "multipliers": (1, 2, 4),
        },
    }

    lf_bb = get_ligand_bb(
        path=wd / "ligands" / "lf_prep.mol",
        optl_path=wd / "ligands" / "lf_optl.mol",
    )
    ls1_bb = get_ligand_bb(
        path=wd / "ligands" / "ls1_prep.mol",
        optl_path=wd / "ligands" / "ls1_optl.mol",
    )
    ls9_bb = get_ligand_bb(
        path=wd / "ligands" / "ls9_prep.mol",
        optl_path=wd / "ligands" / "ls9_optl.mol",
    )
    st5_bb = get_ligand_bb(
        path=wd / "ligands" / "st5_prep.mol",
        optl_path=wd / "ligands" / "st5_optl.mol",
    )
    la_bb = get_ligand_bb(
        path=wd / "ligands" / "la_prep.mol",
        optl_path=wd / "ligands" / "la_optl.mol",
    )
    pd_bb = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)),
        position_matrix=[[0, 0, 0]],
    )

    bb_library = {
        "lf_ls1": {
            1: {
                pd_bb: (0,),
                lf_bb: (1,),
                ls1_bb: (2,),
            },
            2: {
                pd_bb: (0, 1),
                lf_bb: (2, 3),
                ls1_bb: (4, 5),
            },
            3: {
                pd_bb: (0, 1, 2),
                lf_bb: (3, 4, 5),
                ls1_bb: (6, 7, 8),
            },
        },
        "lf_ls9": {
            1: {
                pd_bb: (0,),
                lf_bb: (1,),
                ls9_bb: (2,),
            },
            2: {
                pd_bb: (0, 1),
                lf_bb: (2, 3),
                ls9_bb: (4, 5),
            },
            3: {
                pd_bb: (0, 1, 2),
                lf_bb: (3, 4, 5),
                ls9_bb: (6, 7, 8),
            },
        },
        "la_st5": {
            1: {
                pd_bb: (0, 1, 2),
                la_bb: (3, 4, 5, 6),
                st5_bb: (7, 8),
            },
            2: {
                pd_bb: (0, 1, 2, 3, 4, 5),
                la_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                st5_bb: (14, 15, 16, 17),
            },
            4: {
                pd_bb: range(0, 12),
                la_bb: range(12, 28),
                st5_bb: range(28, 36),
            },
        },
        "la_st52": {
            1: {
                pd_bb: (0, 1, 2),
                la_bb: (3, 4, 5, 6),
                st5_bb: (7, 8),
            },
            2: {
                pd_bb: (0, 1, 2, 3, 4, 5),
                la_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                st5_bb: (14, 15, 16, 17),
            },
            4: {
                pd_bb: range(0, 12),
                la_bb: range(12, 28),
                st5_bb: range(28, 36),
            },
        },
    }

    for pair in pairs:
        forcefield = pairs[pair]["forcefield"]
        converging = pairs[pair]["converging"]
        diverging = pairs[pair]["diverging"]
        tetra = pairs[pair]["tetra"]
        # Prepare ligands.
        for i, precursor in enumerate((converging, diverging, tetra)):
            name = f"{precursor.get_name()}_f{forcefield.get_identifier()}"
            building_block = cgexplore.utilities.optimise_ligand(
                molecule=precursor.get_building_block(),
                name=name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            building_block.write(str(ligand_dir / f"{name}_optl.mol"))
            if i == 0:
                converging_bb = building_block.clone()
            elif i == 1:
                diverging_bb = building_block.clone()
            elif i == 2:
                tetra_bb = building_block.clone()

        for multiplier in pairs[pair]["multipliers"]:
            if args.run:
                # Define a connectivity based on a multiplier.
                iterator = TopologyIterator(
                    multiplier=multiplier,
                    stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                    tetra_bb=tetra_bb,
                    converging_bb=converging_bb,
                    diverging_bb=diverging_bb,
                )
                logging.info(f"doing: pair {pair}, multi {multiplier}")
                count = 0
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    name = f"{pair}_{multiplier}_{idx}"
                    acage.write(structure_dir / f"{name}_unopt.mol")

                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )
                    if num_components != 1:
                        continue

                    # Optimise and save.
                    logging.info(f"building {name}")
                    count += 1

                    try:
                        conformer = optimise_cage(
                            molecule=acage,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        if conformer is not None:
                            conformer.molecule.write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        save_vertex_positions(
                            name=name,
                            calculation_dir=calculation_dir,
                            structure_dir=structure_dir,
                            molecule=acage,
                        )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=constructed.topology_code,
                        )

                    except OpenMMException:
                        pass
                logging.info(f"for: pair {pair}, multi {multiplier}, built {count}!")

            energies = make_plot(
                database_path=database_path,
                pair=pair,
                structure_dir=structure_dir,
                figure_dir=figure_dir,
                filename=figure_dir / f"min_1_{pair}.png",
            )

            top_ten_distinct = sorted(set([i[0] for i in energies[str(multiplier)]]))[
                :10
            ]

            if args.atomise:
                for energy in top_ten_distinct:
                    options = energies[str(multiplier)]

                    for o_energy, o_name in options:
                        if o_energy == energy:
                            chosen = o_name
                            break

                    entry = cgexplore.utilities.AtomliteDatabase(
                        database_path
                    ).get_entry(chosen)

                    atomise(
                        vertices=get_underyling_vertices(pair, multiplier),
                        name=chosen,
                        topology_code=TopologyCode(
                            vertex_map=entry.properties["topology_code_vmap"],
                            as_string=vmap_to_str(
                                entry.properties["topology_code_vmap"]
                            ),
                        ),
                        structure_dir=structure_dir,
                        calculation_dir=calculation_dir,
                        atomistic_dir=atomistic_dir,
                        atomistic_calculation_dir=atomistic_calculation_dir,
                        building_blocks=bb_library[pair][multiplier],
                        optimizer=optimiser(pair, str(multiplier)),
                    )
            make_aa_plot(
                pair=pair,
                atomistic_dir=atomistic_dir,
                atomistic_calculation_dir=atomistic_calculation_dir,
                filename=figure_dir / f"min_3_{pair}.png",
            )


if __name__ == "__main__":
    main()
