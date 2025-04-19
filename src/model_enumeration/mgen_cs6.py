"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
from collections import abc, defaultdict

import bbprep
import cgexplore as cgx
import matplotlib.pyplot as plt
import mchammer as mch
import openmm
import stk
import stko
from openff.toolkit import ForceField
from rdkit import RDLogger

from model_enumeration.mgen_utilities import (
    get_regraphed_molecule,
    get_vertexset_molecule,
)
from model_enumeration.utilities import (
    contains_parallels,
    eb_str,
    multi_cmap,
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
    "set-kamada-20",
    "set-spring-20",
    "set-spectral-20",
)


def get_mch_bonds(molecule: stk.Molecule) -> abc.Generator[mch.Bond]:
    """Yield the bonds of the :mod:`MCHammer` molecule."""
    for i, bond_infos in enumerate(molecule.get_bond_infos()):
        ba1 = bond_infos.get_bond().get_atom1().get_id()
        ba2 = bond_infos.get_bond().get_atom2().get_id()
        # Must ensure bond atom id ordering is the same here as in
        # line 30. Therefore, sort here.
        ba1, ba2 = sorted((ba1, ba2))
        yield mch.Bond(id=i, atom_ids=(ba1, ba2))


def get_subunits(molecule: stk.Molecule) -> dict[int | None, list[int]]:
    """Get connected graphs based on building block ids."""
    subunits = defaultdict(list)
    for atom_info in molecule.get_atom_infos():
        subunits[atom_info.get_building_block_id()].append(
            atom_info.get_atom().get_id()
        )

    return subunits


def get_long_bond_ids(
    molecule: stk.Molecule,
) -> abc.Generator[tuple[int, int], None, None]:
    """Yield the ids of the long bonds to optimize."""
    for bond_infos in molecule.get_bond_infos():
        ba1 = bond_infos.get_bond().get_atom1().get_id()
        ba2 = bond_infos.get_bond().get_atom2().get_id()
        # None if for constructed bonds.
        if bond_infos.get_building_block() is None:
            yield sorted((ba1, ba2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def study_6_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (ax) = plt.subplots(ncols=1, figsize=(8, 5))

    multis = {
        1: (multi_cmap["1"], -0.2),
        2: (multi_cmap["2"], 0.0),
        3: (multi_cmap["3"], 0.2),
        4: (multi_cmap["4"], 0.2),
    }

    xs = {i: j for j, i in enumerate(multis)}
    lbls = set()
    min_energy = float("inf")
    min_energy_key = None
    xys = {}
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue

        ec = "k"
        x = xs[entry.properties["multiplier"]]
        multi = entry.properties["multiplier"]
        y = entry.properties["energy_per_bb"]
        if y < min_energy:
            min_energy = y
            min_energy_key = entry.key

        xys[entry.key] = (x, y, multi, ec)

    for key, (x, y, multi, ec) in xys.items():
        logging.info(
            "E for %s is %s",
            key,
            round(y - min_energy, 2),
        )
        lbl = f"$m$ = {multi}"
        zorder = -1 if ec == "none" else 2
        ax.scatter(
            x,
            y - min_energy,
            c=multis[multi][0],
            alpha=1,
            ec=ec,
            s=120,
            label=lbl if lbl not in lbls else None,
            zorder=zorder,
        )
        lbls.add(lbl)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(f"minimum energy is {min_energy_key}", fontsize=16)
    ax.set_ylabel(f"rel. GFN2-xTB {eb_str()}", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_xticks([xs[i] for i in xs])
    ax.set_xticklabels(list(xs), fontsize=16)
    ax.set_ylim(0, None)

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


def atomistic_optimisation(  # noqa: D103
    name: str,
    molecule: stk.Molecule,
    output_directory: pathlib.Path,
) -> stk.Molecule:
    step1_ = output_directory / f"{name}_step1.mol"
    if not step1_.exists():
        logging.info("step 1 for %s", name)
        # Implement MCHammer.
        optimizer = mch.Optimizer(
            step_size=0.25,
            target_bond_length=2,
            num_steps=500,
            bond_epsilon=50,
            nonbond_epsilon=20,
            nonbond_sigma=1.2,
            nonbond_mu=3,
            beta=2,
            random_seed=1000,
        )
        mch_mol = mch.Molecule(
            atoms=(
                mch.Atom(
                    id=atom.get_id(),
                    element_string=atom.__class__.__name__,
                )
                for atom in molecule.get_atoms()
            ),
            bonds=tuple(get_mch_bonds(molecule)),
            position_matrix=molecule.get_position_matrix(),
        )
        mch_mol, _ = optimizer.get_result(
            mol=mch_mol,
            bond_pair_ids=tuple(get_long_bond_ids(molecule)),
            subunits=get_subunits(molecule),
        )
        molecule = molecule.with_position_matrix(mch_mol.get_position_matrix())
        molecule.write(step1_)
    molecule = molecule.with_structure_from_file(step1_)

    step2_ = output_directory / f"{name}_step2.mol"
    if not step2_.exists():
        logging.info("step 2 for %s", name)
        molecule = stko.MMFF().optimize(molecule)
        molecule.write(step2_)
    molecule = molecule.with_structure_from_file(step2_)

    step3_ = output_directory / f"{name}_step3.mol"
    if not step3_.exists():
        logging.info("step 3 for %s", name)
        molecule = stko.MMFF().optimize(molecule)
        molecule.write(step3_)
    molecule = molecule.with_structure_from_file(step3_)

    # Settings.
    force_field = ForceField("openff_unconstrained-2.2.1.offxml")
    partial_charges = "espaloma-am1bcc"
    step4_ = output_directory / f"{name}_step4.mol"
    if not step4_.exists():
        logging.info("step 4 for %s", name)

        # Define sequence.
        optimisation_sequence = stko.OptimizerSequence(
            # Restricted true to optimised the constructed bonds.
            stko.OpenMMForceField(
                force_field=force_field,
                restricted=True,
                partial_charges_method=partial_charges,
            ),
            # Unrestricted optimisation.
            stko.OpenMMForceField(
                # Load the openff-2.2.1 force field appropriate for
                # vacuum calculations (without constraints)
                force_field=force_field,
                restricted=False,
                partial_charges_method=partial_charges,
            ),
        )

        molecule = optimisation_sequence.optimize(molecule)
        molecule.write(step4_)
    molecule = molecule.with_structure_from_file(step4_)

    step5_ = output_directory / f"{name}_step5.mol"
    if not step5_.exists():
        logging.info("step 5 for %s", name)

        def integrator(
            *,
            temperature: float,
            friction: float,
            time_step: float,
        ) -> openmm.LangevinIntegrator:
            """Define integrator."""
            integrator = openmm.LangevinIntegrator(
                temperature, friction, time_step
            )
            integrator.setRandomNumberSeed(127)
            return integrator

        # Settings.
        temperature = 300 * openmm.unit.kelvin
        friction = 10 / openmm.unit.picoseconds
        time_step = 0.5 * openmm.unit.femtoseconds
        # Define sequence.
        optimisation_sequence = stko.OptimizerSequence(
            # Molecular dynamics, short for equilibration.
            stko.OpenMMMD(
                force_field=force_field,
                output_directory=output_directory / f"{name}_md_equilibration",
                integrator=integrator(
                    temperature=temperature,
                    friction=friction,
                    time_step=time_step,
                ),
                random_seed=275,
                partial_charges_method=partial_charges,
                # Frequency here is not related to the num confs tested.
                reporting_freq=100,
                trajectory_freq=100,
                # 10 ps
                num_steps=10_000,
                num_conformers=10,
                platform="CUDA",
                conformer_optimiser=stko.OpenMMForceField(
                    force_field=force_field,
                    restricted=False,
                    partial_charges_method=partial_charges,
                ),
            ),
            # Long MD, for collecting lowest energy conformers.
            stko.OpenMMMD(
                force_field=force_field,
                output_directory=output_directory / f"{name}_md_production",
                integrator=integrator(
                    temperature=temperature,
                    friction=friction,
                    time_step=time_step,
                ),
                random_seed=275,
                partial_charges_method=partial_charges,
                # Frequency here is not related to the num confs tested.
                reporting_freq=100,
                trajectory_freq=100,
                # 0.2 ns
                num_steps=200_000,
                # 1 every 4 ps
                num_conformers=50,
                platform="CUDA",
                conformer_optimiser=stko.OpenMMForceField(
                    force_field=force_field,
                    restricted=False,
                    partial_charges_method=partial_charges,
                ),
            ),
        )

        molecule = optimisation_sequence.optimize(molecule).with_centroid(
            (0, 0, 0)
        )
        molecule.write(step5_)

    molecule.with_structure_from_file(step5_).with_centroid((0, 0, 0)).write(
        step5_
    )
    return molecule.with_structure_from_file(step5_)


def case_study_6(run: bool) -> None:  # noqa: C901, PLR0915
    """Run case study 6."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")
    calculation_dir = wd / "mgencs6_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "mgencs6_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "mgencs6_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "mgencs6_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "mgencs6"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "mgencs6.db"

    stoichiometry_t_d = (2, 3)
    multipliers = (1, 2, 3, 4)

    ditopic_building_block = stk.BuildingBlock(
        smiles="C1CC(CC(C1)N)N",
        functional_groups=[stk.PrimaryAminoFactory()],
    )
    tritopic_building_block = stk.BuildingBlock(
        smiles="C1=C(C=C(C=C1C=O)C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )
    ditopic_building_block.write(ligand_dir / "di_unopt.mol")
    tritopic_building_block.write(ligand_dir / "tri_unopt.mol")
    # Get lowest energy conformer.
    ensemble = bbprep.generators.ETKDG(num_confs=100).generate_conformers(
        ditopic_building_block
    )
    # Iterate over ensemble.
    minimum_score = 1e24
    minimum_conformer = bbprep.Conformer(
        molecule=ensemble.get_base_molecule().clone(),
        conformer_id=-1,
        source=None,
        permutation=None,
    )
    for conformer in ensemble.yield_conformers():
        # Something here to calculate energy.
        score = stko.MMFFEnergy().get_energy(conformer.molecule)
        if score < minimum_score:
            minimum_score = score
            minimum_conformer = bbprep.Conformer(
                molecule=conformer.molecule.clone(),
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=conformer.permutation,
            )
    ditopic_building_block = minimum_conformer.molecule
    ditopic_building_block.write(ligand_dir / "di_opt.mol")

    if run:
        for multiplier in multipliers:
            logging.info("doing: multi %s", multiplier)

            # Define a connectivity based on a multiplier.
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    tritopic_building_block: stoichiometry_t_d[0] * multiplier,
                    ditopic_building_block: stoichiometry_t_d[1] * multiplier,
                },
                graph_type=f"{stoichiometry_t_d[0] * multiplier}"
                f"P{stoichiometry_t_d[1] * multiplier}",
                graph_set="rx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )

            for idx, topology_code in enumerate(iterator.yield_graphs()):
                # Filter graphs for 1-loops.
                if contains_parallels(topology_code):
                    continue

                generated_conformers = []
                for midx, scale in enumerate(attempts):
                    name = f"p1_{multiplier}_{idx}_{midx}"

                    try:
                        if isinstance(scale, str) and "regraphed" in scale:
                            constructed_molecule = get_regraphed_molecule(
                                scale=scale,
                                topology_code=topology_code,
                                iterator=iterator,
                                bb_config=None,
                            )

                        else:
                            constructed_molecule = get_vertexset_molecule(
                                scale=scale,
                                topology_code=topology_code,
                                iterator=iterator,
                                bb_config=None,
                            )

                    except ValueError:
                        continue

                    constructed_molecule.write(
                        structure_dir / f"{name}_unopt.mol"
                    )
                    # Optimise and save.
                    logging.info("building %s", name)

                    molecule = atomistic_optimisation(
                        name=name,
                        molecule=constructed_molecule,
                        output_directory=calculation_dir,
                    )
                    molecule.write(structure_dir / f"{name}_optc.mol")

                    ey_file = calculation_dir / f"{name}_xtb.ey"
                    if not ey_file.exists():
                        ey = stko.XTBEnergy(
                            xtb_path="/home/atarzia/miniforge3/envs/meproduction/bin/xtb",
                            num_cores=4,
                            output_dir=calculation_dir / f"{name}_xtbey",
                        ).get_energy(molecule)
                        # ey =
                        #  stko.OpenMMEnergy(
                        # force_field=
                        # ForceField(
                        #         "openff_unconstrained-2.2.1.offxml"
                        #     ),
                        #     partial_charges_method
                        # ="espaloma-am1bcc",
                        # ).get_energy(molecule)
                        with ey_file.open("w") as f:
                            f.write(str(ey))
                    else:
                        logging.info("loading energy from %s", ey_file.name)
                        with ey_file.open("r") as f:
                            ey = float(f.read())
                    ey_kjmol = ey * 2625.5
                    energy_per_bb = (
                        ey_kjmol / iterator.get_num_building_blocks()
                    )

                    properties = {
                        "energy_per_bb": energy_per_bb,
                        "num_bbs": (iterator.get_num_building_blocks()),
                        "multiplier": multiplier,
                        "topology_idx": idx,
                        "mash_idx": midx,
                        "topology_code_vmap": tuple(
                            (int(i[0]), int(i[1]))
                            for i in topology_code.vertex_map
                        ),
                    }
                    cgx.utilities.AtomliteDatabase(database_path).add_molecule(
                        key=name,
                        molecule=molecule.with_centroid((0, 0, 0)),
                    )
                    cgx.utilities.AtomliteDatabase(
                        database_path
                    ).add_properties(
                        key=name,
                        property_dict=properties,
                    )
                    generated_conformers.append(
                        (
                            name,
                            molecule.with_centroid((0, 0, 0)),
                            energy_per_bb,
                        )
                    )

                min_energy_conformer = sorted(
                    generated_conformers, key=lambda p: p[2]
                )[0]
                min_energy_name, min_energy_structure, _ = min_energy_conformer

                min_energy_structure.write(
                    str(structure_dir / f"{min_energy_name}_optc.mol")
                )
                cgx.utilities.AtomliteDatabase(database_path).add_properties(
                    key=min_energy_name,
                    property_dict={"lowest_e_of_mash": True},
                )

    study_6_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_1.png",
    )


def main() -> None:
    """Run script."""
    args = _parse_args()

    case_study_6(args.run)


if __name__ == "__main__":
    main()
