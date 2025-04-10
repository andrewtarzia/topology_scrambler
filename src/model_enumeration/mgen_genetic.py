"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
import shutil
import time

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_generation import (
    analyse_cage,
    attempts,
    define_pairs,
    get_regraphed_molecule,
    get_vertexset_molecule,
    make_summary_plot2,
    passes_graph_bb_iso,
)
from model_enumeration.mgen_utilities import precursors_to_forcefield
from model_enumeration.utilities import contains_parallels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def roulette_mutate_population(  # noqa: PLR0913
    chromo_it: cgx.systems_optimisation.ChromosomeGenerator,
    chromosomes: dict[str, cgx.systems_optimisation.Chromosome],
    generator: np.random.Generator,
    gene_range: tuple[int, ...],
    selection: str,
    num_to_select: int,
    database: cgx.utilities.AtomliteDatabase,
) -> list[cgx.systems_optimisation.Chromosome]:
    """Mutate a list of chromosomes in the gene range only.

    Available selections for which chromosomes to mutate:

        random:
            uses generator.choice()

        roulette:
            adds weight to generator.choice() based on
            fitness/sum(fitness)

    """
    # Select chromosomes to mutate.
    if selection == "random":
        selected = generator.choice(
            np.asarray(list(chromosomes.values())),
            size=num_to_select,
        )
    elif selection == "roulette":
        fitness_values: list[float | int] = [
            database.get_property_entry(key).properties["fitness"]  # type: ignore[misc]
            for key in chromosomes
        ]
        # Handle if all fitness values are 0.
        try:
            weights = [i / sum(fitness_values) for i in fitness_values]
        except ZeroDivisionError:
            weights = [1 / len(fitness_values) for i in fitness_values]

        selected = generator.choice(
            np.asarray(list(chromosomes.values())),
            size=num_to_select,
            p=weights,
        )

    else:
        msg = f"{selection} is not defined."
        raise RuntimeError(msg)

    mutated = []
    known_types = set(chromo_it.chromosome_types.values())
    for chromosome in selected:
        gene_dict = {}
        chromosome_name = [0 for i in range(len(chromo_it.chromosome_map))]
        for gene_id in chromo_it.chromosome_map:
            if gene_id not in gene_range:
                gene = chromosome.name[gene_id]
            else:
                chromosome_options = tuple(
                    range(len(chromo_it.chromosome_map[gene_id]))
                )
                gene = generator.choice(chromosome_options)

            gene_value = chromo_it.chromosome_map[gene_id][gene]
            gene_type = chromo_it.chromosome_types[gene_id]
            gene_dict[gene_id] = (gene, gene_value, gene_type)
            chromosome_name[gene_id] = gene

        if "forcefield" in known_types:
            # In this case, the definer dict changes per chromosome!
            ff_id = next(
                i
                for i in chromo_it.chromosome_types
                if chromo_it.chromosome_types[i] == "forcefield"
            )
            definer_dict = gene_dict[ff_id][1]
        else:
            definer_dict = chromo_it.definer_dict
        mutated.append(
            cgx.systems_optimisation.Chromosome(
                name=tuple(chromosome_name),
                prefix=chromo_it.prefix,
                present_beads=chromo_it.present_beads,
                vdw_bond_cutoff=chromo_it.vdw_bond_cutoff,
                gene_dict=gene_dict,
                definer_dict=definer_dict,
                chromosomed_terms=chromo_it.chromosomed_terms,
            )
        )

    return mutated


def roulette_crossover_population(  # noqa: D103, PLR0913
    chromo_it: cgx.systems_optimisation.ChromosomeGenerator,
    chromosomes: dict[str, cgx.systems_optimisation.Chromosome],
    generator: np.random.Generator,
    selection: str,
    num_to_select: int,
    database: cgx.utilities.AtomliteDatabase,
) -> list[cgx.systems_optimisation.Chromosome]:
    # Select chromosomes to cross.
    if selection == "random":
        selected = generator.choice(
            np.asarray(list(chromosomes.values())),
            size=(num_to_select, 2),
        )
    elif selection == "roulette":
        fitness_values: list[float | int] = [
            database.get_property_entry(key).properties["fitness"]  # type: ignore[misc]
            for key in chromosomes
        ]
        # Handle if all fitness values are 0.
        try:
            weights = [i / sum(fitness_values) for i in fitness_values]
        except ZeroDivisionError:
            weights = [1 / len(fitness_values) for i in fitness_values]

        selected = generator.choice(
            np.asarray(list(chromosomes.values())),
            size=(num_to_select, 2),
            p=weights,
        )
    else:
        msg = f"{selection} is not defined."
        raise RuntimeError(msg)

    crossed = []
    for chromosome1, chromosome2 in selected:
        # Randomly select the genes to cross.
        nums_to_select_from = range(len(chromosome1.name))
        num_to_cross = generator.choice(nums_to_select_from, size=1)
        genes_to_cross = set(
            generator.choice(nums_to_select_from, size=num_to_cross[0])
        )

        # Cross them.
        new_chromosome1 = tuple(
            val if i not in genes_to_cross else chromosome2.name[i]
            for i, val in enumerate(chromosome1.name)
        )
        new_chromosome2 = tuple(
            val if i not in genes_to_cross else chromosome1.name[i]
            for i, val in enumerate(chromosome2.name)
        )

        # Append the new chromosomes.
        crossed.append(chromo_it.select_chromosome(new_chromosome1))
        crossed.append(chromo_it.select_chromosome(new_chromosome2))

    return crossed


def fitness_function(  # noqa: PLR0913
    chromosome: cgx.systems_optimisation.Chromosome,
    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
    options: dict,
) -> float:
    """Compute fitness."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    topology_idx, topology_code = chromosome.get_topology_information()
    building_block_config = chromosome.get_vertex_alignments()[0]

    name = f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"
    entry = database.get_entry(name)
    energy = entry.properties["energy_per_bb"]
    fitness = np.exp(-energy * options["beta"])
    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )
    return fitness


def structure_function(  # noqa: C901, PLR0912, PLR0915
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,
) -> None:
    """Compute structure."""
    database = cgx.utilities.AtomliteDatabase(database_path)

    topology_idx, topology_code = chromosome.get_topology_information()
    building_block_config = chromosome.get_vertex_alignments()[0]

    base_name = (
        f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"
    )
    if database.has_molecule(base_name):
        return

    # Check if this has been run before.
    known_entry = None
    for entry in database.get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        try:
            entry_tc = options["topology_codes"][
                entry.properties["topology_idx"]
            ]
            entry_bb_config = options["bb_configs"][
                entry.properties["bb_config_idx"]
            ]
        except KeyError:
            continue

        # Testing bb-config aware graph check.
        if not passes_graph_bb_iso(
            topology_code=topology_code,
            bb_config=building_block_config,
            run_topology_codes=[(entry_tc[1], entry_bb_config)],
        ):
            known_entry = entry
            break

    # Try to avoid recalculation if possible.
    if (
        known_entry is not None
        and known_entry.properties["base_name"] != base_name
    ):
        database.add_properties(
            key=base_name,
            property_dict={
                "is_duplicate": True,
                "duplicate_of": known_entry.key,
            },
        )

        try:
            nd_ = known_entry.properties["num_duplicates"] + 1
        except KeyError:
            nd_ = 1
        database.add_properties(
            key=known_entry.key,
            property_dict={"num_duplicates": nd_},
        )

        logging.info("%s is duplicate", base_name)
        return

    # Iterate over mashes.
    generated_conformers = []
    for midx, scale in enumerate(attempts):
        name = f"{base_name}_{midx}"

        try:
            if isinstance(scale, str) and "regraphed" in scale:
                constructed_molecule = get_regraphed_molecule(
                    scale=scale,
                    topology_code=topology_code,
                    iterator=options["iterator"],
                    bb_config=building_block_config,
                )

            else:
                constructed_molecule = get_vertexset_molecule(
                    scale=scale,
                    topology_code=topology_code,
                    iterator=options["iterator"],
                    bb_config=building_block_config,
                )
        except ValueError:
            continue

        constructed_molecule.write(calculation_output / f"{name}_unopt.mol")

        # Optimise and save.
        logging.info("building %s", name)

        try:
            potential_names = [
                f"{base_name}_{nmash_idx}"
                for nmash_idx in range(len(attempts))
            ]
            if scale is None:
                conformer = cgx.scram.graph_optimise_cage(
                    molecule=constructed_molecule,
                    name=name,
                    output_dir=calculation_output,
                    forcefield=options["forcefield"],
                    platform=None,
                    database_path=database_path,
                )
                # Copy the file over.
                wipfinal = calculation_output / f"{name}_wipfinal.mol"
                wipfinal_new = calculation_output / f"{name}_final.mol"
                if wipfinal.exists():
                    shutil.copy(wipfinal, wipfinal_new)

            else:
                conformer = cgx.scram.optimise_cage(
                    molecule=constructed_molecule,
                    name=name,
                    output_dir=calculation_output,
                    forcefield=options["forcefield"],
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
                energy_per_bb = cgx.utilities.get_energy_per_bb(
                    energy_decomposition=(conformer.energy_decomposition),
                    number_building_blocks=(
                        options["iterator"].get_num_building_blocks()
                    ),
                )

                l1, l2, multiplier = chromosome.prefix.split("_")
                properties = {
                    "base_name": base_name,
                    "forcefield_dict": (
                        options["forcefield"].get_forcefield_dictionary()
                    ),
                    "energy_per_bb": energy_per_bb,
                    "l1": l1,
                    "l2": l2,
                    "pair": f"{l1}_{l2}",
                    "num_components": num_components,
                    "num_bbs": (options["iterator"].get_num_building_blocks()),
                    "multiplier": multiplier,
                    "topology_idx": topology_idx,
                    "mash_idx": midx,
                    "topology_code_vmap": tuple(
                        (int(i[0]), int(i[1]))
                        for i in topology_code.vertex_map
                    ),
                    "bb_config_idx": building_block_config.idx,
                    "contains_parallels": contains_parallels(topology_code),
                    # Add here, if it gets here, then it is not duplicate.
                    "is_duplicate": False,
                    "num_duplicates": 0,
                }

                database.add_properties(key=name, property_dict=properties)
                generated_conformers.append(
                    (
                        name,
                        conformer.molecule.with_centroid((0, 0, 0)),
                        energy_per_bb,
                    )
                )

        except OpenMMException:
            logging.info("failed optimisation of %s", name)

    min_energy_conformer = sorted(generated_conformers, key=lambda p: p[2])[0]
    min_energy_name, min_energy_structure, _ = min_energy_conformer

    min_energy_structure.write(str(structure_output / f"{base_name}_optc.mol"))

    # Write base name to database.
    database.add_molecule(
        key=base_name,
        molecule=database.get_molecule(min_energy_name),
    )
    database.add_properties(
        key=base_name,
        property_dict=database.get_entry(min_energy_name).properties,
    )
    database.add_properties(key=base_name, property_dict={"is_base": True})
    analyse_cage(database_path=database_path, name=base_name)


def progress_plot(
    generations: list,
    output: pathlib.Path,
    num_generations: int,
) -> None:
    """Draw optimisation progress."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = [
        generation.calculate_fitness_values() for generation in generations
    ]

    ax.plot(
        [max(i) for i in fitnesses],
        markerfacecolor="#F9A03F",
        label="max",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    ax.plot(
        [np.mean(i) for i in fitnesses],
        markerfacecolor="#086788",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
        label="mean",
    )
    ax.plot(
        [min(i) for i in fitnesses],
        markerfacecolor="#7A8B99",
        label="min",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_xlim(0, num_generations)
    ax.set_xticks(range(0, num_generations + 1, 5))
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def plot_fitness_curve(
    figure_dir: pathlib.Path,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    x = np.linspace(0, 10, 100)
    for beta in (0.1, 1, 2, 5, 10, 100):
        y = np.exp(-x * beta)
        ax.plot(x, y, label=beta)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("energy", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / "fitness.png", dpi=360, bbox_inches="tight")
    fig.savefig(figure_dir / "fitness.pdf", dpi=360, bbox_inches="tight")
    plt.close()


def case_study_4(run: bool) -> None:  # noqa: C901, PLR0915
    """Run case study 4 studying PW heteroleptic systems."""
    wd = pathlib.Path("/home/atarzia/workingspace/model_enum_data/")

    calculation_dir = wd / "genetic4_calculations"
    calculation_dir.mkdir(exist_ok=True)
    ffcalculation_dir = calculation_dir / "ff_scan"
    ffcalculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "genetic4_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "genetic4_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "genetic4_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "genetic4"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "genetic4.db"

    plot_fitness_curve(figure_dir)

    stoichiometry_l_l_m = (1, 1, 1)
    ligand_measures = {
        "cs41a": {"ba": 1.5, "aa": 9.5, "bac": 145},
        "cs41b": {"ba": 5.7, "aa": 2.4, "bac": 145},
        "cs41c": {"ba": 1.5, "aa": 9.5, "bac": 150},
        "cs41d": {"ba": 5.7, "aa": 2.4, "bac": 150},
        "cs490": {"ba": 1.5, "aa": 5.9, "bac": 135},
    }
    ligand_types = {
        "cs41a": "twoarm",
        "cs41b": "twoarm",
        "cs41c": "twoarm",
        "cs41d": "twoarm",
        "cs490": "twoarm",
    }
    pairs_to_predict = [
        # large, small.
        (("cs490", "cs41c"), (9,)),
        (("cs490", "cs41d"), (9,)),
        (("cs490", "cs41a"), (9,)),
        (("cs490", "cs41b"), (9,)),
    ]

    pairs = define_pairs(pairs_to_predict, ligand_types)

    cs4_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbe": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "ede": ("angle", 180, 1e2),
        # Torsions.
        "bacab": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }

    if run:
        for pair in pairs:
            forcefield = precursors_to_forcefield(
                pair=pair,
                large=pairs[pair]["large"],
                small=pairs[pair]["small"],
                large_meas=ligand_measures[pairs[pair]["large_name"]],
                small_meas=ligand_measures[pairs[pair]["small_name"]],
                vdw_bond_cutoff=pairs[pair]["vdw_cutoff"],
                constant_definer_dict=cs4_definer_dict,
            )

            small_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["small"].get_building_block(),
                name=f"{pair}_{pairs[pair]['small'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            small_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['small'].get_name()}_optl.mol"
                )
            )

            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["tetra"].get_building_block(),
                name=pairs[pair]["tetra"].get_name(),
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            tetra_bb.write(
                str(ligand_dir / f"{pairs[pair]['tetra'].get_name()}_optl.mol")
            )

            large_bb = cgx.utilities.optimise_ligand(
                molecule=pairs[pair]["large"].get_building_block(),
                name=f"{pair}_{pairs[pair]['large'].get_name()}",
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            ).clone()
            large_bb.write(
                str(
                    ligand_dir
                    / f"{pair}_{pairs[pair]['large'].get_name()}_optl.mol"
                )
            )

            for multiplier in pairs[pair]["multipliers"]:
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_l_m[2] * multiplier,
                        large_bb: stoichiometry_l_l_m[0] * multiplier,
                        small_bb: stoichiometry_l_l_m[1] * multiplier,
                    },
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
                    graph_set="rx",
                )
                topology_codes = tuple(enumerate(iterator.yield_graphs()))
                logging.info(
                    "graph iteration has %s graphs", len(topology_codes)
                )

                possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                    iterator=iterator
                )
                logging.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )
                logging.info("no filtering applied!")

                chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
                    prefix=f"{pair}_{multiplier}",
                    present_beads=forcefield.get_present_beads(),
                    vdw_bond_cutoff=forcefield.get_vdw_bond_cutoff(),
                )
                chromo_it.add_gene(
                    iteration=topology_codes,
                    gene_type="topology",
                )
                chromo_it.add_gene(
                    iteration=possible_bbdicts,
                    gene_type="vertex_alignment",
                )

                # Define fitness calculator.
                fitness_calculator = (
                    cgx.systems_optimisation.FitnessCalculator(
                        fitness_function=fitness_function,
                        chromosome_generator=chromo_it,
                        structure_output=structure_dir,
                        calculation_output=calculation_dir,
                        database_path=database_path,
                        options={"beta": 5},
                    )
                )

                # Define structure calculator.
                structure_calculator = (
                    cgx.systems_optimisation.StructureCalculator(
                        structure_function=structure_function,
                        structure_output=structure_dir,
                        calculation_output=calculation_dir,
                        database_path=database_path,
                        options={
                            "topology_codes": list(topology_codes),
                            "bb_configs": possible_bbdicts,
                            "iterator": iterator,
                            "forcefield": forcefield,
                        },
                    )
                )

                seeds = [4]
                num_generations = 1
                selection_size = 10
                num_processes = 1
                timing_file = data_dir / f"np_{num_processes}.txt"
                for seed in seeds:
                    generator = np.random.default_rng(seed)

                    initial_population = chromo_it.select_random_population(
                        generator,
                        size=selection_size,
                    )

                    # Yield this.
                    generations = []
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=initial_population,
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=num_processes,
                    )

                    generation.run_structures()
                    _ = generation.calculate_fitness_values()
                    generations.append(generation)

                    for generation_id in range(1, num_generations + 1):
                        logging.info(
                            "doing generation %s of seed %s",
                            generation_id,
                            seed,
                        )
                        logging.info(
                            "initial size is %s.",
                            generation.get_generation_size(),
                        )
                        logging.info("doing mutations.")
                        merged_chromosomes = []

                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                list_of_chromosomes=generation.chromosomes,
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="random",
                                num_to_select=5,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                list_of_chromosomes=generation.chromosomes,
                                generator=generator,
                                gene_range=chromo_it.get_va_ids(),
                                selection="random",
                                num_to_select=5,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            chromo_it.crossover_population(
                                list_of_chromosomes=generation.chromosomes,
                                generator=generator,
                                selection="random",
                                num_to_select=5,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        # Add the best 5 to the new generation.
                        merged_chromosomes.extend(
                            generation.select_best(selection_size=5)
                        )

                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(
                                merged_chromosomes
                            ),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=num_processes,
                        )
                        logging.info(
                            "new size is %s.", generation.get_generation_size()
                        )

                        # Build, optimise and analyse each structure.
                        st = time.time()
                        generation.run_structures()
                        str_time = time.time() - st
                        st = time.time()
                        _ = generation.calculate_fitness_values()
                        fit_time = time.time() - st
                        with timing_file.open("a") as f:
                            f.write(f"{str_time},{fit_time}\n")

                        # Add final state to generations.
                        generations.append(generation)

                        # Select the best of the generation for the next
                        # generation.
                        best = generation.select_best(
                            selection_size=selection_size
                        )
                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(best),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=num_processes,
                        )
                        logging.info(
                            "final size is %s.",
                            generation.get_generation_size(),
                        )

                        progress_plot(
                            generations=generations,
                            output=figure_dir / f"fitness_progress_{seed}.png",
                            num_generations=num_generations,
                        )

                        # Output best structures as images.
                        best_chromosome = generation.select_best(
                            selection_size=1
                        )[0]

                        best_name = (
                            f"{best_chromosome.prefix}"
                            f"_{best_chromosome.name[0]}_"
                            f"b{best_chromosome.name[1]}"
                        )

                        logging.info(
                            "top scorer is %s (seed: %s)", best_name, seed
                        )

                # Report.
                found = set()
                for generation in generations:
                    for chromo in generation.chromosomes:
                        found.add(chromo.name)

                logging.info(
                    "%s chromosomes found in EA (of %s)",
                    len(found),
                    chromo_it.get_num_chromosomes(),
                )
                raise SystemExit

    fig, ax = plt.subplots(figsize=(8, 5))
    for num_processes in (1, 2, 3, 4, 5, 6, 7, 8):
        timing_file = data_dir / f"np_{num_processes}.txt"
        if not timing_file.exists():
            continue
        with timing_file.open("r") as f:
            lines = f.readlines()
        str_times = [float(i.strip().split(",")[0]) for i in lines]
        fit_times = [float(i.strip().split(",")[1]) for i in lines]
        if num_processes == 1:
            lbl1 = "structure"
            lbl2 = "fitness"
        else:
            lbl1 = None
            lbl2 = None
        ax.scatter(
            [num_processes for i in str_times],
            str_times,
            c="tab:blue",
            s=40,
            edgecolor="none",
            alpha=0.4,
        )
        ax.scatter(
            [num_processes for i in fit_times],
            fit_times,
            c="tab:orange",
            s=40,
            edgecolor="none",
            alpha=0.4,
        )
        ax.scatter(
            num_processes,
            sum(str_times) / len(str_times),
            c="tab:blue",
            s=100,
            edgecolor="k",
            label=lbl1,
        )
        ax.scatter(
            num_processes,
            sum(fit_times) / len(fit_times),
            c="tab:orange",
            s=100,
            edgecolor="k",
            label=lbl2,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num processes", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_ylim(0, None)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "timings.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
        pairs=pairs_to_predict,
        structure_dir=structure_dir,
    )


def main() -> None:
    """Run script."""
    args = _parse_args()

    case_study_4(args.run)
    raise SystemExit("interested in counting known entries, duplicates")
    raise SystemExit("and counting num parallels ove the generations")


if __name__ == "__main__":
    main()
