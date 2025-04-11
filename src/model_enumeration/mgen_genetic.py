"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib
import shutil
import time
from collections import abc

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from model_enumeration.mgen_generation import (
    analyse_cage,
    attempts,
    get_regraphed_molecule,
    get_vertexset_molecule,
    make_summary_plot,
    make_summary_plot2,
    passes_graph_bb_iso,
)
from model_enumeration.mgen_utilities import (
    StericTwoC1Arm,
    a2bead_d,
    abead_c,
    abead_d,
    binder_bead,
    c2bead_d,
    cbead_c,
    cbead_d,
    e2bead_d,
    ebead_c,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)
from model_enumeration.utilities import (
    contains_parallels,
    eb_str,
    isomer_energy,
)

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

    if entry.properties["is_duplicate"]:
        energy = database.get_entry(
            entry.properties["duplicate_of"]
        ).properties["energy_per_bb"]
    else:
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
    l1, l2, stoichstring = chromosome.prefix.split("_")
    multiplier = stoichstring.split("-")[2]

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
        except (KeyError, IndexError):
            continue

        known_stoichstring = entry.properties["stoichstring"]
        if stoichstring != known_stoichstring:
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
        database.add_molecule(
            key=base_name,
            molecule=database.get_molecule(known_entry.key),
        )
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
                    "stoichstring": stoichstring,
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


def plot_timings(figure_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    """Plot timings."""
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


def plot_counters(  # noqa: C901, PLR0912, PLR0915
    database_path: pathlib.Path,
    stoichstring: str,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, (axx, ax1) = plt.subplots(ncols=2, figsize=(16, 5))

    count_has_parallels = {True: 0, False: 0}
    has_parallels = {True: [], False: []}
    gen_entries = {}
    done = 0
    done_path = {}
    min_gen_energy = {}
    maxbbidx = 0
    maxtidx = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        if entry.properties["stoichstring"] != stoichstring:
            continue

        energy = entry.properties["energy_per_bb"]
        count_has_parallels[entry.properties["contains_parallels"]] += 1
        has_parallels[entry.properties["contains_parallels"]].append(energy)

        if energy < 0.1:  # noqa: PLR2004
            c = "tab:blue"
            zorder = 2
            s = 50
        elif energy < 1:
            c = "tab:orange"
            zorder = 1
            s = 20
        else:
            c = "tab:gray"
            zorder = 0
            s = 10

        if "generation_id" in entry.properties:
            if entry.properties["generation_id"] not in gen_entries:
                gen_entries[entry.properties["generation_id"]] = []
                min_gen_energy[entry.properties["generation_id"]] = float(
                    "inf"
                )
            gen_entries[entry.properties["generation_id"]].append(entry)
            min_gen_energy[entry.properties["generation_id"]] = min(
                (energy, min_gen_energy[entry.properties["generation_id"]])
            )

        ax1.scatter(
            entry.properties["topology_idx"],
            entry.properties["bb_config_idx"],
            c=c,
            s=s,
            alpha=1,
            zorder=zorder,
        )
        done += 1
        maxbbidx = max((maxbbidx, entry.properties["bb_config_idx"]))
        maxtidx = max((maxtidx, entry.properties["topology_idx"]))
        if "generation_id" in entry.properties:
            if entry.properties["generation_id"] not in done_path:
                done_path[entry.properties["generation_id"]] = []

            if energy == min_gen_energy[entry.properties["generation_id"]]:
                done_path[entry.properties["generation_id"]] = [
                    entry.properties["topology_idx"],
                    entry.properties["bb_config_idx"],
                ]

    ax1.scatter(done_path[0][0], done_path[0][1], c="r", s=50)
    ax1.plot(
        [i[0] for i in done_path.values()],
        [i[1] for i in done_path.values()],
        c="k",
        lw=1,
        zorder=-1,
    )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("tidx", fontsize=16)
    ax1.set_ylabel("bidx", fontsize=16)
    ax1.set_title(f"{done} of {maxbbidx * maxtidx}", fontsize=16)

    nadded = []
    nduplicates = []
    nparallels = []
    cumul = []
    for gen_id, gen_list in gen_entries.items():
        nduplicates.append(
            sum([1 if i.properties["is_duplicate"] else 0 for i in gen_list])
        )
        nparallels.append(
            sum(
                [
                    1 if i.properties["contains_parallels"] else 0
                    for i in gen_list
                ]
            )
        )
        nadded.append(len(gen_list))
        if gen_id == 0:
            cumul.append(len(gen_list))
        else:
            cumul.append(cumul[-1] + len(gen_list))

    axx.plot(
        nparallels,
        markerfacecolor="#F9A03F",
        lw=2,
        label="n. parallels",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    axx.plot(
        nadded,
        markerfacecolor="#086788",
        lw=2,
        label="n. added",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    axx.plot(
        nduplicates,
        markerfacecolor="#7A8B99",
        lw=2,
        label="n. duplicates",
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )

    axx.tick_params(axis="both", which="major", labelsize=16)
    axx.set_xlabel("generation", fontsize=16)
    axx.set_ylabel("number", fontsize=16)
    axx.legend(fontsize=16)
    axx.set_ylim(0, None)
    axx.set_title(f"tot. gen: {cumul[-1]}", fontsize=16)

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


def plot_vs_parallels(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    count_has_parallels = {True: 0, False: 0}
    has_parallels = {True: [], False: []}

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue
        energy = entry.properties["energy_per_bb"]
        count_has_parallels[entry.properties["contains_parallels"]] += 1
        has_parallels[entry.properties["contains_parallels"]].append(energy)

    steps = range(len(count_has_parallels) - 1, -1, -1)
    xmin = 0
    xmax = max(has_parallels[False] + has_parallels[True])
    xwidth = 0.5
    ystep = 1
    xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
    for i, value in enumerate(has_parallels):
        ax.hist(
            x=has_parallels[value],
            bins=xbins,
            density=True,
            bottom=steps[i] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            label=value,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(eb_str(), fontsize=16)

    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * ystep)
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


def plot_energies(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> dict:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    gen_entries = {}
    min_energy = float("inf")
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        energy = entry.properties["energy_per_bb"]
        stoichstring = entry.properties["stoichstring"]
        min_energy = min((min_energy, energy))

        if stoichstring not in gen_entries:
            gen_entries[stoichstring] = {}

        if "generation_id" in entry.properties:
            gid = entry.properties["generation_id"]
            if gid not in gen_entries[stoichstring]:
                gen_entries[stoichstring][gid] = []

            gen_entries[stoichstring][gid].append(energy)

    for stoichstring, gen_energies in gen_entries.items():
        ax.plot(
            [np.min(gen_energies[i]) for i in gen_energies],
            lw=2,
            marker="o",
            markersize=10,
            label=stoichstring,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.legend(fontsize=16)
    ax.axhline(y=min_energy, c="k", ls="--")
    ax.axhline(y=isomer_energy(), c="r", ls="-")
    ax.set_yscale("log")
    ax.set_xlim(0, None)
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


def define_pairs(
    pairs_to_predict: abc.Sequence[tuple[abc.Sequence[str]]],
    ligand_types: dict[str, str],
) -> dict[str, dict[str, cgx.molecular.Precursor | tuple | int]]:
    """Define pairs from provided information."""
    pairs = {}
    for large, small in pairs_to_predict:
        name = f"{large}_{small}"

        if ligand_types[large] == "sixbead":
            large_prec = cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            )
        elif ligand_types[large] == "twoarm":
            large_prec = cgx.molecular.TwoC1Arm(bead=cbead_c, abead1=abead_c)

        else:
            msg = large
            raise NotImplementedError(msg)

        if ligand_types[small] == "twoarm":
            small_prec = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
        elif ligand_types[small] == "stwoarm":
            small_prec = StericTwoC1Arm(
                bead=cbead_d, abead1=abead_d, steric_bead=steric_bead
            )
        elif ligand_types[small] == "sixbead":
            small_prec = cgx.molecular.SixBead(
                bead=c2bead_d,
                abead1=a2bead_d,
                abead2=e2bead_d,
            )
        else:
            msg = small
            raise NotImplementedError(msg)

        pairs[name] = {
            "large_name": large,
            "small_name": small,
            "large": large_prec,
            "small": small_prec,
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead, abead1=binder_bead
            ),
            "vdw_cutoff": 2,
        }

    return pairs


def case_study_4(run: bool) -> None:  # noqa: C901, PLR0912, PLR0915
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

    stoichiometries_l_s_m = (
        (6, 12, 9),
        (12, 6, 9),
        (9, 9, 9),
        (2, 2, 2),
        (1, 3, 2),
        (3, 1, 2),
        (3, 3, 3),
        (3, 3, 3),
        (2, 4, 3),
        (4, 2, 3),
    )
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
        ("cs490", "cs41c"),
        ("cs490", "cs41d"),
        ("cs490", "cs41a"),
        ("cs490", "cs41b"),
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

    short_scan_config = {
        "seeds": [4, 12689, 18, 999],
        "count_to_spam": 2,
        "mutations": 2,
        "num_generations": 10,
        "selection_size": 10,
        "num_processes": 1,
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

            for stoichiometry_l_s_m in stoichiometries_l_s_m:
                stoichstring = "-".join([str(i) for i in stoichiometry_l_s_m])

                multiplier = stoichiometry_l_s_m[2]
                logging.info(
                    "doing: pair %s, stoichstring %s", pair, stoichstring
                )
                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        tetra_bb: stoichiometry_l_s_m[2],
                        large_bb: stoichiometry_l_s_m[0],
                        small_bb: stoichiometry_l_s_m[1],
                    },
                    graph_type=f"{1 * multiplier}P{2 * multiplier}",
                    graph_set="rx",
                )
                all_topology_codes = tuple(enumerate(iterator.yield_graphs()))
                topology_codes = []
                for tidx, tc in all_topology_codes:
                    if contains_parallels(tc):
                        continue
                    topology_codes.append((tidx, tc))

                logging.info(
                    "graph iteration has %s graphs (from %s)",
                    len(topology_codes),
                    len(all_topology_codes),
                )
                logging.info("filtering for parallels!")

                possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                    iterator=iterator
                )
                logging.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )

                chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
                    prefix=f"{pair}_{stoichstring}",
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
                            "topology_codes": list(all_topology_codes),
                            "bb_configs": possible_bbdicts,
                            "iterator": iterator,
                            "forcefield": forcefield,
                        },
                    )
                )

                timing_file = (
                    data_dir / f"np_{short_scan_config['num_processes']}.txt"
                )
                for seed in short_scan_config["seeds"]:
                    generator = np.random.default_rng(seed)

                    initial_population = chromo_it.select_random_population(
                        generator,
                        size=short_scan_config["selection_size"],
                    )

                    # Yield this.
                    generations = []
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=initial_population,
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=short_scan_config["num_processes"],
                    )

                    generation.run_structures()
                    _ = generation.calculate_fitness_values()
                    generations.append(generation)

                    # Add generational information.
                    for cs in generation.chromosomes:
                        topology_idx, topology_code = (
                            cs.get_topology_information()
                        )
                        building_block_config = cs.get_vertex_alignments()[0]
                        name = (
                            f"{cs.prefix}_{topology_idx}_"
                            f"b{building_block_config.idx}"
                        )
                        entry = cgx.utilities.AtomliteDatabase(
                            database_path
                        ).get_entry(name)
                        if "generation_id" not in entry.properties:
                            cgx.utilities.AtomliteDatabase(
                                database_path
                            ).add_properties(
                                key=name,
                                property_dict={
                                    "generation_id": 0,
                                    "generation_seed": seed,
                                },
                            )
                    best_chromosome = generation.select_best(selection_size=1)[
                        0
                    ]

                    best_name = (
                        f"{best_chromosome.prefix}"
                        f"_{best_chromosome.get_topology_information()[0]}_"
                        f"b{best_chromosome.get_vertex_alignments()[0].idx}"
                    )

                    count_unchanged = 0
                    previous_best = best_name

                    for generation_id in range(
                        1, short_scan_config["num_generations"] + 1
                    ):
                        logging.info(
                            "doing generation %s of seed %s",
                            generation_id,
                            seed,
                        )

                        logging.info(
                            "initial size is %s.",
                            generation.get_generation_size(),
                        )
                        if (
                            count_unchanged
                            == short_scan_config["count_to_spam"]
                        ):
                            count_unchanged = 0
                            logging.info("doing 2x mutations.")
                            num_mutate = short_scan_config["mutations"] * 2
                        else:
                            logging.info("doing mutations.")
                            num_mutate = short_scan_config["mutations"] * 1

                        merged_chromosomes = []
                        merged_chromosomes.extend(
                            chromo_it.mutate_population(
                                list_of_chromosomes=generation.chromosomes,
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="random",
                                num_to_select=num_mutate,
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
                                num_to_select=num_mutate,
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
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            roulette_mutate_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_vertex_alignments()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                gene_range=chromo_it.get_topo_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )
                        merged_chromosomes.extend(
                            roulette_mutate_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_vertex_alignments()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                gene_range=chromo_it.get_va_ids(),
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        merged_chromosomes.extend(
                            roulette_crossover_population(
                                chromo_it=chromo_it,
                                chromosomes={
                                    (
                                        f"{i.prefix}"
                                        f"_{i.get_topology_information()[0]}"
                                        f"_b{i.get_vertex_alignments()[0].idx}"
                                    ): i
                                    for i in generation.chromosomes
                                },
                                generator=generator,
                                selection="roulette",
                                num_to_select=num_mutate,
                                database=cgx.utilities.AtomliteDatabase(
                                    database_path
                                ),
                            )
                        )

                        # Add the best 5 to the new generation.
                        merged_chromosomes.extend(
                            generation.select_best(selection_size=num_mutate)
                        )

                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(
                                merged_chromosomes
                            ),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=short_scan_config["num_processes"],
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

                        # Add generational information.
                        for cs in generation.chromosomes:
                            topology_idx, topology_code = (
                                cs.get_topology_information()
                            )
                            building_block_config = cs.get_vertex_alignments()[
                                0
                            ]
                            name = (
                                f"{cs.prefix}_{topology_idx}_"
                                f"b{building_block_config.idx}"
                            )
                            entry = cgx.utilities.AtomliteDatabase(
                                database_path
                            ).get_entry(name)
                            if "generation_id" not in entry.properties:
                                cgx.utilities.AtomliteDatabase(
                                    database_path
                                ).add_properties(
                                    key=name,
                                    property_dict={
                                        "generation_id": generation_id,
                                        "generation_seed": seed,
                                    },
                                )

                        # Add final state to generations.
                        generations.append(generation)

                        # Select the best of the generation for the next
                        # generation.
                        best = generation.select_best(
                            selection_size=short_scan_config["selection_size"]
                        )
                        generation = cgx.systems_optimisation.Generation(
                            chromosomes=chromo_it.dedupe_population(best),
                            fitness_calculator=fitness_calculator,
                            structure_calculator=structure_calculator,
                            num_processes=short_scan_config["num_processes"],
                        )
                        logging.info(
                            "final size is %s.",
                            generation.get_generation_size(),
                        )

                        progress_plot(
                            generations=generations,
                            output=figure_dir
                            / f"fp_{pair}_{stoichstring}_{seed}.png",
                            num_generations=short_scan_config[
                                "num_generations"
                            ],
                        )

                        # Output best structures as images.
                        best_chromosome = generation.select_best(
                            selection_size=1
                        )[0]

                        best_name = (
                            f"{best_chromosome.prefix}"
                            f"_{best_chromosome.get_topology_information()[0]}_"
                            f"b{best_chromosome.get_vertex_alignments()[0].idx}"
                        )
                        if best_name != previous_best:
                            count_unchanged = 0
                            previous_best = best_name
                        else:
                            count_unchanged += 1

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
            break

    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_4.png",
        pairs=tuple((i, None) for i in pairs_to_predict),
        structure_dir=structure_dir,
    )
    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_3.png",
        pairs=tuple((i, None) for i in pairs_to_predict),
    )

    plot_energies(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_2.png",
    )
    for stoichiometry_l_s_m in stoichiometries_l_s_m:
        stoichstring = "-".join([str(i) for i in stoichiometry_l_s_m])
        plot_counters(
            database_path=database_path,
            stoichstring=stoichstring,
            figure_dir=figure_dir,
            filename=f"mgen_1_{stoichstring}.png",
        )
    plot_vs_parallels(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="mgen_9.png",
    )
    plot_timings(figure_dir, data_dir)
    logging.info(
        "first test run clearly showed parallels not worth simulating"
    )
    logging.info("so started again without them, keeping that data.")
    logging.info(
        "what I think is interesting, is having an algo that adapts to that"
    )


def main() -> None:
    """Run script."""
    args = _parse_args()

    case_study_4(args.run)


if __name__ == "__main__":
    main()
