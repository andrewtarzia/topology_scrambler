"""Definition of conversion utilities."""

import logging
import pathlib

import bbprep
import numpy as np
import stk
import stko
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def extract_ensemble(molecule: stk.Molecule, crest_run: pathlib.Path) -> dict:
    """Extract and save an ensemble from a crest run."""
    ensemble_dir = crest_run / "ensemble"
    num_atoms = molecule.get_num_atoms()
    ensemble = {}
    ensemble_dir.mkdir(exist_ok=True, parents=True)

    # Calculate geometrical properties.
    conformer_file = crest_run / "crest_conformers.xyz"
    with conformer_file.open("r") as f:
        linex = f.readlines()

    split_line = linex[0].rstrip()
    for i, conformers in enumerate("".join(linex).split(split_line)):
        lines = conformers.split("\n")

        if len(lines) != num_atoms + 3:
            continue

        energy = lines[1]
        position_matrix = []
        for line in lines[2:]:
            splits = line.rstrip().split()
            if len(splits) != 4:  # noqa: PLR2004
                continue
            symb, x, y, z = splits
            x = float(x)
            y = float(y)
            z = float(z)

            position_matrix.append(np.array((x, y, z)))

        conf_molecule = molecule.with_position_matrix(
            np.array(position_matrix)
        )

        calc = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        adjacent_centroids = calc.get_adjacent_centroids(conf_molecule)
        adjacent_distance = np.linalg.norm(
            adjacent_centroids[0] - adjacent_centroids[1]
        )

        ensemble[i] = {
            "energy": float(energy),
            "molecule": conf_molecule,
            "binder_angles": calc.get_binder_angles(conf_molecule),
            "binder_binder_angle": calc.get_binder_binder_angle(conf_molecule),
            "binder_distance": calc.get_binder_distance(conf_molecule),
            "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(
                conf_molecule
            ),
            "adjacent_distance": adjacent_distance,
            "binder_com_angle": calc.get_binder_centroid_angle(conf_molecule),
        }

        ensemble[i]["molecule"].write(ensemble_dir / f"conf_{i}.mol")
    return ensemble


def get_ligand_bb(
    path: pathlib.Path,
    optl_path: pathlib.Path,
) -> stk.BuildingBlock:
    """Get building block for the target ligand and prepare for cage model."""
    try:
        return stk.BuildingBlock.init_from_file(
            path=path,
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        )
    except OSError:
        temp = stk.BuildingBlock.init_from_file(
            path=optl_path,
            functional_groups=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
        )
        generator = bbprep.generators.ETKDG(num_confs=100)
        ensemble = generator.generate_conformers(temp)
        process = bbprep.DitopicFitter(ensemble=ensemble)
        min_molecule = process.get_minimum()
        min_molecule.molecule.write(path)

    return stk.BuildingBlock.init_from_file(
        path=path,
        functional_groups=(
            stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
        ),
    )


def optimisation_sequence(
    cage: stk.Molecule,
    name: str,
    calculation_dir: pathlib.Path,
) -> stk.Molecule:
    """Cage optimisation sequence."""
    raise SystemExit("generalise dirs")
    gulp_dir = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")

    gulp1_output = calculation_dir / f"{name}_gulp1.mol"
    gulp2_output = calculation_dir / f"{name}_gulp2.mol"

    if not gulp1_output.exists():
        output_dir = calculation_dir / f"{name}_gulp1"

        logging.info("    UFF4MOF optimisation 1 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_dir,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(cage)
        gulp1_mol = gulp_opt.optimize(mol=cage)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info("    loading %s", gulp1_output)
        gulp1_mol = cage.with_structure_from_file(gulp1_output)

    if not gulp2_output.exists():
        output_dir = calculation_dir / f"{name}_gulp2"
        logging.info("    UFF4MOF optimisation 2 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_dir,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=False,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        logging.info("    loading %s", gulp2_output)
        gulp2_mol = cage.with_structure_from_file(gulp2_output)

    return cage.with_structure_from_file(gulp2_output)


def desymm_optimisation_sequence(  # noqa: PLR0915
    mol: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
) -> stk.Molecule:
    """Cage optimisation sequence."""
    raise SystemExit("generalise dirs")
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")
    xtb_path = wd / "env" / "bin" / "xtb"

    gulp1_output = calc_dir / f"{name}_gulp1.mol"
    gulp2_output = calc_dir / f"{name}_gulp2.mol"
    gulpmd_output = calc_dir / f"{name}_gulpmd.mol"
    xtbopt_output = calc_dir / f"{name}_xtb.mol"
    xtbsolvopt_output = calc_dir / f"{name}_xtb_dmso.mol"

    if not xtb_path.exists():
        msg = f"xtb is not installed here: {xtb_path}"
        raise ValueError(msg)
    if not gulp_path.exists():
        msg = f"gulp is not installed here: {gulp_path}"
        raise ValueError(msg)

    if not gulp1_output.exists():
        output_dir = calc_dir / f"{name}_gulp1"

        logging.info("    UFF4MOF optimisation 1 of %s CG: True", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(mol)
        gulp1_mol = gulp_opt.optimize(mol=mol)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info("    loading %s", gulp1_output)
        gulp1_mol = mol.with_structure_from_file(str(gulp1_output))

    if not gulp2_output.exists():
        output_dir = calc_dir / f"{name}_gulp2"

        logging.info("    UFF4MOF optimisation 2 of %s CG: False", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=False,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        logging.info("    loading %s", gulp2_output)
        gulp2_mol = mol.with_structure_from_file(str(gulp2_output))

    if not gulpmd_output.exists():
        logging.info("    UFF4MOF equilib MD of %s", name)
        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=gulp_path,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmde",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.25,
            equilbration=0.5,
            production=0.5,
            N_conformers=2,
            opt_conformers=False,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulp2_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulp2_mol)

        logging.info("    UFF4MOF production MD of %s", name)
        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=gulp_path,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmd",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.75,
            equilbration=0.5,
            production=200.0,
            N_conformers=40,
            opt_conformers=True,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulpmd_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulpmd_mol)
        gulpmd_mol.write(gulpmd_output)
    else:
        logging.info("    loading %s", gulpmd_output)
        gulpmd_mol = mol.with_structure_from_file(str(gulpmd_output))

    if not xtbopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbopt"
        logging.info("    xtb optimisation of %s", name)
        xtb_opt = stko.XTB(
            xtb_path=xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent=None,
        )
        xtbopt_mol = xtb_opt.optimize(mol=gulpmd_mol)
        xtbopt_mol.write(xtbopt_output)
    else:
        logging.info("    loading %s", xtbopt_output)
        xtbopt_mol = mol.with_structure_from_file(str(xtbopt_output))

    if not xtbsolvopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbsolvopt"
        logging.info("    solvated xtb optimisation of %s", name)
        xtb_opt = stko.XTB(
            xtb_path=xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent_model="alpb",
            solvent="dmso",
            solvent_grid="verytight",
        )
        xtbsolvopt_mol = xtb_opt.optimize(mol=xtbopt_mol)
        xtbsolvopt_mol.write(xtbsolvopt_output)
    else:
        logging.info("    loading %s", xtbsolvopt_output)
        xtbsolvopt_mol = mol.with_structure_from_file(str(xtbsolvopt_output))

    return mol.with_structure_from_file(str(xtbsolvopt_output))
