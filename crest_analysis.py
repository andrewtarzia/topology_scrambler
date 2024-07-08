"""Perform crest analysis on ligand."""

import logging
import pathlib
import stko
import os
import shutil
import subprocess as sp
import uuid
from collections import abc

import matplotlib.pyplot as plt
import numpy as np
import stk
from utilities import (
    plot_xy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class Crest(stko.Optimizer):
    """Run CREST conformer search algorithm."""

    def __init__(  # noqa: PLR0913
        self,
        crest_path: pathlib.Path,
        xtb_path: pathlib.Path,
        gfn_method: str = "2",
        output_dir: str | None = None,
        num_cores: int = 4,
        charge: int = 0,
        electronic_temperature: float = 300,
        solvent_model: str = "gbsa",
        solvent: str | None = None,
        solvent_grid: str = "normal",
        num_unpaired_electrons: int = 0,
        unlimited_memory: bool = False,
        additional_commands: tuple[str, ...] = (),
    ) -> None:
        """Initialise calculator."""
        if solvent is not None:
            solvent = solvent.lower()
            if gfn_method in ("gfnff", "0"):
                msg = "XTB: No solvent valid for version", f" {gfn_method!r}."
                raise stko.InvalidSolventError(msg)
            if not stko.is_valid_xtb_solvent(
                gfn_version=int(gfn_method),
                solvent_model=solvent_model,
                solvent=solvent,
            ):
                msg = (
                    f"XTB: Solvent {solvent!r} and model {solvent_model!r}",
                    f" is invalid for version {gfn_method!r}.",
                )
                raise stko.InvalidSolventError(msg)

        self._check_path(crest_path)
        self._check_path(xtb_path)
        self._crest_path = crest_path
        self._xtb_path = xtb_path
        self._gfn_method = (
            f"--gfn{gfn_method}" if gfn_method not in ("gfnff",) else "--gfnff"
        )
        self._output_dir = None if output_dir is None else pathlib.Path(output_dir)
        self._additional_commands = additional_commands
        self._num_cores = str(num_cores)
        self._electronic_temperature = str(electronic_temperature)
        self._solvent_model = solvent_model
        self._solvent = solvent
        self._solvent_grid = solvent_grid
        self._charge = str(charge)
        self._num_unpaired_electrons = str(num_unpaired_electrons)
        self._unlimited_memory = unlimited_memory

    def _check_path(self, path: pathlib.Path | str) -> None:
        path = pathlib.Path(path)
        if not path.exists():
            msg = f"XTB or CREST not found at {path}"
            raise pathlib.PathError(msg)

    def _write_detailed_control(self) -> None:
        string = f"$gbsa\n   gbsagrid={self._solvent_grid}"

        with pathlib.Path("det_control.in").open("w") as f:
            f.write(string)

    def _is_complete(
        self,
        output_file: pathlib.Path | str,
        output_xyzs: abc.Iterable[pathlib.Path],
    ) -> bool:
        output_file = pathlib.Path(output_file)
        if not output_file.exists():
            # No simulation has been run.
            msg = "CREST run did not start"
            raise stko.NotStartedError(msg)

        for xyz in output_xyzs:
            if not xyz.exists():
                msg = f"CREST run did not complete, {xyz} is not present!"
                raise stko.NotCompletedError(msg)

        return True

    def _run_crest(self, xyz: str, out_file: pathlib.Path | str) -> None:
        out_file = pathlib.Path(out_file)

        # Modify the memory limit.
        memory = "ulimit -s unlimited ;" if self._unlimited_memory else ""

        if self._solvent is not None:
            solvent = f"--{self._solvent_model} {self._solvent}"
        else:
            solvent = ""

        additions = " ".join(self._additional_commands)

        cmd = (
            f"{memory} {self._crest_path} {xyz} "
            f"-xnam {self._xtb_path} "
            f"{solvent} -chrg {self._charge} "
            f"--etemp {self._electronic_temperature}"
            f"-uhf {self._num_unpaired_electrons} "
            f"{self._gfn_method} "
            f"-T {self._num_cores} {additions} -I det_control.in"
        )

        with out_file.open("w") as f:
            # Note that sp.call will hold the program until completion
            # of the calculation.
            sp.call(
                cmd,
                stdin=sp.PIPE,
                stdout=f,
                stderr=sp.PIPE,
                # Shell is required to run complex arguments.
                shell=True,  # noqa: S602
            )

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:
        """Optimise a solute-solvent pair."""
        if self._output_dir is None:
            output_dir = pathlib.Path(str(uuid.uuid4().int)).resolve()
        else:
            output_dir = self._output_dir.resolve()

        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir()
        init_dir = pathlib.Path.cwd()
        os.chdir(output_dir)

        try:
            xyz = "input.xyz"
            molecule.write(xyz)
            self._write_detailed_control()

            out_file = "crest.output"

            self._run_crest(xyz=xyz, out_file=out_file)

            # Check if the optimization is complete.
            output_xyzs = [
                pathlib.Path("crest_best.xyz"),
                pathlib.Path("crest_conformers.xyz"),
                pathlib.Path("crest_rotamers.xyz"),
            ]

            opt_complete = self._is_complete(out_file, output_xyzs)

            molecule = molecule.with_structure_from_file(pathlib.Path("crest_best.xyz"))

        finally:
            os.chdir(init_dir)

        if not opt_complete:
            msg = f"CREST run is incomplete for {molecule}."
            logging.warning(msg)

        return molecule


def run_conformer_analysis(  # noqa: PLR0913
    ligand_name: str,
    molecule: stk.Molecule,
    ligand_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    functional_group_factories: tuple[stk.FunctionalGroupFactory, ...],
    crest_path: pathlib.Path,
    xtb_path: pathlib.Path,
) -> dict:
    """Analyse conformers."""

    opt_file = ligand_dir / f"{ligand_name}_optl.mol"
    crest_run = calculation_dir / f"{ligand_name}_crest"
    ensemble_dir = crest_run / "ensemble"
    ensemble_dir.mkdir(exist_ok=True)

    molecule = stk.BuildingBlock.init_from_molecule(
        molecule=molecule,
        functional_groups=functional_group_factories,
    )

    if not opt_file.exists():
        # Run calculation.
        optimiser = Crest(
            crest_path=crest_path,
            xtb_path=xtb_path,
            output_dir=crest_run,
            gfn_method="2",
            num_cores=12,
            unlimited_memory=True,
            solvent="dmso",
            solvent_model="alpb",
            solvent_grid="verytight",
            additional_commands=(
                "--optlev extreme",
                # No z matrix sorting.
                "--nosz",
                "--keepdir",
                # Set energy threshold (kcal.mol)
                "--ewin 10",
            ),
            charge=0,
            electronic_temperature=300,
            num_unpaired_electrons=0,
        )

        opt_molecule = optimiser.optimize(molecule)
        opt_molecule.write(opt_file)

    num_atoms = molecule.get_num_atoms()
    ensemble = {}

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

        conf_molecule = molecule.with_position_matrix(np.array(position_matrix))

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
            "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(conf_molecule),
            "adjacent_distance": adjacent_distance,
        }

        ensemble[i]["molecule"].write(ensemble_dir / f"conf_{i}.mol")

    return ensemble


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/clever_challenge/")
    ligand_dir = wd / "ligands"
    ligand_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    calculation_dir = wd / "calculations"
    calculation_dir.mkdir(exist_ok=True)
    crest_path = wd / "env" / "bin" / "crest"
    xtb_path = wd / "env" / "bin" / "crest"

    ligands = {
        "ls1": {"smiles": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3"},
        "lf": {
            "smiles": (
                "C1=C(C2=CC=C(C3C=CC4C(=O)C5C=CC(C6=CC=C(C7=CC=CN=C7)C=C6)=CC"
                "=5C=4C=3)C=C2)C=NC=C1"
            ),
        },
        "ls9": {"smiles": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3"},
        "st5": {"input": ligand_dir / "st5_manual.mol"},
        "la": {"smiles": "C1=C(C2C=CC3C4C=CC(C5=CC=CN=C5)=CC=4C(=O)C=3C=2)C=NC=C1"},
    }

    for ligand in ligands:
        logging.info("doing %s", ligand)
        if "smiles" in ligands[ligand]:
            molecule = stk.BuildingBlock(ligands[ligand]["smiles"])
        elif "input" in ligands[ligand]:
            molecule = stk.BuildingBlock.init_from_file(ligands[ligand]["input"])

        molecule.write(ligand_dir / f"{ligand}_unopt.mol")

        ensemble = run_conformer_analysis(
            ligand_name=ligand,
            molecule=molecule,
            ligand_dir=ligand_dir,
            calculation_dir=calculation_dir,
            functional_group_factories=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
            crest_path=crest_path,
            xtb_path=xtb_path,
        )

        min_energy = min([ensemble[i]["energy"] for i in ensemble])

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 5))
        xwidth = 0.5
        relative_energies_kjmol = [
            (ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble
        ]
        xmin = min(relative_energies_kjmol)
        xmax = 20
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ax.hist(
            x=relative_energies_kjmol,
            bins=xbins,
            density=False,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("relative energy [kJ/mol]", fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"dist_energy_{ligand}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()

        for x_property in (
            "binder_angles",
            "binder_binder_angle",
            "binder_distance",
            "binder_adjacent_torsion",
            "adjacent_distance",
        ):
            plot_xy(
                xproperty=x_property,
                ensemble=ensemble,
                min_energy=min_energy,
                figure_dir=figure_dir,
                ligand_name=ligand,
            )


if __name__ == "__main__":
    main()
