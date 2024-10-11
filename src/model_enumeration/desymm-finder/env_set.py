"""Define environment."""

import pathlib


def stoich_map(tstr: str) -> int:
    """Stoichiometry maps to the number of building blocks."""
    return {
        "2P3": 5,
        "4P6": 10,
        "4P62": 10,
        "6P9": 15,
        "8P12": 20,
        "2P4": 6,
        "3P6": 9,
        "4P8": 12,
        "4P82": 12,
        "6P12": 18,
        "8P16": 24,
        "6P122": 18,
        "8P162": 24,
        "12P24": 36,
        "6P8": 14,
    }[tstr]


def convert_topo(topo_str: str) -> str:
    """Convert topology to fancy name."""
    return {
        "2P4": r"Tet$^{2}$Di$^{4}$",
        "3P6": r"Tet$^{3}_{3}$Di$^{6}$",
        "4P8": r"Tet$^{4}_{4}$Di$^{8}$",
        "4P82": r"Tet$^{4}_{2}$Di$^{8}$",
        "6P12": r"Tet$^{6}$Di$^{12}$",
        "6P122": r"Tet$^{6}_{2}$Di$^{12}$",
        "8P16": r"Tet$^{8}$Di$^{16}$",
        "8P162": r"Tet$^{8}_{2}$Di$^{16}$",
    }[topo_str]


class EnvVariables:
    """Define environment variables."""

    project_dir = pathlib.Path(
        "/home/atarzia/workingspace/desymm_finder/cg_models/"
    )
    project_dir.mkdir(exist_ok=True, parents=True)

    pymol_path = pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )

    cg_figures = project_dir / pathlib.Path("figures/")
    cg_figures.mkdir(exist_ok=True, parents=True)
    cg_structures = project_dir / pathlib.Path("structures/")
    cg_structures.mkdir(exist_ok=True, parents=True)
    cg_ligands = project_dir / pathlib.Path("ligands/")
    cg_ligands.mkdir(exist_ok=True, parents=True)
    cg_calculations = project_dir / pathlib.Path("calculations/")
    cg_calculations.mkdir(exist_ok=True, parents=True)
    cg_outputdata = project_dir / pathlib.Path("outputdata/")
    cg_outputdata.mkdir(exist_ok=True, parents=True)

    isomer_energy = 0.3
    eb_str = r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"
    max_uniformity_threshold = 0.3
    dihedral_state_threshold = 5
