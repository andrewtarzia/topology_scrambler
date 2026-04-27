"""Script to generate and optimise CG models."""

import logging
import os
import pathlib
from collections import defaultdict

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import atomlite
import cgexplore as cgx
import chemiscope
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def atomistic_data() -> None:
    """Make chemiscope output."""
    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "all_database_analysis"
    figure_dir.mkdir(exist_ok=True)

    database_paths = {"atomistic": wd / "atomistic_data" / "atomistic.db"}

    properties = defaultdict(list)
    structures = []
    for db_name, db_path in database_paths.items():
        if not db_path.exists():
            msg = f"database {db_name} not found."
            raise FileNotFoundError(msg)
        db = cgx.utilities.AtomliteDatabase(db_path)
        num_entries = db.get_num_entries()
        logger.info(
            "processing database %s with %s entries", db_name, num_entries
        )

        for entry in db.get_entries():
            if "dft_energy_per_bb" not in entry.properties:
                continue
            if "lowest_e_of_mash" not in entry.properties:
                continue

            if "p1" not in entry.key:
                continue

            energy = entry.properties["dft_energy_per_bb"]
            structures.append(
                stk.BuildingBlock.init_from_rdkit_mol(
                    atomlite.json_to_rdkit(entry.molecule)
                )
            )
            properties["key"].append(entry.key)
            properties["E_b / kjmol-1"].append(energy)
            properties["num_bbs"].append(int(entry.properties["num_bbs"]))

    min_energy = min(properties["E_b / kjmol-1"])
    properties["rel. r2SCAN-3c E_b / kjmol-1"] = [
        (i - min_energy) for i in properties["E_b / kjmol-1"]
    ]
    logger.info("saving %s entries", len(structures))
    chemiscope.write_input(
        path=str(figure_dir / "atomistic.json.gz"),
        frames=structures,
        properties=properties,
        meta={
            "name": "Selected structures with Eb<1 kJmol-1 from atomistic"
            " case study.",
            "description": (
                "Atomistic models from blind structure prediction."
            ),
            "authors": ["Andrew Tarzia"],
            "references": ["TBD"],
        },
        settings=chemiscope.quick_settings(
            map_settings={
                "y": {
                    "property": "rel. r2SCAN-3c E_b / kjmol-1",
                    "min": 0,
                    "max": 25,
                }
            },
            x="num_bbs",
            structure_settings={
                "atoms": True,
                "bonds": True,
                "spaceFilling": False,
            },
        ),
    )


def cu18_data() -> None:
    """Make chemiscope output."""
    wd = pathlib.Path("/home/tarziaa/workingspace/tscram_production/")
    (wd / "figures").mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "all_database_analysis"
    figure_dir.mkdir(exist_ok=True)

    data_restrictors = {"cu18": "cs490_cs41d"}
    database_paths = {"cu18": wd / "cu18_data" / "cu18.db"}

    properties = defaultdict(list)
    structures = []
    for db_name, db_path in database_paths.items():
        if not db_path.exists():
            msg = f"database {db_name} not found."
            raise FileNotFoundError(msg)
        db = cgx.utilities.AtomliteDatabase(db_path)
        num_entries = db.get_num_entries()
        logger.info(
            "processing database %s with %s entries", db_name, num_entries
        )

        for entry in db.get_entries():
            if data_restrictors[db_name] not in entry.key:
                continue
            if "energy_per_bb" not in entry.properties:
                continue
            energy = entry.properties["energy_per_bb"]
            if energy > 1:
                continue

            structures.append(
                stk.BuildingBlock.init_from_rdkit_mol(
                    atomlite.json_to_rdkit(entry.molecule)
                )
            )
            properties["key"].append(entry.key)
            properties["E_b / kjmol-1"].append(energy)
            properties["num_bbs"].append(int(entry.properties["num_bbs"]))
            properties["stoichstring"].append(entry.properties["stoichstring"])

    logger.info("saving %s entries", len(structures))
    shape_dict = chemiscope.convert_stk_bonds_as_shapes(
        frames=structures,
        bond_color="#00000",
        bond_radius=0.1,
    )
    shape_string = ",".join(shape_dict.keys())
    chemiscope.write_input(
        path=str(figure_dir / "cu18.json.gz"),
        frames=structures,
        properties=properties,
        meta={
            "name": "Selected structures with Eb<1 kJmol-1 from Cu18 case"
            " study with `cs490_cs41d` buidling block pair.",
            "description": ("Minimal models from blind structure prediction"),
            "authors": ["Andrew Tarzia"],
            "references": ["TBD"],
        },
        shapes=shape_dict,
        settings=chemiscope.quick_settings(
            x="num_bbs",
            y="E_b / kjmol-1",
            structure_settings={
                "shape": shape_string,
                "atoms": True,
                "bonds": False,
                "spaceFilling": False,
            },
        ),
    )


def main() -> None:
    """Run script."""
    atomistic_data()
    cu18_data()


if __name__ == "__main__":
    main()
