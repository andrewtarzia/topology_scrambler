"""Utilities module."""

import os

# A fix for something with threads.
os.environ["OMP_NUM_THREADS"] = "6"
import cgexplore as cgx
import matplotlib as mpl
from matplotlib.colors import to_hex

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": to_hex(tstr_cmap(0.05)),
    "2": to_hex(tstr_cmap(0.0)),
    "3": to_hex(tstr_cmap(0.1)),
    "4": to_hex(tstr_cmap(0.2)),
    "5": to_hex(tstr_cmap(0.3)),
    "6": to_hex(tstr_cmap(0.4)),
    "7": to_hex(tstr_cmap(0.5)),
    "8": to_hex(tstr_cmap(0.6)),
    "9": to_hex(tstr_cmap(0.7)),
    "10": to_hex(tstr_cmap(0.8)),
    "11": to_hex(tstr_cmap(0.15)),
    "12": to_hex(tstr_cmap(0.9)),
}
topology_cmap = {
    "2P4": to_hex(tstr_cmap(0.0)),
    "3P6": to_hex(tstr_cmap(0.1)),
    "4P8": to_hex(tstr_cmap(0.2)),
    "4P82": to_hex(tstr_cmap(0.5)),
    "6P12": to_hex(tstr_cmap(0.9)),
    "6P122": to_hex(tstr_cmap(0.3)),
    "8P16": to_hex(tstr_cmap(0.4)),
    "8P162": to_hex(tstr_cmap(0.7)),
    "2P3": to_hex(tstr_cmap(0.05)),
    "4P6": to_hex(tstr_cmap(0.15)),
    "4P62": to_hex(tstr_cmap(0.25)),
    "6P9": to_hex(tstr_cmap(0.35)),
    "8P12": to_hex(tstr_cmap(0.45)),
}


core_bead = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)

core_bead2 = cgx.molecular.CgBead(
    element_string="O",
    bead_class="o",
    bead_type="o",
    coordination=2,
)


arm_bead = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)


binder_bead = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)


tetragonal_bead = cgx.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)

trigonal_bead = cgx.molecular.CgBead(
    element_string="C",
    bead_class="n",
    bead_type="n",
    coordination=3,
)


tetragonal_bead2 = cgx.molecular.CgBead(
    element_string="Cr",
    bead_class="y",
    bead_type="y",
    coordination=4,
)

trigonal_bead2 = cgx.molecular.CgBead(
    element_string="Ge",
    bead_class="x",
    bead_type="x",
    coordination=3,
)


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def isomer_energy() -> float:
    """Get constant."""
    return 0.3


def contains_parallels(topology_code: cgx.scram.TopologyCode) -> bool:
    """True if the graph contains "1-loops"."""
    weighted_graph = topology_code.get_weighted_graph()
    num_parallel_edges = len(
        [
            i
            for i in weighted_graph.edges()
            if i == 2  # noqa: PLR2004
        ]
    )

    return num_parallel_edges != 0
