"""Utilities module."""

import matplotlib as mpl

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": tstr_cmap(0.05),
    "2": tstr_cmap(0.0),
    "3": tstr_cmap(0.1),
    "4": tstr_cmap(0.2),
    "5": tstr_cmap(0.3),
    "6": tstr_cmap(0.4),
    "7": tstr_cmap(0.5),
    "8": tstr_cmap(0.6),
    "9": tstr_cmap(0.7),
    "10": tstr_cmap(0.8),
    "11": tstr_cmap(0.15),
    "12": tstr_cmap(0.9),
}


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def pore_str() -> str:
    """A unit str."""
    return r"pore size [$\mathrm{\AA}$]"


def isomer_energy() -> float:
    """Get constant."""
    return 0.3


def max_uniformity_threshold() -> float:
    """Get constant."""
    return 0.3


def dihedral_state_threshold() -> float:
    """Get constant."""
    return 5.0
