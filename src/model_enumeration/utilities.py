"""Utilities module."""

import matplotlib as mpl

tstr_cmap = mpl.colormaps["tab20"].resampled(20)
multi_cmap = {
    "1": tstr_cmap(0.0),
    "2": tstr_cmap(0.05),
    "3": tstr_cmap(0.1),
    "4": tstr_cmap(0.15),
    "5": tstr_cmap(0.2),
    "6": tstr_cmap(0.25),
    "7": tstr_cmap(0.30),
    "8": tstr_cmap(0.35),
    "9": tstr_cmap(0.40),
    "10": tstr_cmap(0.45),
    "11": tstr_cmap(0.5),
    "12": tstr_cmap(0.55),
}


def eb_str(no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def pore_str() -> str:
    """A unit str."""
    return r"pore size [$\mathrm{\AA}$]"
