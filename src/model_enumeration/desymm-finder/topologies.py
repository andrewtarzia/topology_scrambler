"""Topologies module."""

import logging
from collections import abc

import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def topology_bead_legend(tstr: str) -> dict:
    """Get legend for a topology."""
    outer_colour = "#c057a1"
    crosser_colour = "#17becf"
    third_colour = "#75499c"

    if tstr == "4P82":
        return {"outer": outer_colour, "crosser": crosser_colour}

    if tstr in ("2P4", "3P6", "4P8", "6P12"):
        return {"all": "#c057a1"}

    if tstr == "6P122":
        return {"outer": outer_colour, "crosser": crosser_colour}

    if tstr == "8P16":
        return {
            "outer-1": outer_colour,
            "crosser": crosser_colour,
            "outer-2": third_colour,
        }

    if tstr == "8P162":
        return {"outer": outer_colour, "crosser": crosser_colour}

    msg = f"{tstr} not defined."
    raise ValueError(msg)


def topology_bead_map(tstr: str, tor_label: str) -> str:
    """Map topology to beads."""
    outer_colour = "#c057a1"
    crosser_colour = "#17becf"
    third_colour = "#75499c"

    if tstr == "4P82":
        pot_labels = {
            "Pb_Ba_Ag_Ba_Pb": outer_colour,
            "Pb_Ba_N_Ba_Pb": crosser_colour,
            "Pb_Ba_O_Ba_Pb": crosser_colour,
            "Pb_Ba_Cr_Ba_Pb": crosser_colour,
            "Pb_Ba_Ne_Ba_Pb": crosser_colour,
            "Pb_Ba_Na_Ba_Pb": outer_colour,
            "Pb_Ba_Mg_Ba_Pb": outer_colour,
            "Pb_Ba_Al_Ba_Pb": outer_colour,
        }
        return pot_labels[tor_label]

    if tstr in ("2P4", "3P6", "4P8", "6P12"):
        return "#c057a1"

    if tstr == "6P122":
        pot_labels = {
            "Pb_Ba_Ag_Ba_Pb": crosser_colour,
            "Pb_Ba_N_Ba_Pb": crosser_colour,
            "Pb_Ba_O_Ba_Pb": crosser_colour,
            "Pb_Ba_Cr_Ba_Pb": crosser_colour,
            "Pb_Ba_Ne_Ba_Pb": crosser_colour,
            "Pb_Ba_Na_Ba_Pb": crosser_colour,
            "Pb_Ba_Mg_Ba_Pb": outer_colour,
            "Pb_Ba_Al_Ba_Pb": outer_colour,
            "Pb_Ba_Si_Ba_Pb": outer_colour,
            "Pb_Ba_P_Ba_Pb": outer_colour,
            "Pb_Ba_S_Ba_Pb": outer_colour,
            "Pb_Ba_Ni_Ba_Pb": outer_colour,
        }
        return pot_labels[tor_label]

    if tstr == "8P162":
        pot_labels = {
            "Pb_Ba_Ag_Ba_Pb": crosser_colour,
            "Pb_Ba_N_Ba_Pb": crosser_colour,
            "Pb_Ba_O_Ba_Pb": crosser_colour,
            "Pb_Ba_Cr_Ba_Pb": crosser_colour,
            "Pb_Ba_Ne_Ba_Pb": crosser_colour,
            "Pb_Ba_Na_Ba_Pb": crosser_colour,
            "Pb_Ba_Mg_Ba_Pb": crosser_colour,
            "Pb_Ba_Al_Ba_Pb": crosser_colour,
            "Pb_Ba_Si_Ba_Pb": outer_colour,
            "Pb_Ba_P_Ba_Pb": outer_colour,
            "Pb_Ba_S_Ba_Pb": outer_colour,
            "Pb_Ba_Ni_Ba_Pb": outer_colour,
            "Pb_Ba_Ar_Ba_Pb": outer_colour,
            "Pb_Ba_Cu_Ba_Pb": outer_colour,
            "Pb_Ba_Ca_Ba_Pb": outer_colour,
            "Pb_Ba_Sc_Ba_Pb": outer_colour,
        }
        return pot_labels[tor_label]

    if tstr == "8P16":
        pot_labels = {
            "Pb_Ba_Ag_Ba_Pb": crosser_colour,
            "Pb_Ba_N_Ba_Pb": crosser_colour,
            "Pb_Ba_O_Ba_Pb": crosser_colour,
            "Pb_Ba_Cr_Ba_Pb": crosser_colour,
            "Pb_Ba_Ne_Ba_Pb": crosser_colour,
            "Pb_Ba_Na_Ba_Pb": crosser_colour,
            "Pb_Ba_Mg_Ba_Pb": crosser_colour,
            "Pb_Ba_Al_Ba_Pb": crosser_colour,
            "Pb_Ba_Si_Ba_Pb": third_colour,
            "Pb_Ba_P_Ba_Pb": third_colour,
            "Pb_Ba_S_Ba_Pb": third_colour,
            "Pb_Ba_Ni_Ba_Pb": third_colour,
            "Pb_Ba_Ar_Ba_Pb": outer_colour,
            "Pb_Ba_Cu_Ba_Pb": outer_colour,
            "Pb_Ba_Ca_Ba_Pb": outer_colour,
            "Pb_Ba_Sc_Ba_Pb": outer_colour,
        }
        return pot_labels[tor_label]

    msg = f"{tstr} not defined."
    raise ValueError(msg)


def topology_bb_dicts(
    tstr: str,
    tet_bb: stk.BuildingBlock,
    di_bbs: tuple[stk.BuildingBlock],
) -> dict:
    """Building block dictionaries for specific topologies."""
    if tstr == "2P4":
        return {
            tet_bb: (0, 1),
            di_bbs[0]: (2,),
            di_bbs[1]: (3,),
            di_bbs[2]: (4,),
            di_bbs[3]: (5,),
        }
    if tstr == "3P6":
        return {
            tet_bb: (0, 1, 2),
            di_bbs[0]: (3,),
            di_bbs[1]: (4,),
            di_bbs[2]: (5,),
            di_bbs[3]: (6,),
            di_bbs[4]: (7,),
            di_bbs[5]: (8,),
        }
    if tstr in ("4P8", "4P82"):
        return {
            tet_bb: (0, 1, 2, 3),
            di_bbs[0]: (4,),
            di_bbs[1]: (5,),
            di_bbs[2]: (6,),
            di_bbs[3]: (7,),
            di_bbs[4]: (8,),
            di_bbs[5]: (9,),
            di_bbs[6]: (10,),
            di_bbs[7]: (11,),
        }
    if tstr in ("6P12", "6P122"):
        return {
            tet_bb: (0, 1, 2, 3, 4, 5),
            di_bbs[0]: (6,),
            di_bbs[1]: (7,),
            di_bbs[2]: (8,),
            di_bbs[3]: (9,),
            di_bbs[4]: (10,),
            di_bbs[5]: (11,),
            di_bbs[6]: (12,),
            di_bbs[7]: (13,),
            di_bbs[8]: (14,),
            di_bbs[9]: (15,),
            di_bbs[10]: (16,),
            di_bbs[11]: (17,),
        }
    if tstr in ("8P16", "8P162"):
        return {
            tet_bb: (0, 1, 2, 3, 4, 5, 6, 7),
            di_bbs[0]: (8,),
            di_bbs[1]: (9,),
            di_bbs[2]: (10,),
            di_bbs[3]: (11,),
            di_bbs[4]: (12,),
            di_bbs[5]: (13,),
            di_bbs[6]: (14,),
            di_bbs[7]: (15,),
            di_bbs[8]: (16,),
            di_bbs[9]: (17,),
            di_bbs[10]: (18,),
            di_bbs[11]: (19,),
            di_bbs[12]: (20,),
            di_bbs[13]: (21,),
            di_bbs[14]: (22,),
            di_bbs[15]: (23,),
        }

    msg = f"{tstr} not defined."
    raise ValueError(msg)


def topology_het_bb_dicts(
    tstr: str,
    tet_bb: stk.BuildingBlock,
    di_bb_1: stk.BuildingBlock,
    di_bb_2: stk.BuildingBlock,
) -> dict:
    """Building block dictionaries for specific heteroleptic topologies."""
    if tstr == "4P82":
        return {
            tet_bb: (0, 1, 2, 3),
            di_bb_1: (4, 9, 10, 11),
            di_bb_2: (5, 6, 7, 8),
        }
    if tstr == "6P122":
        return {
            tet_bb: (0, 1, 2, 3, 4, 5),
            di_bb_1: (6, 7, 8, 9, 10, 11),
            di_bb_2: (12, 13, 14, 15, 16, 17),
        }
    if tstr in ("8P16", "8P162"):
        return {
            tet_bb: (0, 1, 2, 3, 4, 5, 6, 7),
            di_bb_1: (8, 9, 10, 11, 12, 13, 14, 15),
            di_bb_2: (16, 17, 18, 19, 20, 21, 22, 23),
        }

    msg = f"{tstr} not defined."
    raise ValueError(msg)


def cage_topology_options(
    study: str,
) -> abc.Sequence[tuple[str, abc.Callable]]:
    """Topology options."""
    if study == "homoleptic_2p4":
        topologies = (
            ("2P4", stk.cage.M2L4Lantern),
            ("3P6", stk.cage.M3L6),
            ("4P8", CGM4L8),
            ("4P82", M4L82),
            ("6P12", stk.cage.M6L12Cube),
            ("6P122", M6L122),
            ("8P16", stk.cage.EightPlusSixteen),
            ("8P162", M8L162),
        )
    elif study == "shortened_homoleptic_2p4":
        topologies = (
            ("2P4", stk.cage.M2L4Lantern),
            ("3P6", stk.cage.M3L6),
            ("4P8", CGM4L8),
            ("4P82", M4L82),
            ("6P12", stk.cage.M6L12Cube),
        )
    elif study == "repr_6P122":
        topologies = (("6P122", M6L122),)

    elif study == "homoleptic_2p3":
        topologies = (
            ("2P3", stk.cage.TwoPlusThree),
            ("4P6", stk.cage.FourPlusSix),
            ("4P62", stk.cage.FourPlusSix2),
            ("6P9", stk.cage.SixPlusNine),
            ("8P12", stk.cage.EightPlusTwelve),
        )

    elif study == "homoleptic_3p4":
        topologies = (("6P8", stk.cage.SixPlusEight),)

    elif study == "homoleptic_2p3_3x":
        topologies = (
            ("2P3", stk.cage.TwoPlusThree),
            ("4P6", stk.cage.FourPlusSix),
            ("4P62", stk.cage.FourPlusSix2),
            ("6P9", stk.cage.SixPlusNine),
            ("8P12", stk.cage.EightPlusTwelve),
        )

    elif study == "homoleptic_2p4_4x":
        topologies = (
            ("2P4", stk.cage.M2L4Lantern),
            ("3P6", stk.cage.M3L6),
            ("4P8", CGM4L8),
            ("4P82", M4L82),
            ("6P12", stk.cage.M6L12Cube),
            ("6P122", M6L122),
            ("8P16", stk.cage.EightPlusSixteen),
            ("8P162", M8L162),
        )

    elif study == "heteroleptic":
        topologies = (("8P16", stk.cage.EightPlusSixteen),)
    elif study == "li2023":
        topologies = (("4P82", M4L82), ("6P122", M6L122), ("8P162", M8L162))

    return topologies
