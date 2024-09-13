"""Topologies module."""

import logging
from collections import abc

import numpy as np
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def length_2_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two ditopic building blocks to be added as:
        0: larger -- constant
        1: smaller
        2: smaller2

    """
    return {
        # tstr:
        "2P4": (({0: [0, 1], 1: [], 2: []}), 4),
        "3P6": (({0: [0, 1, 2], 1: [], 2: []}), 6),
        "4P8": (({0: [0, 1, 2, 3], 1: [], 2: []}), 8),
        "4P82": (({0: [0, 1, 2, 3], 1: [], 2: []}), 8),
        "6P12": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 12),
        "6P122": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 12),
        "8P162": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 16),
        "8P16": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 16),
        "2P3": (({0: [0, 1], 1: [], 2: []}), 3),
        "4P6": (({0: [0, 1, 2, 3], 1: [], 2: []}), 6),
        "4P62": (({0: [0, 1, 2, 3], 1: [], 2: []}), 6),
        "6P9": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 9),
        "8P12": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 12),
    }[tstr]


def length_3_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two tritopic building blocks to be added as:
        0: larger
        1: larger2
        2: smaller -- constant

    """
    return {
        # tstr:
        "2P3": (({0: [], 1: [], 2: list(range(2, 5))}), 2),
        "4P6": (({0: [], 1: [], 2: list(range(4, 10))}), 4),
        "4P62": (({0: [], 1: [], 2: list(range(4, 10))}), 4),
        "6P9": (({0: [], 1: [], 2: list(range(6, 15))}), 6),
        "8P12": (({0: [], 1: [], 2: list(range(8, 20))}), 8),
    }[tstr]


def length_4_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two tetratopic building blocks to be added as:
        0: larger
        1: larger2
        2: smaller -- constant

    """
    return {
        # tstr:
        "2P4": (({0: [], 1: [], 2: list(range(2, 6))}), 2),
        "3P6": (({0: [], 1: [], 2: list(range(3, 9))}), 3),
        "4P8": (({0: [], 1: [], 2: list(range(4, 12))}), 4),
        "4P82": (({0: [], 1: [], 2: list(range(4, 12))}), 4),
        "6P12": (({0: [], 1: [], 2: list(range(6, 18))}), 6),
        "6P122": (({0: [], 1: [], 2: list(range(6, 18))}), 6),
        "8P162": (({0: [], 1: [], 2: list(range(8, 24))}), 8),
        "8P16": (({0: [], 1: [], 2: list(range(8, 24))}), 8),
    }[tstr]


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
    study: str
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


class M4L82(stk.cage.Cage):
    """Cage topology."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, [0, 0, np.sqrt(6) / 2]),
        stk.cage.NonLinearVertex(1, [-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(2, [1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(3, [0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6]),
    )

    paired_wall_1_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[0], _non_linears[1])
        )
        / 2
    )
    wall_1_shift = np.array((0.2, 0.2, 0))

    paired_wall_2_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[2], _non_linears[3])
        )
        / 2
    )
    wall_2_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            id=4,
            position=paired_wall_1_coord + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex.init_at_center(
            id=5,
            vertices=(_non_linears[0], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=6,
            vertices=(_non_linears[0], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=7,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=8,
            vertices=(_non_linears[1], _non_linears[3]),
        ),
        stk.cage.LinearVertex(
            id=9,
            position=paired_wall_2_coord + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=10,
            position=paired_wall_1_coord - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=11,
            position=paired_wall_2_coord - wall_2_shift,
            use_neighbor_placement=False,
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[10]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[11]),
    )


class CGM4L8(stk.cage.M4L8):
    """Cage topology."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, [2, 0, 0]),
        stk.cage.NonLinearVertex(1, [0, 2, 0]),
        stk.cage.NonLinearVertex(2, [-2, 0, 0]),
        stk.cage.NonLinearVertex(3, [0, -2, 0]),
        stk.cage.LinearVertex(4, [1, 1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(5, [1, 1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(6, [1, -1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(7, [1, -1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(8, [-1, -1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(9, [-1, -1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(10, [-1, 1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(11, [-1, 1, -0.5], use_neighbor_placement=False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
    )


class M6L122(stk.cage.Cage):
    """Cage topology."""

    _x = 2 * np.sqrt(3) / 4
    _y = 2
    _non_linears = (
        stk.cage.NonLinearVertex(0, [0, _x, 1]),
        stk.cage.NonLinearVertex(1, [_y / 2, -_x, 1]),
        stk.cage.NonLinearVertex(2, [-_y / 2, -_x, 1]),
        stk.cage.NonLinearVertex(3, [0, _x, -1]),
        stk.cage.NonLinearVertex(4, [_y / 2, -_x, -1]),
        stk.cage.NonLinearVertex(5, [-_y / 2, -_x, -1]),
    )

    paired_wall_1_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[0], _non_linears[1])
        )
        / 2
    )
    wall_1_shift = np.array((0.2, 0.2, 0))

    paired_wall_2_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[2], _non_linears[3])
        )
        / 2
    )
    wall_2_shift = np.array((0.2, 0.2, 0))

    paired_wall_3_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[4], _non_linears[5])
        )
        / 2
    )
    wall_3_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            6,
            np.array([0, _x, 0]) + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            7,
            np.array([0, _x, 0]) - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            8,
            np.array([_y / 2, -_x, 0]) + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            9,
            np.array([_y / 2, -_x, 0]) - wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            10,
            np.array([-_y / 2, -_x, 0]) + wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            11,
            np.array([-_y / 2, -_x, 0]) - wall_3_shift,
            use_neighbor_placement=False,
        ),
        #
        stk.cage.LinearVertex.init_at_center(
            id=12,
            vertices=(_non_linears[0], _non_linears[1]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=13,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=14,
            vertices=(_non_linears[2], _non_linears[0]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=15,
            vertices=(_non_linears[3], _non_linears[4]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=16,
            vertices=(_non_linears[4], _non_linears[5]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=17,
            vertices=(_non_linears[5], _non_linears[3]),
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[12]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[14]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[12]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[13]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[9]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[14]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[15]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[17]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[15]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[16]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[8]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[9]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[16]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[17]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[10]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[11]),
    )


class M8L162(stk.cage.Cage):
    """Cage topology."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, [1, 1, 1]),
        stk.cage.NonLinearVertex(1, [1, -1, 1]),
        stk.cage.NonLinearVertex(2, [-1, -1, 1]),
        stk.cage.NonLinearVertex(3, [-1, 1, 1]),
        stk.cage.NonLinearVertex(4, [1, 1, -1]),
        stk.cage.NonLinearVertex(5, [1, -1, -1]),
        stk.cage.NonLinearVertex(6, [-1, -1, -1]),
        stk.cage.NonLinearVertex(7, [-1, 1, -1]),
    )

    paired_wall_1_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[0], _non_linears[1])
        )
        / 2
    )
    wall_1_shift = np.array((0.2, 0.2, 0))

    paired_wall_2_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[2], _non_linears[3])
        )
        / 2
    )
    wall_2_shift = np.array((0.2, 0.2, 0))

    paired_wall_3_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[4], _non_linears[5])
        )
        / 2
    )
    wall_3_shift = np.array((0.2, 0.2, 0))

    paired_wall_4_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[6], _non_linears[7])
        )
        / 2
    )
    wall_4_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            8,
            np.array([1, 1, 0]) + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            9,
            np.array([1, 1, 0]) - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            10,
            np.array([1, -1, 0]) + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            11,
            np.array([1, -1, 0]) - wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            12,
            np.array([-1, -1, 0]) + wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            13,
            np.array([-1, -1, 0]) - wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            14,
            np.array([-1, 1, 0]) + wall_4_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            15,
            np.array([-1, 1, 0]) - wall_4_shift,
            use_neighbor_placement=False,
        ),
        #
        stk.cage.LinearVertex.init_at_center(
            id=16,
            vertices=(_non_linears[0], _non_linears[1]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=17,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=18,
            vertices=(_non_linears[2], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=19,
            vertices=(_non_linears[3], _non_linears[0]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=20,
            vertices=(_non_linears[4], _non_linears[5]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=21,
            vertices=(_non_linears[5], _non_linears[6]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=22,
            vertices=(_non_linears[6], _non_linears[7]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=23,
            vertices=(_non_linears[7], _non_linears[4]),
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[16]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[19]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[8]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[9]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[16]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[17]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[17]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[18]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[12]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[18]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[19]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[14]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[15]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[20]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[23]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[8]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[9]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[20]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[21]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[10]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[11]),
        stk.Edge(24, _vertex_prototypes[6], _vertex_prototypes[21]),
        stk.Edge(25, _vertex_prototypes[6], _vertex_prototypes[22]),
        stk.Edge(26, _vertex_prototypes[6], _vertex_prototypes[12]),
        stk.Edge(27, _vertex_prototypes[6], _vertex_prototypes[13]),
        stk.Edge(28, _vertex_prototypes[7], _vertex_prototypes[22]),
        stk.Edge(29, _vertex_prototypes[7], _vertex_prototypes[23]),
        stk.Edge(30, _vertex_prototypes[7], _vertex_prototypes[14]),
        stk.Edge(31, _vertex_prototypes[7], _vertex_prototypes[15]),
    )
