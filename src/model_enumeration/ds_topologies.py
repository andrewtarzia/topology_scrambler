"""Topologies module."""

import logging
from collections import abc

import stk

import scram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def cage_topology_options(
    study: str,
) -> abc.Sequence[tuple[str, abc.Callable]]:
    """Topology options."""
    match study:
        case "homoleptic_2p4":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", scram.topologies.CGM4L8),
                ("4P82", scram.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
                ("6P122", scram.topologies.M6L122),
                ("8P16", stk.cage.EightPlusSixteen),
                ("8P162", scram.topologies.M8L162),
            )

        case "shortened_homoleptic_2p4":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", scram.topologies.CGM4L8),
                ("4P82", scram.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
            )

        case "repr_6P122":
            topologies = (("6P122", scram.topologies.M6L122),)

        case "homoleptic_2p3":
            topologies = (
                ("2P3", stk.cage.TwoPlusThree),
                ("4P6", stk.cage.FourPlusSix),
                ("4P62", stk.cage.FourPlusSix2),
                ("6P9", stk.cage.SixPlusNine),
                ("8P12", stk.cage.EightPlusTwelve),
            )

        case "homoleptic_3p4":
            topologies = (("6P8", stk.cage.SixPlusEight),)

        case "homoleptic_2p3_3x":
            topologies = (
                ("2P3", stk.cage.TwoPlusThree),
                ("4P6", stk.cage.FourPlusSix),
                ("4P62", stk.cage.FourPlusSix2),
                ("6P9", stk.cage.SixPlusNine),
                ("8P12", stk.cage.EightPlusTwelve),
            )

        case "homoleptic_2p4_4x":
            topologies = (
                ("2P4", stk.cage.M2L4Lantern),
                ("3P6", stk.cage.M3L6),
                ("4P8", scram.topologies.CGM4L8),
                ("4P82", scram.topologies.M4L82),
                ("6P12", stk.cage.M6L12Cube),
                ("6P122", scram.topologies.M6L122),
                ("8P16", stk.cage.EightPlusSixteen),
                ("8P162", scram.topologies.M8L162),
            )

        case "heteroleptic":
            topologies = (("8P16", stk.cage.EightPlusSixteen),)

        case "li2023":
            topologies = (
                ("4P82", scram.topologies.M4L82),
                ("6P122", scram.topologies.M6L122),
                ("8P162", scram.topologies.M8L162),
            )

        case _:
            msg = f"topology option {study} not defined"
            raise RuntimeError(msg)

    return topologies
