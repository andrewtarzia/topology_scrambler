"""analysis package."""

from scram._internal.topologies.building_block_enum import (
    get_potential_bb_dicts,
)
from scram._internal.topologies.custom_topology import CustomTopology
from scram._internal.topologies.enumeration import (
    HomolepticTopologyIterator,
    IHomolepticTopologyIterator,
    TopologyIterator,
)
from scram._internal.topologies.graphs import (
    CGM4L8,
    CGM12L24,
    M4L82,
    M6L122,
    M8L162,
    UnalignedM1L2,
    stoich_map,
)
from scram._internal.topologies.topology_code import Constructed, TopologyCode
from scram._internal.topologies.utilities import (
    get_graph_type,
    get_underyling_vertices,
    vmap_to_str,
)

__all__ = [
    "get_underyling_vertices",
    "get_potential_bb_dicts",
    "get_graph_type",
    "vmap_to_str",
    "stoich_map",
    "CustomTopology",
    "UnalignedM1L2",
    "CGM4L8",
    "CGM12L24",
    "M8L162",
    "M6L122",
    "M4L82",
    "TopologyCode",
    "Constructed",
    "TopologyIterator",
    "HomolepticTopologyIterator",
    "IHomolepticTopologyIterator",
]
