"""Script to generate and optimise CG models."""

import logging
import stk
from copy import deepcopy
import numpy as np
from rdkit import RDLogger
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


class CustomTopology:
    def __init__(
        self,
        building_blocks,
        vertex_prototypes,
        edge_prototypes,
        vertex_alignments=None,
        vertex_positions: dict[int, np.ndarray] | None = None,
        reaction_factory=stk.GenericReactionFactory(),
        num_processes=1,
        optimizer=stk.NullOptimizer(),
        scale_multiplier: float = 1.0,
    ):
        class InternalTopology(stk.cage.Cage):
            _vertex_prototypes = vertex_prototypes
            _edge_prototypes = edge_prototypes

        self._topology_graph = InternalTopology(
            building_blocks=building_blocks,
            vertex_alignments=vertex_alignments,
            vertex_positions=vertex_positions,
            reaction_factory=reaction_factory,
            num_processes=num_processes,
            scale_multiplier=scale_multiplier,
            optimizer=optimizer,
        )

    def construct(self):
        return self._topology_graph.construct()


class UnalignedM1L2(stk.cage.Cage):
    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([0, 0, 0])),
        stk.cage.UnaligningVertex(1, np.array([-3, 0, 0]), False),
        stk.cage.UnaligningVertex(2, np.array([3, 0, 0]), False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[1]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[2]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[1]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[2]),
    )


class CGM4L8(stk.cage.M4L8):
    """New topology definition."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, [2, 0, 0]),
        stk.cage.NonLinearVertex(1, [0, 2, 0]),
        stk.cage.NonLinearVertex(2, [-2, 0, 0]),
        stk.cage.NonLinearVertex(3, [0, -2, 0]),
        stk.cage.LinearVertex(4, [1, 1, 0.5], False),
        stk.cage.LinearVertex(5, [1, 1, -0.5], False),
        stk.cage.LinearVertex(6, [1, -1, 0.5], False),
        stk.cage.LinearVertex(7, [1, -1, -0.5], False),
        stk.cage.LinearVertex(8, [-1, -1, 0.5], False),
        stk.cage.LinearVertex(9, [-1, -1, -0.5], False),
        stk.cage.LinearVertex(10, [-1, 1, 0.5], False),
        stk.cage.LinearVertex(11, [-1, 1, -0.5], False),
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


class CGM12L24(stk.cage.M12L24):
    """New topology definition."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, [1.25, 0, 0]),
        stk.cage.NonLinearVertex(1, [-1.25, 0, 0]),
        stk.cage.NonLinearVertex(2, [0, 1.25, 0]),
        stk.cage.NonLinearVertex(3, [0, -1.25, 0]),
        stk.cage.NonLinearVertex(4, [0.625, 0.625, 0.88]),
        stk.cage.NonLinearVertex(5, [0.625, -0.625, 0.88]),
        stk.cage.NonLinearVertex(6, [-0.625, 0.625, 0.88]),
        stk.cage.NonLinearVertex(7, [-0.625, -0.625, 0.88]),
        stk.cage.NonLinearVertex(8, [0.625, 0.625, -0.88]),
        stk.cage.NonLinearVertex(9, [0.625, -0.625, -0.88]),
        stk.cage.NonLinearVertex(10, [-0.625, 0.625, -0.88]),
        stk.cage.NonLinearVertex(11, [-0.625, -0.625, -0.88]),
        stk.cage.LinearVertex(12, [0.9, 0.31, 0.31], False),
        stk.cage.LinearVertex(13, [0.9, 0.31, -0.31], False),
        stk.cage.LinearVertex(14, [0.9, -0.31, 0.31], False),
        stk.cage.LinearVertex(15, [0.9, -0.31, -0.31], False),
        stk.cage.LinearVertex(16, [-0.9, 0.31, 0.31], False),
        stk.cage.LinearVertex(17, [-0.9, 0.31, -0.31], False),
        stk.cage.LinearVertex(18, [-0.9, -0.31, 0.31], False),
        stk.cage.LinearVertex(19, [-0.9, -0.31, -0.31], False),
        stk.cage.LinearVertex(20, [0.31, 0.9, 0.31], False),
        stk.cage.LinearVertex(21, [0.31, 0.9, -0.31], False),
        stk.cage.LinearVertex(22, [-0.31, 0.9, 0.31], False),
        stk.cage.LinearVertex(23, [-0.31, 0.9, -0.31], False),
        stk.cage.LinearVertex(24, [0.31, -0.9, 0.31], False),
        stk.cage.LinearVertex(25, [0.31, -0.9, -0.31], False),
        stk.cage.LinearVertex(26, [-0.31, -0.9, 0.31], False),
        stk.cage.LinearVertex(27, [-0.31, -0.9, -0.31], False),
        stk.cage.LinearVertex(28, [0.58, 0, 0.82], False),
        stk.cage.LinearVertex(29, [-0.58, 0, 0.82], False),
        stk.cage.LinearVertex(30, [0, 0.58, 0.82], False),
        stk.cage.LinearVertex(31, [0, -0.58, 0.82], False),
        stk.cage.LinearVertex(32, [0.58, 0, -0.82], False),
        stk.cage.LinearVertex(33, [-0.58, 0, -0.82], False),
        stk.cage.LinearVertex(34, [0, 0.58, -0.82], False),
        stk.cage.LinearVertex(35, [0, -0.58, -0.82], False),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[12]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[13]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[14]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[15]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[16]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[17]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[18]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[19]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[20]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[21]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[22]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[23]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[24]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[25]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[26]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[27]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[28]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[30]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[12]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[20]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[14]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[24]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[28]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[31]),
        stk.Edge(24, _vertex_prototypes[6], _vertex_prototypes[16]),
        stk.Edge(25, _vertex_prototypes[6], _vertex_prototypes[29]),
        stk.Edge(26, _vertex_prototypes[6], _vertex_prototypes[30]),
        stk.Edge(27, _vertex_prototypes[6], _vertex_prototypes[22]),
        stk.Edge(28, _vertex_prototypes[7], _vertex_prototypes[18]),
        stk.Edge(29, _vertex_prototypes[7], _vertex_prototypes[26]),
        stk.Edge(30, _vertex_prototypes[7], _vertex_prototypes[31]),
        stk.Edge(31, _vertex_prototypes[7], _vertex_prototypes[29]),
        stk.Edge(32, _vertex_prototypes[8], _vertex_prototypes[13]),
        stk.Edge(33, _vertex_prototypes[8], _vertex_prototypes[32]),
        stk.Edge(34, _vertex_prototypes[8], _vertex_prototypes[34]),
        stk.Edge(35, _vertex_prototypes[8], _vertex_prototypes[21]),
        stk.Edge(36, _vertex_prototypes[9], _vertex_prototypes[15]),
        stk.Edge(37, _vertex_prototypes[9], _vertex_prototypes[32]),
        stk.Edge(38, _vertex_prototypes[9], _vertex_prototypes[35]),
        stk.Edge(39, _vertex_prototypes[9], _vertex_prototypes[25]),
        stk.Edge(40, _vertex_prototypes[10], _vertex_prototypes[17]),
        stk.Edge(41, _vertex_prototypes[10], _vertex_prototypes[23]),
        stk.Edge(42, _vertex_prototypes[10], _vertex_prototypes[34]),
        stk.Edge(43, _vertex_prototypes[10], _vertex_prototypes[33]),
        stk.Edge(44, _vertex_prototypes[11], _vertex_prototypes[19]),
        stk.Edge(45, _vertex_prototypes[11], _vertex_prototypes[33]),
        stk.Edge(46, _vertex_prototypes[11], _vertex_prototypes[27]),
        stk.Edge(47, _vertex_prototypes[11], _vertex_prototypes[35]),
    )

    def _get_scale(self, building_block_vertices, scale_multiplier):  # noqa: ARG002
        return 10

    def get_vertex_alignments(self):
        return self._vertex_alignments


@dataclass
class Constructed:
    constructed_molecule: stk.ConstructedMolecule
    idx: int


class TopologyIterator:
    def __init__(
        self,
        tetra_bb,
        converging_bb,
        diverging_bb,
        multiplier,
        stoichiometry,
    ):
        if stoichiometry == (1, 1, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    converging_bb: (1,),
                    diverging_bb: (2,),
                }
                self._underlying_topology = UnalignedM1L2
                self._scale_multiplier = 2

            elif multiplier == 2:
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    converging_bb: (2, 3),
                    diverging_bb: (4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2

            elif multiplier == 3:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5),
                    diverging_bb: (6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2

            elif multiplier == 4:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    converging_bb: (4, 5, 6, 7),
                    diverging_bb: (8, 9, 10, 11),
                }
                self._underlying_topology = CGM4L8
                self._scale_multiplier = 2

        if stoichiometry == (4, 2, 3):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2

            elif multiplier == 2:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3, 4, 5),
                    converging_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                    diverging_bb: (14, 15, 16, 17),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5

            elif multiplier == 4:
                self._building_blocks = {
                    tetra_bb: range(0, 12),
                    converging_bb: range(12, 28),
                    diverging_bb: range(28, 36),
                }
                self._underlying_topology = CGM12L24
                self._scale_multiplier = 5

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement,
            )
            for i in self._underlying_topology._vertex_prototypes
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes
        )
        self._num_scrambles = 200
        self._num_coordinates = 5
        self._skip_initial = True

    def get_num_building_blocks(self):
        return len(self._init_vertex_prototypes)

    def get_constructed_molecules(self):
        if not self._skip_initial:
            try:
                constructed = stk.ConstructedMolecule(
                    self._underlying_topology(self._building_blocks)
                )
                yield Constructed(constructed_molecule=constructed, idx=0)
            except ValueError:
                pass

        vertex_connections = {}
        for edge in self._init_edge_prototypes:
            if edge.get_vertex1_id() not in vertex_connections:
                vertex_connections[edge.get_vertex1_id()] = 0
            vertex_connections[edge.get_vertex1_id()] += 1

            if edge.get_vertex2_id() not in vertex_connections:
                vertex_connections[edge.get_vertex2_id()] = 0
            vertex_connections[edge.get_vertex2_id()] += 1

        type1 = [i for i in vertex_connections if vertex_connections[i] == 4]
        type2 = [i for i in vertex_connections if vertex_connections[i] == 2]

        rng = np.random.default_rng(seed=100)

        combinations_tested = set()

        count = 0
        for step in range(self._num_scrambles):
            # Scramble the edges.
            remaining_connections = deepcopy(vertex_connections)
            available_type1s = deepcopy(type1)
            available_type2s = deepcopy(type2)

            new_edges = []
            combination = []
            for ie in range(len(self._init_edge_prototypes)):
                try:
                    vertex1 = rng.choice(available_type1s)
                    vertex2 = rng.choice(available_type2s)
                except ValueError:
                    if len(remaining_connections) == 1:
                        vertex1 = list(remaining_connections.keys())[0]
                        vertex2 = list(remaining_connections.keys())[0]

                new_edge = stk.Edge(
                    id=len(new_edges),
                    vertex1=self._vertices[vertex1],
                    vertex2=self._vertices[vertex2],
                )
                new_edges.append(new_edge)

                remaining_connections[vertex1] += -1
                remaining_connections[vertex2] += -1

                remaining_connections = {
                    i: remaining_connections[i]
                    for i in remaining_connections
                    if remaining_connections[i] != 0
                }

                available_type1s = [i for i in type1 if i in remaining_connections]
                available_type2s = [i for i in type2 if i in remaining_connections]
                combination.append(tuple(sorted((vertex1, vertex2))))

            # If you broke early, do not try to build.
            if len(new_edges) != len(self._edges):
                continue

            if tuple(sorted(combination)) in combinations_tested:
                continue

            combinations_tested.add(tuple(sorted(combination)))

            count += 1
            try:
                # Try with aligning vertices.
                constructed = stk.ConstructedMolecule(
                    CustomTopology(
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._init_vertex_prototypes,
                        edge_prototypes=new_edges,
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                yield Constructed(constructed_molecule=constructed, idx=count)
            except ValueError:
                # Try with unaligning.
                try:
                    constructed = stk.ConstructedMolecule(
                        CustomTopology(
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._vertices,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=None,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield Constructed(constructed_molecule=constructed, idx=count)
                except ValueError:
                    pass

            # Scramble the vertex positions.
            for step2 in range(self._num_coordinates - 1):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10 for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        CustomTopology(
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._init_vertex_prototypes,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=new_vertex_positions,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield Constructed(constructed_molecule=constructed, idx=count)
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            CustomTopology(
                                building_blocks=self._building_blocks,
                                vertex_prototypes=self._vertices,
                                edge_prototypes=new_edges,
                                vertex_alignments=None,
                                vertex_positions=new_vertex_positions,
                                scale_multiplier=self._scale_multiplier,
                            )
                        )
                        yield Constructed(constructed_molecule=constructed, idx=count)
                    except ValueError:
                        pass


class HomolepticTopologyIterator(TopologyIterator):
    def __init__(
        self,
        tetra_bb,
        ditopic_bb,
        multiplier,
        stoichiometry,
    ):
        if stoichiometry == (2, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    ditopic_bb: (1, 2),
                }
                self._underlying_topology = UnalignedM1L2
                self._scale_multiplier = 2
                self._num_scrambles = 10

            if multiplier == 2:
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    ditopic_bb: (2, 3, 4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2
                self._num_scrambles = 40

            if multiplier == 3:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    ditopic_bb: (3, 4, 5, 6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._num_scrambles = 60

            if multiplier == 4:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    ditopic_bb: (4, 5, 6, 7, 8, 9, 10, 11),
                }
                self._underlying_topology = CGM4L8
                self._scale_multiplier = 2
                self._num_scrambles = 60

            if multiplier == 6:
                self._building_blocks = {
                    tetra_bb: range(0, 6),
                    ditopic_bb: range(6, 18),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5
                self._num_scrambles = 200

            if multiplier == 8:
                self._building_blocks = {
                    tetra_bb: range(0, 8),
                    ditopic_bb: range(8, 24),
                }
                self._underlying_topology = stk.cage.EightPlusSixteen
                self._scale_multiplier = 5
                self._num_scrambles = 150

            if multiplier == 10:
                self._building_blocks = {
                    tetra_bb: range(0, 10),
                    ditopic_bb: range(10, 30),
                }
                self._underlying_topology = stk.cage.TenPlusTwenty
                self._scale_multiplier = 5
                self._num_scrambles = 150

            if multiplier == 12:
                self._building_blocks = {
                    tetra_bb: range(0, 12),
                    ditopic_bb: range(12, 36),
                }
                self._underlying_topology = CGM12L24
                self._scale_multiplier = 5
                self._num_scrambles = 150

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement,
            )
            for i in self._underlying_topology._vertex_prototypes
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes
        )
        self._num_coordinates = 5
        self._skip_initial = True
