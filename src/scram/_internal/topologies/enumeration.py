"""Script to generate and optimise CG models."""

import logging
from collections import abc
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import stk
from rdkit import RDLogger

from .graphs import CGM4L8, CGM12L24, UnalignedM1L2
from .utilities import vmap_to_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


class CustomTopology:
    """Container for a custom topology graph."""

    def __init__(  # noqa: PLR0913
        self,
        building_blocks: (
            abc.Iterable[stk.BuildingBlock]
            | dict[stk.BuildingBlock, tuple[int, ...]]
        ),
        vertex_prototypes: list[stk.Vertex],
        edge_prototypes: list[stk.Edge],
        vertex_alignments: dict[int, int] | None = None,
        vertex_positions: dict[int, np.ndarray] | None = None,
        reaction_factory: stk.ReactionFactory = stk.GenericReactionFactory(),  # noqa: B008
        num_processes: int = 1,
        optimizer: stk.Optimizer = stk.NullOptimizer(),  # noqa: B008
        scale_multiplier: float = 1.0,
    ) -> None:
        """Initialize."""

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

    def construct(self) -> stk.ConstructionResult:
        """Construct topology."""
        return self._topology_graph.construct()


@dataclass
class TopologyCode:
    """Naming convention for topology graphs."""

    vertex_map: abc.Sequence[tuple[int, int]]
    as_string: str


@dataclass
class Constructed:
    """Container for constructed molecule and topology graph."""

    constructed_molecule: stk.ConstructedMolecule
    idx: int | None
    topology_code: TopologyCode


class TopologyIterator:
    """Iterate over topology graphs."""

    def __init__(
        self,
        tetra_bb: stk.BuildingBlock,
        converging_bb: stk.BuildingBlock,
        diverging_bb: stk.BuildingBlock,
        multiplier: int,
        stoichiometry: tuple[int, int, int],
    ) -> None:
        """Initialize."""
        if stoichiometry == (1, 1, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    converging_bb: (1,),
                    diverging_bb: (2,),
                }
                self._underlying_topology = UnalignedM1L2
                self._scale_multiplier = 2
                self._skip_initial = True

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    converging_bb: (2, 3),
                    diverging_bb: (4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 3:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5),
                    diverging_bb: (6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    converging_bb: (4, 5, 6, 7),
                    diverging_bb: (8, 9, 10, 11),
                }
                self._underlying_topology = CGM4L8
                self._scale_multiplier = 2
                self._skip_initial = False

        if stoichiometry == (4, 2, 3):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3, 4, 5),
                    converging_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                    diverging_bb: (14, 15, 16, 17),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(12),
                    converging_bb: range(12, 28),
                    diverging_bb: range(28, 36),
                }
                self._underlying_topology = CGM12L24
                self._scale_multiplier = 5
                self._skip_initial = False

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement,
            )
            for i in self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._num_scrambles = 200
        self._num_mashes = 2

        self._define_underlying()
        self._beta = 10

    def _define_underlying(self) -> None:
        self._vertex_connections = {}
        for edge in self._init_edge_prototypes:
            if edge.get_vertex1_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex1_id()] = 0
            self._vertex_connections[edge.get_vertex1_id()] += 1

            if edge.get_vertex2_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex2_id()] = 0
            self._vertex_connections[edge.get_vertex2_id()] += 1

        self._type1 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 4  # noqa: PLR2004
        ]
        self._type2 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 2  # noqa: PLR2004
        ]

        combination = [
            tuple(sorted((i.get_vertex1_id(), i.get_vertex2_id())))
            for i in self._init_edge_prototypes
        ]
        self._initial_topology_code = TopologyCode(
            vertex_map=combination,
            as_string=vmap_to_str(combination),
        )

    def get_num_building_blocks(self) -> int:
        """Get number of building blocks."""
        return len(self._init_vertex_prototypes)

    def get_beta(self) -> float:
        """Get beta for MC algorithm."""
        return self._beta

    def get_num_scrambles(self) -> int:
        """Get num. scrambles algorithm."""
        return self._num_scrambles

    def get_num_mashes(self) -> int:
        """Get num. mashes algorithm."""
        return self._num_mashes

    def get_constructed_molecules(self) -> abc.Generator[Constructed]:  # noqa: C901, PLR0912, PLR0915
        """Get constructed molecules from iteration."""
        combinations_tested = set()
        rng = np.random.default_rng(seed=100)
        count = 0

        if not self._skip_initial:
            try:
                constructed = stk.ConstructedMolecule(
                    self._underlying_topology(
                        building_blocks=self._building_blocks,
                        vertex_positions=None,
                    )
                )

                yield Constructed(
                    constructed_molecule=constructed,
                    idx=0,
                    topology_code=self._initial_topology_code,
                )
            except ValueError:
                pass
            combinations_tested.add(self._initial_topology_code.as_string)

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        self._underlying_topology(
                            building_blocks=self._building_blocks,
                            vertex_positions=None,
                        )
                    )
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=self._initial_topology_code,
                    )
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            self._underlying_topology(
                                building_blocks=self._building_blocks,
                                vertex_positions=None,
                            )
                        )
                        yield Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=self._initial_topology_code,
                        )
                    except ValueError:
                        pass

        for _ in range(self._num_scrambles):
            # Scramble the edges.
            remaining_connections = deepcopy(self._vertex_connections)
            available_type1s = deepcopy(self._type1)
            available_type2s = deepcopy(self._type2)

            new_edges = []
            combination = []
            for _ in range(len(self._init_edge_prototypes)):
                try:
                    vertex1 = rng.choice(available_type1s)
                    vertex2 = rng.choice(available_type2s)
                except ValueError:
                    if len(remaining_connections) == 1:
                        vertex1 = next(iter(remaining_connections.keys()))
                        vertex2 = next(iter(remaining_connections.keys()))

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

                available_type1s = [
                    i for i in self._type1 if i in remaining_connections
                ]
                available_type2s = [
                    i for i in self._type2 if i in remaining_connections
                ]
                combination.append(tuple(sorted((vertex1, vertex2))))

            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            # If you broke early, do not try to build.
            if len(new_edges) != len(self._edges):
                continue

            if topology_code.as_string in combinations_tested:
                continue

            combinations_tested.add(topology_code.as_string)

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
                yield Constructed(
                    constructed_molecule=constructed,
                    idx=count,
                    topology_code=topology_code,
                )
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
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
                except ValueError:
                    pass

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
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
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
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
                        yield Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=topology_code,
                        )
                    except ValueError:
                        pass

    def _get_random_topology_code(
        self, generator: np.random.Generator
    ) -> TopologyCode:
        remaining_connections = deepcopy(self._vertex_connections)
        available_type1s = deepcopy(self._type1)
        available_type2s = deepcopy(self._type2)

        vertex_map = []
        for _ in range(len(self._init_edge_prototypes)):
            try:
                vertex1 = generator.choice(available_type1s)
                vertex2 = generator.choice(available_type2s)
            except ValueError:
                if len(remaining_connections) == 1:
                    vertex1 = next(iter(remaining_connections.keys()))
                    vertex2 = next(iter(remaining_connections.keys()))

            vertex_map.append(tuple(sorted((vertex1, vertex2))))

            remaining_connections[vertex1] += -1
            remaining_connections[vertex2] += -1
            remaining_connections = {
                i: remaining_connections[i]
                for i in remaining_connections
                if remaining_connections[i] != 0
            }
            available_type1s = [
                i for i in self._type1 if i in remaining_connections
            ]
            available_type2s = [
                i for i in self._type2 if i in remaining_connections
            ]

        return TopologyCode(
            vertex_map=vertex_map, as_string=vmap_to_str(vertex_map)
        )

    def _shuffle_topology_code(
        self,
        topology_code: TopologyCode,
        generator: np.random.Generator,
    ) -> TopologyCode:
        old_vertex_map = topology_code.vertex_map

        size = (
            generator.integers(
                low=1, high=int(len(old_vertex_map) / 2), size=1
            )
            * 2
        )

        swaps = list(
            generator.choice(
                range(len(old_vertex_map)),
                size=int(size[0]),
                replace=False,
            )
        )

        new_vertex_map = []
        already_done = set()
        for vmap_idx in range(len(old_vertex_map)):
            if vmap_idx in already_done:
                continue
            if vmap_idx in swaps:
                possible_ids = [i for i in swaps if i != vmap_idx]
                other_idx = generator.choice(possible_ids, size=1)[0]

                # Swap connections.
                old1 = old_vertex_map[vmap_idx]
                old2 = old_vertex_map[other_idx]

                new1 = (old1[0], old2[1])
                new2 = (old2[0], old1[1])

                new_vertex_map.append(new1)
                new_vertex_map.append(new2)
                swaps = [i for i in swaps if i not in (vmap_idx, other_idx)]

                already_done.add(other_idx)
            else:
                new_vertex_map.append(old_vertex_map[vmap_idx])

        return TopologyCode(
            vertex_map=new_vertex_map, as_string=vmap_to_str(new_vertex_map)
        )

    def get_topology(
        self,
        input_topology_code: TopologyCode | None,
        generator: np.random.Generator,
    ) -> Constructed | None:
        """Get a topology."""
        if input_topology_code is None:
            topology_code = self._get_random_topology_code(generator=generator)
        else:
            topology_code = self._shuffle_topology_code(
                topology_code=input_topology_code,
                generator=generator,
            )

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                CustomTopology(
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=None,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    CustomTopology(
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None

    def get_mashed_topology(
        self,
        topology_code: TopologyCode,
        generator: np.random.Generator,
    ) -> Constructed | None:
        """Get a mashed topology, where vertex coordinates are changed."""
        coordinates = generator.random(size=(len(self._vertices), 3))
        new_vertex_positions = {
            j: coordinates[j] * 10 for j, i in enumerate(self._vertices)
        }

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                CustomTopology(
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=new_vertex_positions,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    CustomTopology(
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=new_vertex_positions,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None


class HomolepticTopologyIterator(TopologyIterator):
    """Iterate over topology graphs."""

    def __init__(  # noqa: PLR0915
        self,
        tetra_bb: stk.BuildingBlock,
        ditopic_bb: stk.BuildingBlock,
        multiplier: int,
        stoichiometry: tuple[int, int],
    ) -> None:
        """Initialize."""
        if stoichiometry == (2, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    ditopic_bb: (1, 2),
                }
                self._underlying_topology = UnalignedM1L2
                self._scale_multiplier = 2
                self._num_scrambles = 10
                self._num_mashes = 2
                self._beta = 10

            if multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    ditopic_bb: (2, 3, 4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2
                self._num_scrambles = 40
                self._num_mashes = 1
                self._beta = 10

            if multiplier == 3:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    ditopic_bb: (3, 4, 5, 6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._num_scrambles = 100
                self._num_mashes = 1
                self._beta = 10

            if multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    ditopic_bb: (4, 5, 6, 7, 8, 9, 10, 11),
                }
                self._underlying_topology = CGM4L8
                self._scale_multiplier = 2
                self._num_scrambles = 100
                self._num_mashes = 1
                self._beta = 10

            if multiplier == 6:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(6),
                    ditopic_bb: range(6, 18),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5
                self._num_scrambles = 500
                self._num_mashes = 1
                self._beta = 10

            if multiplier == 8:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(8),
                    ditopic_bb: range(8, 24),
                }
                self._underlying_topology = stk.cage.EightPlusSixteen
                self._scale_multiplier = 5
                self._num_scrambles = 500
                self._num_mashes = 1
                self._beta = 1

            if multiplier == 10:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(10),
                    ditopic_bb: range(10, 30),
                }
                self._underlying_topology = stk.cage.TenPlusTwenty
                self._scale_multiplier = 5
                self._num_scrambles = 500
                self._num_mashes = 1
                self._beta = 1

            if multiplier == 12:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(12),
                    ditopic_bb: range(12, 36),
                }
                self._underlying_topology = CGM12L24
                self._scale_multiplier = 5
                self._num_scrambles = 500
                self._num_mashes = 1
                self._beta = 1

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement,
            )
            for i in self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes  # noqa: SLF001
        )

        self._skip_initial = True
        self._define_underlying()
