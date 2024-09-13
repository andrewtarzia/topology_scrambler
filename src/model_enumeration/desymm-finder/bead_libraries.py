"""Module containing bead libraries."""

import cgexplore

core_bead = cgexplore.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)

core_bead2 = cgexplore.molecular.CgBead(
    element_string="O",
    bead_class="o",
    bead_type="o",
    coordination=2,
)


arm_bead = cgexplore.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)


binder_bead = cgexplore.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)


tetragonal_bead = cgexplore.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)

trigonal_bead = cgexplore.molecular.CgBead(
    element_string="C",
    bead_class="n",
    bead_type="n",
    coordination=3,
)


tetragonal_bead2 = cgexplore.molecular.CgBead(
    element_string="Cr",
    bead_class="y",
    bead_type="y",
    coordination=4,
)

trigonal_bead2 = cgexplore.molecular.CgBead(
    element_string="Ge",
    bead_class="x",
    bead_type="x",
    coordination=3,
)
