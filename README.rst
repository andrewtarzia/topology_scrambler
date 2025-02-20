:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_

Overview
========

A series of case studies showing the first example of toy-model cgx-driven
cage structure prediction using graph enumeration.

This project is built on
`cgx <https://cgexplore.readthedocs.io/en/latest/>`_ and
`stk <https://stk.readthedocs.io/en/stable/>`_.

Installation
============

This code can be installed by cloning this repository and using `just <https://github.com/casey/just>`_:

.. code-block:: bash

  just dev

This will install the ``model_enumeration`` in a development state allowing
you to edit the source code, and will install the required dependencies
`openmm <https://openmm.org/>`_ and
`openmmtools <https://openmmtools.readthedocs.io/en/stable/gettingstarted.html>`_
using ``mamba``.

This will not install other dependencies available through conda:

.. code-block:: bash

  # for xtb
  mamba install xtb

CREST must be downloaded from `crest <https://crest-lab.github.io/crest-docs/page/documentation/keywords.html>`_

Important
---------

In each script, you will have to update the environment variables, namely
the paths where the data is written to and where some software is.

Projects scripts
================

From ``model_enumeration``, you get the following scripts available in the
command line.

Section 1: Angle hunter
-----------------------

``angle_hunter`` Does something.

``plot_li2023_maps`` Does something.

``plot_sudan2021_maps`` Does something.


Section 2: Torsion environment hunter
-------------------------------------

``environment_hunter`` Does something.


Section 3: Structure prediction
-------------------------------

``validation_mnl2n`` Does something.
``mgen_crest_analysis`` Does something.
``mgen_generation`` Does something.
``mgen_scan`` Does something.
``mgen_steric`` Does something.

Section 4: Graph and database analysis
--------------------------------------

``all_database_analysis`` Does something.
``analyse_graphs`` Does something.
``plot_geometrical`` Does something.


Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006.
