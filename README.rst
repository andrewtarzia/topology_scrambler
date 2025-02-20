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

.. important::
    In each script, you will have to update the environment variables, namely
    the paths where the data is written to and where some software is.

Projects scripts
================

* ``model_enumeration.all_database_analysis`` Does something.


* ``model_enumeration.analyse_graphs`` Does something.


* ``model_enumeration.angle_hunter`` Does something.


* ``model_enumeration.plot_geometrical`` Does something.


* ``model_enumeration.plot_li2023_maps`` Does something.


* ``model_enumeration.plot_sudan2021_maps`` Does something.


* ``model_enumeration.environment_hunter`` Does something.


* ``model_enumeration.mgen_crest_analysis`` Does something.


* ``model_enumeration.mgen_generation`` Does something.


* ``model_enumeration.mgen_scan`` Does something.


* ``model_enumeration.mgen_steric`` Does something.


* ``model_enumeration.validation_mnl2n`` Does something.


Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006.
