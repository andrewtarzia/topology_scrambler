:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_

Overview
========

A series of case studies showing the first example of toy-model `cgx``-driven
cage structure prediction using graph enumeration.

This project is built on
`cgx <https://cgexplore.readthedocs.io/en/latest/>`_ and
`stk <https://stk.readthedocs.io/en/stable/>`_.

Installation
============

Using ``conda``/``mamba`` (this will give the exact environment used in the paper):

.. code-block:: bash

  mamba env create -f environment.yml

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


Projects scripts
================

From ``model_enumeration``, you get the following scripts available in the
command line.

.. warning::

  In each script, you will have to update the environment variables, namely
  the paths where the data is written to and where some software is.

Each script should be self-contained and run and produce the output used in the
manuscript. Not all scripts were used in the final version, and will be
inspirations for recipes and examples in ``cgx``.

``mgen_crest_analysis``
    Does something.

``mgen_cs1``
    Does something.

``mgen_cs2``
    Does something.

``mgen_cs3``
    Does something.

``mgen_cs4``
    Does something.

``mgen_cs6``
    Does something.

``mgen_genetic``
    Does something.

``mgen_star``
    Reproduces the minimal model prediction of starship structures.

``validation_mnl2n``
    Does something.

``write_chemiscope``
    Produces the dataset visualisation using chemiscope of the output of
    the atomistic case study and genetic algorithm case study.

Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006.
