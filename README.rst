=============================
cosmo-archive-retrieve
=============================

Python library for extracting datasets from the COSMO archive as zarr.

Setup virtual environment

.. code-block:: console

    $ cd cosmo-archive-retrieve
    $ poetry install


Run tests

.. code-block:: console

    $ poetry run pytest

Generate documentation

.. code-block:: console

    $ poetry run sphinx-build doc doc/_build

Then open the index.html file generated in *cosmo-archive-retrieve/build/_build/*

Build wheels

.. code-block:: console

    $ poetry build
