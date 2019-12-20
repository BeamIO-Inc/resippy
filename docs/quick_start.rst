ReSIPPy Quick Start
===================

The installation of ReSIPPy requires a few tricky dependencies:

* GDAL
* PDAL
* PyProj


Using conda
-----------

We like to use conda to manage our Python environments and dependencies.
To install Anaconda (or Miniconda), follow the instructions in the `Anaconda Docs <https://docs.anaconda.com/anaconda/install/>`_.
Once you have a working conda installation, clone this repo and create an environment for working with ReSIPPy:

.. code-block:: bash

    conda env create -f environment.yml


Then, activate the environment to work with ReSIPPy:

.. code-block:: bash

    conda activate resippy


Using pip
---------

ReSIPPy is also available on PyPi and can be installed using pip:

.. code-block:: bash

    pip install resippy


Installation using pip may require the user to download and install additional system level dependencies.


Documentation
-------------

This project uses Sphinx for documentation.
To build HTML documentation, navigate to the ``docs`` folder and run the following:

.. code-block:: bash

    make html


When completed, the HTML documentation will be available in ``docs/_build/html``.
