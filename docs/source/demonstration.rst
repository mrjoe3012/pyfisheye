Demonstration Code
==================

The ``pyfisheye`` repository includes a demonstration script which downloads a fisheye image dataset and finds the optimal calibration parameters.

This script allows you to quickly gauge the performance of the library.

.. note::

   This demonstration script relies on a dataset located in this external repository:
   https://github.com/urbste/ImprovedOcamCalib

   This repository is not affiliated with the ``pyfisheye`` project. All rights and copyrights 
   to the dataset remain with the original authors.

Installation:

.. code-block:: console

    pip install pyfisheye

Downloading the demo data:

Visit the link above and download the GitHub repository as a ZIP file. Extract it into your working directory before running the demo.

Running the demo script:

.. note::
    You may need to change `python` to `python3` on some systems.

.. code-block:: console

    python -m pyfisheye.scripts.demo

You will be asked which of the datasets within the repository to use as wel as the grid
size to use when refining the distortion centre. Use the smaller grid size for a quick calibration
(under one minute), choose a larger grid size for a more accurate calibration which can take
several minutes.
