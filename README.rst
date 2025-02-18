+-----------------------+
| **Documentation**     |
+=======================+
| |Documentation|       |
+-----------------------+

.. |Documentation| image:: https://img.shields.io/badge/api-reference-blue.svg
   :target: https://arpes.readthedocs.io/en/latest/

|test_status| |coverage| |docs_status| |conda| |pypi|


.. |docs_status| image:: https://readthedocs.org/projects/arpes/badge/?version=latest&style=flat
   :target: https://arpes.readthedocs.io/en/latest/
.. |coverage| image:: https://codecov.io/gh/chstan/arpes/branch/master/graph/badge.svg?token=mVrFuVRr3p
   :target: https://codecov.io/gh/chstan/arpes
.. |test_status| image:: https://github.com/chstan/arpes/actions/workflows/test.yml/badge.svg?branch=master
   :target: https://github.com/chstan/arpes/actions/workflows/test.yml
.. |pypi| image:: https://img.shields.io/pypi/v/arpes
   :target: https://pypi.org/project/arpes/
.. |conda| image:: https://img.shields.io/conda/v/arpes/arpes.svg
   :target: https://anaconda.org/arpes/arpes

PyARPES
=======

.. image:: docs/source/_static/video/intro-video.gif

========

PyARPES simplifies the analysis and collection of angle-resolved photoemission spectroscopy (ARPES) and emphasizes

* modern, best practices for data science
* support for a standard library of ARPES analysis tools mirroring those available in Igor Pro
* interactive and extensible analysis tools

It supports a variety of data formats from synchrotron and laser-ARPES sources including ARPES at the Advanced
Light Source (ALS), the data produced by Scienta Omicron GmbH's "SES Wrapper", data and experiment files from
Igor Pro, NeXuS files, and others.

To learn more about installing and using PyARPES in your analysis or data collection application,
visit `the documentation site`_.

PyARPES is currently developed by Conrad Stansbury of the Lanzara Group at the University of California, Berkeley.

Citing PyARPES
--------------

If you use PyARPES in your work, please support the development of scientific software by acknowledging its usefulness to you with a citation. The simplest way to do this is to cite the paper describing the package in SoftwareX


    @article{
        stansburypyarpes,
        title = {PyARPES: An analysis framework for multimodal angle-resolved photoemission spectroscopies},
        journal = {SoftwareX},
        volume = {11},
        pages = {100472},
        year = {2020},
        issn = {2352-7110},
        doi = {https://doi.org/10.1016/j.softx.2020.100472},
        url = {https://www.sciencedirect.com/science/article/pii/S2352711019301633},
        author = {Conrad Stansbury and Alessandra Lanzara},
        keywords = {ARPES, NanoARPES, Pump-probe ARPES, Photoemission, Python, Qt, Jupyter},
        abstract = {},
    }


Installation
============

PyARPES can be installed from source, or using either ``pip`` or ``conda`` into a Python 3.6 or 3.7 environment.
``conda`` is preferred as a package manager in order to facilitate installing the libraries for reading HDF and
NetCDF files.

Pip installation
----------------

::

   pip install arpes


Conda installation
------------------

PyARPES is distributed through the ``arpes`` Anaconda channel, but includes dependencies through ``conda-forge``.
Please make sure not to put conda-forge above the main channel priority, as this can cause issues with installing BLAS.
A minimal install looks like

::

   conda config --append channels conda-forge
   conda install -c arpes -c conda-forge arpes


Local installation from source
------------------------------

If you want to modify the source for PyARPES as you use it, you might prefer a local installation from source.
Details can be found on `the documentation site`_.


Suggested steps
---------------

1. Clone or duplicate the folder structure in the repository ``arpes-analysis-scaffold``,
   skipping the example folder and data if you like
2. Install and configure standard tools like Jupyter_ or Jupyter Lab. Notes on installing
   and configuring Jupyter based installations can be found in ``jupyter.md``
3. Explore the documentation and example notebooks at `the documentation site`_.

Contact
=======

Questions, difficulties, and suggestions can be directed to Conrad Stansbury (chstan@berkeley.edu)
or added to the repository as an issue. In the case of trouble, also check the `FAQ`_.

Copyright |copy| 2018-2019 by Conrad Stansbury, all rights reserved.

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

.. _Jupyter: https://jupyter.org/
.. _the documentation site: https://arpes.readthedocs.io/en/latest
.. _contributing: https://arpes.readthedocs.io/en/latest/contributing
.. _FAQ: https://arpes.readthedocs.io/en/latest/faq

