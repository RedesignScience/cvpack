Collective Variable Library
===========================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/craabreu/cvlib/workflows/Linux/badge.svg)](https://github.com/craabreu/cvlib/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/craabreu/cvlib/workflows/MacOS/badge.svg)](https://github.com/craabreu/cvlib/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/craabreu/cvlib/workflows/Windows/badge.svg)](https://github.com/craabreu/cvlib/actions?query=workflow%3AWindows)
[![GitHub Super-Linter](https://github.com/craabreu/cvlib/workflows/Linter/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![codecov](https://codecov.io/gh/craabreu/cvlib/branch/main/graph/badge.svg)](https://codecov.io/gh/craabreu/cvlib/branch/main)
[![Documentation Status](https://readthedocs.org/projects/cvlib-for-openmm/badge/?style=flat)](https://readthedocs.org/projects/cvlib-for-openmm)

Useful Collective Variables for OpenMM

### Overview

In [OpenMM](https://openmm.org), a collective variable (CV) involved in a CustomCVForce is nothing but an instance of some Force or CustomForce child class. CVlib provides with several predefined CVs, such as:

* Square radius of gyration of a group of atoms
* Number of contacts between two groups of atoms
* Different flavors of alpha-helix content measures, based on angles, dihedrals, and hydrogen bonds

### Copyright

Copyright (c) 2023, Charlles Abreu


#### Acknowledgements

Project based on the
[CMS Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
