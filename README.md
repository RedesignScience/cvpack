Collective Variable Library
===========================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvlib/workflows/Linux/badge.svg)](https://github.com/RedesignScience/cvlib/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvlib/workflows/MacOS/badge.svg)](https://github.com/RedesignScience/cvlib/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvlib/workflows/Windows/badge.svg)](https://github.com/RedesignScience/cvlib/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvlib/workflows/Linter/badge.svg)](https://github.com/RedesignScience/cvlib/actions?query=workflow%3ALinter)
[![codecov](https://codecov.io/gh/RedesignScience/cvlib/branch/main/graph/badge.svg)](https://codecov.io/gh/RedesignScience/cvlib/branch/main)
[![Documentation Status](https://readthedocs.org/projects/cvlib-for-openmm/badge/?style=flat)](https://readthedocs.org/projects/cvlib-for-openmm)

### Overview

In [OpenMM], collective variables (CV) involved in a [CustomCVForce] are nothing but OpenMM [Force]
objects. This CV Library provides several CVs of common use in molecular dynamics simulations, such
as:

* [Distance](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Distance.html)
* [Angle](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Angle.html)
* [Torsion](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Torsion.html)
* [Radius of gyration](https://cvlib-for-openmm.readthedocs.io/en/latest/api/RadiusOfGyration.html)

### Copyright

Copyright (c) 2023, Charlles Abreu & Redesign Science


#### Acknowledgements

Project based on the [CMS Cookiecutter] version 1.1.


[CMS Cookiecutter]: https://github.com/molssi/cookiecutter-cms
[CustomCVForce]:    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:            http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[OpenMM]:           https://openmm.org
