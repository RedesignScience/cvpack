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

* [Angle](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Angle.html):
    The angle formed by three atoms.
* [Atomic Function](https://cvlib-for-openmm.readthedocs.io/en/latest/api/AtomicFunction.html):
    A user-defined function of the coordinates of a group of atoms.
* [Centroid Function](https://cvlib-for-openmm.readthedocs.io/en/latest/api/CentroidFunction.html)
    A user-defined function of the centroids of groups of atoms.
* [Distance](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Distance.html):
    The distance between two atoms.
* [Helix angle content](https://cvlib-for-openmm.readthedocs.io/en/latest/api/HelixAngleContent.html):
    The fractional alpha-helix angle content of a sequence of residues.
* [Helix H-bond content](https://cvlib-for-openmm.readthedocs.io/en/latest/api/HelixHBondContent.html):
    The fractional alpha-helix hydrogen-bond content of a sequence of residues.
* [Helix torsion content](https://cvlib-for-openmm.readthedocs.io/en/latest/api/HelixTorsionContent.html):
    The fractional alpha-helix Ramachandran content of a sequence of residues.
* [Number of contacts](https://cvlib-for-openmm.readthedocs.io/en/latest/api/NumberOfContacts.html):
    The number of contacts between two groups of atoms.
* [Radius of gyration](https://cvlib-for-openmm.readthedocs.io/en/latest/api/RadiusOfGyration.html):
    The radius of gyration of a group of atoms.
* [RMSD](https://cvlib-for-openmm.readthedocs.io/en/latest/api/RMSD.html):
    The RMSD of a group of atoms with respect to a reference structure.
* [Torsion](https://cvlib-for-openmm.readthedocs.io/en/latest/api/Torsion.html):
    The torsion angle formed by four atoms.
* [Torsion similarity](https://cvlib-for-openmm.readthedocs.io/en/latest/api/TorsionSimilarity.html):
    The degree of similarity between pairs of torsion angles.

### Copyright

Copyright (c) 2023, Charlles Abreu & Redesign Science


#### Acknowledgements

Project based on the [CMS Cookiecutter] version 1.1.


[CMS Cookiecutter]: https://github.com/molssi/cookiecutter-cms
[CustomCVForce]:    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:            http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[OpenMM]:           https://openmm.org
