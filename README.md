Collective Variable Package
===========================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linux/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/MacOS/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Windows/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linter/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinter)
[![codecov](https://codecov.io/gh/RedesignScience/cvpack/branch/main/graph/badge.svg)](https://codecov.io/gh/RedesignScience/cvpack/branch/main)
[![Documentation Status](https://readthedocs.org/projects/cvpack/badge/?style=flat)](https://readthedocs.org/projects/cvpack)

### Overview

In [OpenMM], collective variables (CV) involved in a [CustomCVForce] are nothing but OpenMM [Force]
objects. This CV Package provides several CVs of common use in molecular dynamics simulations, such
as:

* [Angle](https://cvpack-for-openmm.readthedocs.io/en/latest/api/Angle.html):
    The angle formed by three atoms.
* [Atomic Function](https://cvpack-for-openmm.readthedocs.io/en/latest/api/AtomicFunction.html):
    A user-defined function of the coordinates of a group of atoms.
* [Centroid Function](https://cvpack-for-openmm.readthedocs.io/en/latest/api/CentroidFunction.html)
    A user-defined function of the centroids of groups of atoms.
* [Distance](https://cvpack-for-openmm.readthedocs.io/en/latest/api/Distance.html):
    The distance between two atoms.
* [Helix angle content](https://cvpack-for-openmm.readthedocs.io/en/latest/api/HelixAngleContent.html):
    The alpha-helix angle content of a sequence of residues.
* [Helix H-bond content](https://cvpack-for-openmm.readthedocs.io/en/latest/api/HelixHBondContent.html):
    The alpha-helix hydrogen-bond content of a sequence of residues.
* [Helix RMSD content](https://cvpack-for-openmm.readthedocs.io/en/latest/api/HelixRMSDContent.html):
    The alpha-helix RMSD content of a sequence of residues
* [Helix torsion content](https://cvpack-for-openmm.readthedocs.io/en/latest/api/HelixTorsionContent.html):
    The alpha-helix Ramachandran content of a sequence of residues.
* [Number of contacts](https://cvpack-for-openmm.readthedocs.io/en/latest/api/NumberOfContacts.html):
    The number of contacts between two groups of atoms.
* [Radius of gyration](https://cvpack-for-openmm.readthedocs.io/en/latest/api/RadiusOfGyration.html):
    The radius of gyration of a group of atoms.
* [RMSD](https://cvpack-for-openmm.readthedocs.io/en/latest/api/RMSD.html):
    The RMSD of a group of atoms with respect to a reference structure.
* [Torsion](https://cvpack-for-openmm.readthedocs.io/en/latest/api/Torsion.html):
    The torsion angle formed by four atoms.
* [Torsion similarity](https://cvpack-for-openmm.readthedocs.io/en/latest/api/TorsionSimilarity.html):
    The degree of similarity between pairs of torsion angles.

### Copyright

Copyright (c) 2023, [C. Abreu](https://github.com/craabreu) & [Redesign Science](https://www.redesignscience.com)


#### Acknowledgements

Project based on the [CMS Cookiecutter] version 1.1.


[CMS Cookiecutter]: https://github.com/molssi/cookiecutter-cms
[CustomCVForce]:    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:            http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[OpenMM]:           https://openmm.org
