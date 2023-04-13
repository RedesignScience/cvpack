Collective Variable Package
===========================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linux/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/MacOS/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Windows/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linter/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinter)
[![codecov](https://codecov.io/gh/RedesignScience/cvpack/branch/main/graph/badge.svg)](https://codecov.io/gh/RedesignScience/cvpack/branch/main)
[![Documentation Status](https://readthedocs.org/projects/cvpack/badge/?style=flat)](https://cvpack.readthedocs.io/en/latest)

[![Conda cvpack version](https://img.shields.io/conda/v/redesign-science/cvpack.svg)](https://anaconda.org/redesign-science/cvpack)
[![Conda cvpack platforms](https://img.shields.io/conda/pn/redesign-science/cvpack.svg)](https://anaconda.org/redesign-science/cvpack)
[![Conda cvpack downloads](https://img.shields.io/conda/dn/redesign-science/cvpack.svg)](https://anaconda.org/redesign-science/cvpack)

[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/RedesignScience/cvpack/blob/main/LICENSE.md)
[![Twitter](https://badgen.net/badge/follow%20us/@RedesignScience?icon=twitter)](https://twitter.com/RedesignScience)

### Overview

Collective variables (CVs) are functions of the coordinates of a molecular system and provide a
means to project its conformational state onto a lower-dimensional space. By stimulating the
dynamics of a judiciously chosen set of CVs, one can obtain an enhanced sampling of the
configuration space, including regions that are otherwise difficult to access. The system's
free energy as a function of these CVs can be used to characterize the relative stability of
different states and to identify pathways connecting them.

CVPack is a Python package that provides pre-defined CVs for the powerful molecular dynamics engine
[OpenMM]. All these CVs are subclasses of OpenMM's [Force] class and, as such, can be directly added
to a [CustomCVForce] or used to define a [BiasVariable] for [Metadynamics], for instance.

### Collective Variables

The CVs implemented in CVPack are:

* [Angle](https://cvpack.readthedocs.io/en/latest/api/Angle.html):
    angle formed by three atoms
* [Atomic Function](https://cvpack.readthedocs.io/en/latest/api/AtomicFunction.html):
    a user-defined function of the coordinates of a group of atoms
* [Attraction Strength](https://cvpack.readthedocs.io/en/latest/api/AttractionStrength.html):
    strength of the attraction between two groups of atoms
* [Centroid Function](https://cvpack.readthedocs.io/en/latest/api/CentroidFunction.html):
    a user-defined function of the centroids of groups of atoms
* [Distance](https://cvpack.readthedocs.io/en/latest/api/Distance.html):
    distance between two atoms
* [Helix angle content](https://cvpack.readthedocs.io/en/latest/api/HelixAngleContent.html):
    alpha-helix angle content of a sequence of residues
* [Helix H-bond content](https://cvpack.readthedocs.io/en/latest/api/HelixHBondContent.html):
    alpha-helix hydrogen-bond content of a sequence of residues
* [Helix RMSD content](https://cvpack.readthedocs.io/en/latest/api/HelixRMSDContent.html):
    alpha-helix RMSD content of a sequence of residues
* [Helix torsion content](https://cvpack.readthedocs.io/en/latest/api/HelixTorsionContent.html):
    alpha-helix Ramachandran content of a sequence of residues
* [Number of contacts](https://cvpack.readthedocs.io/en/latest/api/NumberOfContacts.html):
    number of contacts between two groups of atoms
* [Radius of gyration](https://cvpack.readthedocs.io/en/latest/api/RadiusOfGyration.html):
    radius of gyration of a group of atoms
* [RMSD](https://cvpack.readthedocs.io/en/latest/api/RMSD.html):
    root-mean-square deviation of atoms with respect to a reference structure
* [Torsion](https://cvpack.readthedocs.io/en/latest/api/Torsion.html):
    torsion angle formed by four atoms
* [Torsion similarity](https://cvpack.readthedocs.io/en/latest/api/TorsionSimilarity.html):
    degree of similarity between pairs of torsion angles

### Installation and Usage

CVPack is available as a conda package on the
[redesign-science](https://anaconda.org/redesign-science/cvpack) channel. To install it, simply run:

```bash
    conda install -c conda-forge -c redesign-science cvpack
```

Or, if you prefer to use [mamba](https://mamba.readthedocs.io/en/latest) instead of conda:

```bash
    mamba install -c conda-forge -c redesign-science cvpack
```

To use CVPack in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import cvpack
```

### Documentation

The documentation for CVPack is available at [Read the Docs](https://cvpack.readthedocs.io/en/stable).

### Copyright

Copyright (c) 2023, [Redesign Science](https://www.redesignscience.com)


#### Acknowledgements

Project based on the [CMS Cookiecutter] version 1.1.

[BiasVariable]:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
[CMS Cookiecutter]:   https://github.com/molssi/cookiecutter-cms
[CollectiveVariable]: https://ufedmm.readthedocs.io/en/latest/pythonapi/ufedmm.html#ufedmm.ufedmm.CollectiveVariable
[CustomCVForce]:      https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:              https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[Metadynamics]:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.
[OpenMM]:             https://openmm.org
[UFED]:               https://ufedmm.readthedocs.io/en/latest/index.html
