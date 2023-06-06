Collective Variable Package
===========================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linux/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/MacOS/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Windows/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linter/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinter)
[![Documentation Status](https://github.com/RedesignScience/cvpack/workflows/Docs/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ADocs)
[![Coverage Report](https://redesignscience.github.io/cvpack/coverage/coverage.svg)](https://redesignscience.github.io/cvpack/coverage/coverage)

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

The CVs implemented in CVPack are listed in the table below.

| Collective Variable     | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| [Angle]                 | angle formed by three atoms                                      |
| [Atomic Function]       | a user-defined function of the coordinates of a group of atoms   |
| [Attraction Strength]   | strength of the attraction between two groups of atoms           |
| [Centroid Function]     | a user-defined function of the centroids of groups of atoms      |
| [Distance]              | distance between two atoms                                       |
| [Helix angle content]   | alpha-helix angle content of a sequence of residues              |
| [Helix H-bond content]  | alpha-helix hydrogen-bond content of a sequence of residues      |
| [Helix RMSD content]    | alpha-helix RMSD content of a sequence of residues               |
| [Helix torsion content] | alpha-helix Ramachandran content of a sequence of residues       |
| [Number of contacts]    | number of contacts between two groups of atoms                   |
| [Radius of gyration]    | radius of gyration of a group of atoms                           |
| [Rg squared]            | square of the radius of gyration of a group of atoms             |
| [RMSD]                  | root-mean-square deviation with respect to a reference structure |
| [Torsion]               | torsion angle formed by four atoms                               |
| [Torsion similarity]    | degree of similarity between pairs of torsion angles             |

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
[Metadynamics]:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
[OpenMM]:             https://openmm.org
[UFED]:               https://ufedmm.readthedocs.io/en/latest/index.html

[Angle]:                 https://cvpack.readthedocs.io/en/latest/api/Angle.html
[Atomic Function]:       https://cvpack.readthedocs.io/en/latest/api/AtomicFunction.html
[Attraction Strength]:   https://cvpack.readthedocs.io/en/latest/api/AttractionStrength.html
[Centroid Function]:     https://cvpack.readthedocs.io/en/latest/api/CentroidFunction.html
[Distance]:              https://cvpack.readthedocs.io/en/latest/api/Distance.html
[Helix angle content]:   https://cvpack.readthedocs.io/en/latest/api/HelixAngleContent.html
[Helix H-bond content]:  https://cvpack.readthedocs.io/en/latest/api/HelixHBondContent.html
[Helix RMSD content]:    https://cvpack.readthedocs.io/en/latest/api/HelixRMSDContent.html
[Helix torsion content]: https://cvpack.readthedocs.io/en/latest/api/HelixTorsionContent.html
[Number of contacts]:    https://cvpack.readthedocs.io/en/latest/api/NumberOfContacts.html
[Radius of gyration]:    https://cvpack.readthedocs.io/en/latest/api/RadiusOfGyration.html
[Rg squared]:            https://cvpack.readthedocs.io/en/latest/api/RgSquared.html
[RMSD]:                  https://cvpack.readthedocs.io/en/latest/api/RMSD.html
[Torsion]:               https://cvpack.readthedocs.io/en/latest/api/Torsion.html
[Torsion similarity]:    https://cvpack.readthedocs.io/en/latest/api/TorsionSimilarity.html
