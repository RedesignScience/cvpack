Collective Variable Package
===========================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linux/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/MacOS/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Windows/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/RedesignScience/cvpack/workflows/Linter/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ALinter)
[![Documentation Status](https://github.com/RedesignScience/cvpack/workflows/Docs/badge.svg)](https://github.com/RedesignScience/cvpack/actions?query=workflow%3ADocs)
[![Coverage Report](https://redesignscience.github.io/cvpack/development/coverage/coverage.svg)](https://redesignscience.github.io/cvpack/development/coverage)

[![Conda cvpack version](https://img.shields.io/conda/v/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![Conda cvpack platforms](https://img.shields.io/conda/pn/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![Conda cvpack downloads](https://img.shields.io/conda/dn/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)

[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/RedesignScience/cvpack/blob/main/LICENSE.md)
[![Twitter](https://badgen.net/badge/About/RedesignScience)](https://www.redesignscience.com)

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
| [Composite RMSD]        | multibody RMSD with concerted-rotation alignment                 |
| [Distance]              | distance between two atoms                                       |
| [Helix angle content]   | alpha-helix angle content of a sequence of residues              |
| [Helix H-bond content]  | alpha-helix hydrogen-bond content of a sequence of residues      |
| [Helix RMSD content]    | alpha-helix RMSD content of a sequence of residues               |
| [Helix torsion content] | alpha-helix Ramachandran content of a sequence of residues       |
| [Number of contacts]    | number of contacts between two groups of atoms                   |
| [Radius of gyration]    | radius of gyration of a group of atoms                           |
| [(Radius of gyration)^2]| square of the radius of gyration of a group of atoms             |
| [Residue coordination]  | number of contacts between two disjoint groups of residues       |
| [RMSD]                  | root-mean-square deviation with respect to a reference structure |
| [Sheet RMSD content]    | beta-sheet RMSD content of a sequence of residues                |
| [Torsion]               | torsion angle formed by four atoms                               |
| [Torsion similarity]    | degree of similarity between pairs of torsion angles             |

### Installation and Usage

CVPack is available as a conda package on the
[mdtools](https://anaconda.org/mdtools/cvpack) channel. To install it, run:

```bash
    conda install -c conda-forge -c mdtools cvpack
```

Or:

```bash
    mamba install -c mdtools cvpack
```

To use CVPack in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import cvpack
```

### Documentation

Documentation for the latest CVPack version is available at [Github Pages](https://redesignscience.github.io/cvpack/latest).

### Copyright

Copyright (c) 2023-2024 [C. Abreu](https://github.com/craabreu) & [Redesign Science](https://www.redesignscience.com)


#### Acknowledgements

Initial project based on the [CMS Cookiecutter] version 1.1.

[BiasVariable]:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
[CMS Cookiecutter]:   https://github.com/molssi/cookiecutter-cms
[CollectiveVariable]: https://ufedmm.readthedocs.io/en/latest/pythonapi/ufedmm.html#ufedmm.ufedmm.CollectiveVariable
[CustomCVForce]:      https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:              https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[Metadynamics]:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
[OpenMM]:             https://openmm.org
[UFED]:               https://ufedmm.readthedocs.io/en/latest/index.html

[Angle]:                  https://redesignscience.github.io/cvpack/api/Angle.html
[Atomic Function]:        https://redesignscience.github.io/cvpack/api/AtomicFunction.html
[Attraction Strength]:    https://redesignscience.github.io/cvpack/api/AttractionStrength.html
[Centroid Function]:      https://redesignscience.github.io/cvpack/api/CentroidFunction.html
[Composite RMSD]:         https://redesignscience.github.io/cvpack/api/CompositeRMSD.html
[Distance]:               https://redesignscience.github.io/cvpack/api/Distance.html
[Helix angle content]:    https://redesignscience.github.io/cvpack/api/HelixAngleContent.html
[Helix H-bond content]:   https://redesignscience.github.io/cvpack/api/HelixHBondContent.html
[Helix RMSD content]:     https://redesignscience.github.io/cvpack/api/HelixRMSDContent.html
[Helix torsion content]:  https://redesignscience.github.io/cvpack/api/HelixTorsionContent.html
[Number of contacts]:     https://redesignscience.github.io/cvpack/api/NumberOfContacts.html
[Radius of gyration]:     https://redesignscience.github.io/cvpack/api/RadiusOfGyration.html
[(Radius of gyration)^2]: https://redesignscience.github.io/cvpack/api/RadiusOfGyrationSq.html
[Residue coordination]:   https://redesignscience.github.io/cvpack/api/ResidueCoordination.html
[RMSD]:                   https://redesignscience.github.io/cvpack/api/RMSD.html
[Sheet RMSD content]:     https://redesignscience.github.io/cvpack/api/SheetRMSDContent.html
[Torsion]:                https://redesignscience.github.io/cvpack/api/Torsion.html
[Torsion similarity]:     https://redesignscience.github.io/cvpack/api/TorsionSimilarity.html
