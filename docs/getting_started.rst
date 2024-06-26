Getting Started
===============

Introduction
------------

CVPack is a Python package with pre-defined Collective variable (CV) classes for `OpenMM`_.

Possible use cases include:

*   Adding CVs to `CustomCVForce`_ instances in `OpenMM`_
*   Defining `BiasVariable`_ instances for `Metadynamics`_ simulations
*   Defining `CollectiveVariable`_ instances for `UFED`_ simulations
*   Harnessing the power of `OpenMM`_ for trajectory analysis

Installation
------------

To install CVPack in a conda environment, run the following command::

    conda install -c conda-forge -c mdtools cvpack

Or use mamba instead::

    mamba install -c mdtools cvpack

Usage
-----

To use CVPack, import the package in your Python script or Jupyter notebook::

    import cvpack

Example
-------

The following example shows how to use CVPack to define CVs for quantifying the helix content of
a sequence of residues in a protein. The example uses the `testsystems` module from the
`OpenMMTools`_ package to create a system with a T4 lysozyme L99A molecule. The longest helix in
this protein occurs between residues LYS60 and ARG80. Four different CVs are defined and added
to the system, which is then used to create an OpenMM `Context`_. The CVs are finally evaluated,
rounded to seven decimal places, and printed to the screen::

    import cvpack
    import openmm
    from openmm import app, unit
    from openmmtools import testsystems
    model = testsystems.LysozymeImplicit()
    num_atoms = model.system.getNumParticles()
    residues = [*model.topology.residues()][59:80]
    helix_content = [
        cvpack.HelixAngleContent(residues, normalize=True),
        cvpack.HelixHBondContent(residues, normalize=True),
        cvpack.HelixRMSDContent(residues, num_atoms, normalize=True),
        cvpack.HelixTorsionContent(residues, normalize=True),
    ]
    for cv in helix_content:
        model.system.addForce(cv)
    platform = openmm.Platform.getPlatformByName('Reference')
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    for cv in helix_content:
        print(f"{cv.getName()}: {cv.getValue(context)}")

The output should be::

    HelixAngleContent: 0.9873989 dimensionless
    HelixHBondContent: 0.93414 dimensionless
    HelixRMSDContent: 0.9946999 dimensionless
    HelixTorsionContent: 0.918571 dimensionless

The output shows that the normalized helix content of residues LYS60-ARG80 of the T4 lysozyme L99A
molecule is close to one, as expected, no matter which measure is used.

.. _BiasVariable:       http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
.. _CollectiveVariable: https://craabreu.github.io/ufedmm/pythonapi/ufedmm.html#ufedmm.ufedmm.CollectiveVariable
.. _Context:            http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
.. _CustomCVForce:      http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _Force:              http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
.. _Metadynamics:       http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
.. _OpenMM:             http://openmm.org
.. _OpenMMTools:        https://openmmtools.readthedocs.io/en/stable
.. _UFED:               https://craabreu.github.io/ufedmm/
