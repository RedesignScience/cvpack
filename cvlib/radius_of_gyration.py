"""
.. class:: RadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: The radius of gyration of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import openmm
from openmm import unit as mmunit

from .cvlib import AbstractCollectiveVariable


class RadiusOfGyration(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\\bf r}) = \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n \\left\\|{\\bf r}_i -
                                \\frac{1}{n} \\sum_{i=j}^n {\\bf r}_j \\right\\|^2 }.

    Parameters
    ----------
        group
            The indices of the atoms in the group

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> radius_of_gyration = cvlib.RadiusOfGyration(list(range(num_atoms)))
        >>> model.system.addForce(radius_of_gyration)
        5
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> integrator = mm.VerletIntegrator(0)
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(radius_of_gyration.evaluateInContext(context, 6))
        0.295143 nm

    """

    def __init__(self, group: Iterable[int]) -> None:
        self._group = group
        num_atoms = len(group)
        num_groups = num_atoms + 1
        rgsq = "+".join([f"distance(g{i+1}, g{num_groups})^2" for i in range(num_atoms)])
        super().__init__(num_groups, f"sqrt(({rgsq})/{num_atoms})")
        for atom in group:
            self.addGroup([atom], [1])
        self.addGroup(group, [1] * num_atoms)
        self.addBond(list(range(num_groups)), [])
        self.setUsesPeriodicBoundaryConditions(False)
        self._registerCV(mmunit.nanometers, group)
