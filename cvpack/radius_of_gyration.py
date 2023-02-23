"""
.. class:: RadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: The radius of gyration of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import openmm
from openmm import unit as mmunit

from .cvpack import AbstractCollectiveVariable


class RadiusOfGyration(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\\bf r}) = \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n \\left\\|
            {\\bf r}_i - {\\bf r}_c({\\bf r})
        \\right\\|^2 }.

    where :math:`{\\bf r}_c({\\bf r})` is the centroid of the group:

    .. math::

        {\\bf r}_c({\\bf r}) = \\frac{1}{n} \\sum_{i=j}^n {\\bf r}_j

    Optionally, the atoms can be weighted by their masses. In this case, the centroid is computed
    as:

    .. math::

        {\\bf r}_c({\\bf r}) = \\frac{1}{M} \\sum_{i=1}^n m_i {\\bf r}_i

    where :math:`M = \\sum_{i=1}^n m_i` is the total mass of the group.

    Parameters
    ----------
        group
            The indices of the atoms in the group
        pbc
            Whether to use periodic boundary conditions
        weighByMass
            Whether to weigh the atoms by their masses

    Example
    -------
        >>> import cvpack
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> radius_of_gyration = cvpack.RadiusOfGyration(list(range(num_atoms)))
        >>> model.system.addForce(radius_of_gyration)
        5
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> integrator = mm.VerletIntegrator(0)
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(radius_of_gyration.getValue(context, digits=6))
        0.295143 nm

    """

    def __init__(self, group: Iterable[int], pbc: bool = False, weighByMass: bool = False) -> None:
        self._group = group
        num_atoms = len(group)
        num_groups = num_atoms + 1
        sum_dist_sq = "+".join([f"distance(g{i+1}, g{num_atoms + 1})^2" for i in range(num_atoms)])
        super().__init__(num_groups, f"sqrt(({sum_dist_sq})/{num_atoms})")
        for atom in group:
            self.addGroup([atom], [1])
        self.addGroup(group, None if weighByMass else [1] * num_atoms)
        self.addBond(list(range(num_groups)), [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(mmunit.nanometers, group, pbc, weighByMass)
