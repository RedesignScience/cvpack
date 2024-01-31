"""
.. class:: Distance
   :platform: Linux, MacOS, Windows
   :synopsis: The distance between two atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


class Distance(openmm.CustomBondForce, BaseCollectiveVariable):
    r"""
    The distance between two atoms:

    .. math::

        d({\bf r}) = \| {\bf r}_2 - {\bf r}_1 \|.

    Parameters
    ----------
        atom1
            The index of the first atom
        atom2
            The index of the second atom
        pbc
            Whether to use periodic boundary conditions

    Example:
        >>> import cvpack
        >>> import openmm
        >>> system =openmm.System()
        >>> [system.addParticle(1) for i in range(2)]
        [0, 1]
        >>> distance = cvpack.Distance(0, 1)
        >>> system.addForce(distance)
        0
        >>> integrator =openmm.VerletIntegrator(0)
        >>> platform =openmm.Platform.getPlatformByName('Reference')
        >>> context =openmm.Context(system, integrator, platform)
        >>> context.setPositions([openmm.Vec3(0, 0, 0),openmm.Vec3(1, 1, 1)])
        >>> print(distance.getValue(context, digits=5))
        1.73205 nm

    """

    def __init__(self, atom1: int, atom2: int, pbc: bool = False) -> None:
        super().__init__("r")
        self.addBond(atom1, atom2, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(mmunit.nanometers, atom1, atom2, pbc)
