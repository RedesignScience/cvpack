"""
.. class:: Torsion
   :platform: Linux, MacOS, Windows
   :synopsis: The torsion angle formed by four atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable


class Torsion(CollectiveVariable, openmm.CustomTorsionForce):
    r"""
    The torsion angle formed by four atoms:

    .. math::

        \varphi({\bf r}) = {\rm atan2}\left(\frac{
            ({\bf r}_{2,1} \times {\bf r}_{3,4}) \cdot {\bf u}_{2,3}
        }{
            {\bf r}_{2,1} \cdot {\bf r}_{3,4} -
            ({\bf r}_{2,1} \cdot {\bf u}_{2,3})
            ({\bf r}_{3,4} \cdot {\bf u}_{2,3})
        }\right),

    where :math:`{\bf r}_{i,j} = {\bf r}_j - {\bf r}_i`,
    :math:`{\bf u}_{2,3} = {\bf r}_{2,3}/\|{\bf r}_{2,3}\|`,
    and `atan2 <https://en.wikipedia.org/wiki/Atan2>`_ is the arctangent function that
    receives the numerator and denominator above as separate arguments.

    Parameters
    ----------
    atom1
        The index of the first atom.
    atom2
        The index of the second atom.
    atom3
        The index of the third atom.
    atom4
        The index of the fourth atom.
    pbc
        Whether to use periodic boundary conditions in distance calculations.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import openmm
    >>> system = openmm.System()
    >>> [system.addParticle(1) for i in range(4)]
    [0, 1, 2, 3]
    >>> torsion = cvpack.Torsion(0, 1, 2, 3, pbc=False)
    >>> system.addForce(torsion)
    0
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(system, integrator, platform)
    >>> positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    >>> context.setPositions([openmm.Vec3(*pos) for pos in positions])
    >>> torsion.getValue(context)
    1.5707... rad

    """

    def __init__(
        self,
        atom1: int,
        atom2: int,
        atom3: int,
        atom4: int,
        pbc: bool = False,
        name: str = "torsion",
    ) -> None:
        super().__init__("theta")
        self.addTorsion(atom1, atom2, atom3, atom4, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(
            name,
            mmunit.radians,
            atom1=atom1,
            atom2=atom2,
            atom3=atom3,
            atom4=atom4,
            pbc=pbc,
        )
        self._registerPeriodicBounds(-np.pi, np.pi)


Torsion.registerTag("!cvpack.Torsion")
