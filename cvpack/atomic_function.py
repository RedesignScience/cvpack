"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Sequence

import openmm
from openmm import unit as mmunit

from .cvpack import AbstractCollectiveVariable
from .unit import SerializableUnit


class AtomicFunction(openmm.CustomCompoundBondForce, AbstractCollectiveVariable):
    """
    A generic function of the coordinates of a group of `n` atoms:

    .. math::

        f({\\bf r}) = F({\\bf r}_1, {\\bf r}_2, \\dots, {\\bf r}_n)

    where :math:`F` is a user-defined function. The function :math:`F` is defined as a string and
    can be any valid :OpenMM:`CustomCompoundBondForce` expression.

    Parameters
    ----------
        function
            The function to be evaluated. It must be a valid :OpenMM:`CustomCompoundBondForce`
            expression
        group
            The group of atoms to be used in the function
        unit
            The unit of measurement of the collective variable. It must be compatible with the
            MD unit system (mass in `daltons`, distance in `nanometers`, time in `picoseconds`,
            temperature in `kelvin`, energy in `kilojoules_per_mol`, angle in `radians`). If the
            collective variables does not have a unit, use `dimensionless`
        pbc
            Whether to use periodic boundary conditions

    Raises
    ------
        ValueError
            If the collective variable is not compatible with the MD unit system

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> angle = cvpack.Angle(0, 11, 21)
        >>> colvar = cvpack.AtomicFunction('angle(p1, p2, p3)', [0, 11, 21], unit.radians, False)
        >>> [model.system.addForce(f) for f in [angle, colvar]]
        [5, 6]
        >>> integrator =openmm.VerletIntegrator(0)
        >>> platform =openmm.Platform.getPlatformByName('Reference')
        >>> context =openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(angle.getValue(context, digits=6))
        2.318322 rad
        >>> print(colvar.getValue(context, digits=6))
        2.318322 rad
    """

    def __init__(
        self,
        function: str,
        group: Sequence[int],
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> None:
        super().__init__(len(group), function)
        self.addBond(group, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        if mmunit.Quantity(1, unit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")
        self._registerCV(unit, function, group, SerializableUnit(unit), pbc)
