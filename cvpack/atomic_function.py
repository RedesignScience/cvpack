"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import openmm

from .cvpack import AbstractCollectiveVariable, UnitOrStr, in_md_units, str_to_unit


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
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> angle = cvpack.Angle(0, 11, 21)
        >>> colvar = cvpack.AtomicFunction('angle(p1, p2, p3)', [0, 11, 21], "radians", False)
        >>> [model.system.addForce(f) for f in [angle, colvar]]
        [5, 6]
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(angle.getValue(context, digits=6))
        2.318322 rad
        >>> print(colvar.getValue(context, digits=6))
        2.318322 rad
    """

    def __init__(
        self,
        function: str,
        group: Iterable[int],
        unit: UnitOrStr,
        pbc: bool = False,
    ) -> None:
        super().__init__(len(group), function)
        self.addBond(group, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        cv_unit = str_to_unit(unit) if isinstance(unit, str) else unit
        if in_md_units(1 * cv_unit) != 1:
            raise ValueError(f"Unit {cv_unit} is not compatible with the MD unit system.")
        self._registerCV(cv_unit, function, group, str(unit), pbc)
