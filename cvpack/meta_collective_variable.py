"""
.. class:: MetaCollectiveVariable
   :platform: Linux, MacOS, Windows
   :synopsis: A function of other collective variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t
from copy import copy

import openmm
from openmm import unit as mmunit

from .cvpack import BaseCollectiveVariable
from .units import ScalarQuantity, VectorQuantity, preprocess_units, Quantity


class MetaCollectiveVariable(openmm.CustomCVForce, BaseCollectiveVariable):
    r"""
    A collective variable that is a function of other collective variables:

    .. math::

        f({\bf r}) = F\left(c_1({\bf r}), c_2({\bf r}), \ldots, c_n({\bf r})\right)

    where :math:`F(c_1,c_2,\ldots,c_n)` is a user-defined function and
    :math:`c_i({\bf r})` is the value of the :math:`i`-th collective variable at a
    given configuration :math:`{\bf r}`.

    The function :math:`F` is defined as a string and can be any expression supported
    by the :OpenMM:`CustomCVForce` class. If the expression contains named parameters,
    these must be passed as keyword arguments to the :class:`MetaCollectiveVariable`
    constructor. Only scalar parameters are supported.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid :OpenMM:`CustomCVForce`
        expression.
    collective_variables
        A sequence of collective variables to be used in the function. The name of each
        collective variable must be unique and match a symbol used in the function.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `unit.dimensionless`
    period
        The period of the collective variable if it is periodic, or `None` if it is not

    Keyword Args
    ------------
    **parameters
        The named parameters of the function. They will become :OpenMM:`Context`
        parameters if this collective variable is added to an :OpenMM:`System`.

    Example
    -------
    >>> import cvpack
    >>> import math
    >>> import openmm
    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> driving_force = cvpack.MetaCollectiveVariable(
    ...     f"kappa/2 * min(delta,{2*math.pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     unit.kilojoules_per_mole,
    ...     kappa = 1000 * unit.kilojoules_per_mole/unit.radian**2,
    ...     phi0 = 120 * unit.degrees
    ... )
    >>> driving_force.addToSystem(model.system)
    >>> integrator = openmm.VerletIntegrator(0)
    >>> platform = openmm.Platform.getPlatformByName('Reference')
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> driving_force.getValue(context)
    548.3... kJ/mol
    >>> driving_force.getValues(context)
    {'phi': 3.1415912746972743 rad}
    """

    @preprocess_units
    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[str],
        unit: mmunit.Unit,
        period: t.Optional[ScalarQuantity] = None,
        name: str = "meta_collective_variable",
        **parameters: t.Union[ScalarQuantity, VectorQuantity],
    ) -> None:
        super().__init__(function)
        self._cvs = {cv.getName(): copy(cv) for cv in collective_variables}
        for name, cv in self._cvs.items():
            self.addCollectiveVariable(name, cv)
        for parameter, value in parameters.items():
            self.addGlobalParameter(parameter, value)
        self._registerCV(
            name, unit, function, collective_variables, unit, period, **parameters
        )
        self._registerPeriod(period)

    def getValues(self, context: openmm.Context) -> ScalarQuantity:
        return {
            name: Quantity(value, cv.getUnit())
            for (name, cv), value in zip(
                self._cvs.items(), self.getCollectiveVariableValues(context)
            )
        }


MetaCollectiveVariable.registerTag("!cvpack.MetaCollectiveVariable")
