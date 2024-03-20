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

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


class MetaCollectiveVariable(openmm.CustomCVForce, BaseCollectiveVariable):
    r"""
    A collective variable that is a function of :math:`n` other collective variables:

    .. math::

        f({\bf r}) = F\left(c_1({\bf r}), c_2({\bf r}), \ldots, c_n({\bf r})\right)

    where :math:`F` is a user-defined function and :math:`c_i({\bf r})` is the value of
    the :math:`i`-th collective variable at the configuration :math:`{\bf r}`.

    The function :math:`F` is defined as a string and can be any expression supported
    by :OpenMM:`CustomCVForce`. If it contains named parameters, they must be passed as
    keyword arguments to the :class:`MetaCollectiveVariable` constructor. Only scalar
    parameters are supported.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid :OpenMM:`CustomCVForce`
        expression.
    collective_variables
        A dictionary with the collective variables used in the function. The keys are
        the names of the collective variables and the values are the corresponding
        :class:`BaseCollectiveVariable` objects.
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
        >>> from cvpack import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> phi = cvpack.Torsion(6, 8, 14, 16)
        >>> driving = cvpack.MetaCollectiveVariable(
        ...     f"kappa/2 * min(delta,{2*math.pi}-delta)^2; delta=abs(phi-phi0)",
        ...     {"phi": phi},
        ...     unit.kilojoules_per_mole,
        ...     kappa = 1000 * unit.kilojoules_per_mole/unit.radian**2,
        ...     phi0 = math.pi/2 * unit.radian,
        ... )
        >>> _ = model.system.addForce(driving)
        >>> integrator = openmm.VerletIntegrator(0)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(driving.getValue(context))
        1233.6... kJ/mol
    """

    yaml_tag = "!cvpack.MetaCollectiveVariable"

    @mmunit.convert_quantities
    def __init__(
        self,
        function: str,
        collective_variables: t.Dict[str, BaseCollectiveVariable],
        unit: mmunit.Unit,
        period: t.Optional[mmunit.ScalarQuantity] = None,
        **parameters: t.Union[mmunit.ScalarQuantity, mmunit.VectorQuantity],
    ) -> None:
        super().__init__(function)
        self._collective_variables = {
            name: copy(cv) for name, cv in collective_variables.items()
        }
        for name, cv in self._collective_variables.items():
            cv.setName(name)
            index = self.addCollectiveVariable(name, cv)
            cv.this = self.getCollectiveVariable(index).this
        for parameter, value in parameters.items():
            self.addGlobalParameter(parameter, value)
        unit = mmunit.SerializableUnit(unit)
        self._registerCV(
            unit,
            function,
            self._collective_variables,
            unit,
            period,
            **parameters,
        )
        self._registerPeriod(period)
