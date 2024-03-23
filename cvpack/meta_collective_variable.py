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
from .units import Quantity, ScalarQuantity, VectorQuantity
from .utils import compute_effective_mass


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
    variables
        A sequence of :class:`BaseCollectiveVariable` instances that represent the
        collective variables on which the meta-collective variable depends. The name
        of each collective variable must be unique and match a symbol used in the
        function.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `unit.dimensionless`
    periodicBounds
        The periodic bounds of the collective variable if it is periodic, or `None` if
        it is not

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
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> driving_force = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*math.pi}-delta)^2; delta=abs(phi-phi0)",
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
    >>> driving_force.getInnerValues(context)
    {'phi': 3.14... rad}
    >>> driving_force.getInnerEffectiveMasses(context)
    {'phi': 0.05119... nm**2 Da/(rad**2)}
    """

    def __init__(
        self,
        function: str,
        variables: t.Iterable[str],
        unit: mmunit.Unit,
        periodicBounds: t.Optional[VectorQuantity] = None,
        name: str = "meta_collective_variable",
        **parameters: t.Union[ScalarQuantity, VectorQuantity],
    ) -> None:
        super().__init__(function)
        self._cvs = {cv.getName(): copy(cv) for cv in variables}
        for cvname, cv in self._cvs.items():
            self.addCollectiveVariable(cvname, cv)
        for parameter, value in parameters.items():
            self.addGlobalParameter(parameter, value)
        self._registerCV(
            name,
            unit,
            function,
            variables,
            unit,
            periodicBounds,
            **parameters,
        )
        if periodicBounds is not None:
            self._registerPeriodicBounds(*periodicBounds)

    def getInnerValues(self, context: openmm.Context) -> t.Dict[str, Quantity]:
        """
        Get the values of the collective variables on which the meta-collective variable
        depends. The values are returned as a dictionary with the names of the
        collective variables as keys.

        Parameters
        ----------
        context
            The context in which the collective variables will be evaluated.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the collective variables as keys and their
            values as values.
        """
        values = self.getCollectiveVariableValues(context)
        return {
            name: Quantity(value, cv.getUnit())
            for (name, cv), value in zip(self._cvs.items(), values)
        }

    def getInnerEffectiveMasses(self, context: openmm.Context) -> t.Dict[str, Quantity]:
        """
        Get the effective masses of the collective variables on which the
        meta-collective variable depends. The effective masses are calculated from the
        forces acting on the particles that represent the collective variables.

        Parameters
        ----------
        context
            The context in which the collective variables will be evaluated.

        Returns
        -------
        Dict[str, Quantity]
            A dictionary with the names of the collective variables as keys and their
            effective masses as values.
        """
        inner_context = self.getInnerContext(context)
        masses = [
            compute_effective_mass(force, inner_context)
            for force in inner_context.getSystem().getForces()
        ]
        return {
            name: Quantity(mass, cv.getMassUnit())
            for (name, cv), mass in zip(self._cvs.items(), masses)
        }


MetaCollectiveVariable.registerTag("!cvpack.MetaCollectiveVariable")
