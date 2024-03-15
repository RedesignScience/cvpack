"""
.. class:: Generic
   :platform: Linux, MacOS, Windows
   :synopsis: Generic collective variable

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


class OpenMMForceWrapper(BaseCollectiveVariable):
    r"""
    A collective variable whose numerical value is computed from the potential energy,
    in kJ/mol, of an OpenMM force object.

    Parameters
    ----------
    openmmForce
        The OpenMM force whose potential energy will be used to define the collective
        variable. It can be passed as an instance of :OpenMM:`Force` or as a string
        containing the XML serialization of the force.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `dimensionless`.
    period
        The period of the collective variable if it is periodic, or `None` if it is not.

    Example:
        >>> import cvpack
        >>> import numpy as np
        >>> import openmm
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> angle = openmm.CustomAngleForce("theta")
        >>> angle.addAngle(0, 1, 2)
        0
        >>> cv = cvpack.OpenMMForceWrapper(angle, unit.radian, period=2*np.pi)
        >>> assert isinstance(cv, openmm.CustomAngleForce)
        >>> cv.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(cv)
        5
        >>> integrator = openmm.VerletIntegrator(0)
        >>> platform = openmm.Platform.getPlatformByName("Reference")
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(cv.getValue(context))
        1.911... rad
        >>> print(cv.getEffectiveMass(context))
        0.00538... nm**2 Da/(rad**2)
    """

    yaml_tag = "!cvpack.OpenMMForceWrapper"

    def __init__(  # pylint: disable=too-many-arguments, super-init-not-called
        self,
        openmmForce: t.Union[openmm.Force, str],
        unit: mmunit.Unit,
        period: t.Optional[mmunit.ScalarQuantity] = None,
    ) -> None:
        if isinstance(openmmForce, openmm.Force):
            openmmForce = openmm.XmlSerializer.serialize(openmmForce)
        unit = mmunit.SerializableUnit(unit)
        force_copy = openmm.XmlSerializer.deserialize(openmmForce)
        self.this = force_copy.this
        self.__class__.__bases__ = (BaseCollectiveVariable, type(force_copy))
        self._registerCV(unit, openmmForce, unit, period)
        if period is not None:
            self._registerPeriod(period)
