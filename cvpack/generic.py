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


class Generic(BaseCollectiveVariable):
    r"""
    A generic collective variable.

    Parameters
    ----------
        atom1
            The index of the first atom
        atom2
            The index of the second atom
        atom3
            The index of the third atom
        pbc
            Whether to use periodic boundary conditions

    Example:
        >>> import cvpack
        >>> import numpy as np
        >>> import openmm
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> cv1 = cvpack.Angle(0, 1, 2)
        >>> model.system.addForce(cv1)
        5
        >>> angle = openmm.CustomAngleForce("theta")
        >>> angle.addAngle(0, 1, 2)
        0
        >>> cv2 = cvpack.Generic(angle, unit.radian, period=2*np.pi)
        >>> model.system.addForce(cv2)
        6
        >>> integrator = openmm.VerletIntegrator(0)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(cv1.getValue(context))
        1.911... rad
        >>> print(cv2.getValue(context))
        1.911... rad
        >>> assert isinstance(cv2, openmm.CustomAngleForce)
    """

    yaml_tag = "!cvpack.Generic"

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
        self.__dict__.update(force_copy.__dict__)
        self.__class__.__bases__ += (force_copy.__class__,)
        self._registerCV(unit, openmmForce, unit, period)
        if period is not None:
            self._registerPeriod(period)
