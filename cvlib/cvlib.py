"""
.. module:: cvlib
   :platform: Unix, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import itertools
import openmm

from openmm import unit
from typing import List


def _in_md_units(quantity: unit.Quantity) -> float:
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    """
    return quantity.value_in_unit_system(unit.md_unit_system)


class AbstractCollectiveVariable(openmm.Force):
    def setUnit(self, unit: unit.Unit):
        self._unit = unit

    def getUnit(self) -> unit.Unit:
        return self._unit

    def evaluate(self, context: openmm.Context) -> unit.Quantity:
        forces = context.getSystem().getForces()
        free_groups = set(range(32)) - set(f.getForceGroup() for f in forces)
        old_group = self.getForceGroup()
        new_group = next(iter(free_groups))
        self.setForceGroup(new_group)
        state = context.getState(getEnergy=True, groups={new_group})
        self.setForceGroup(old_group)
        return _in_md_units(state.getPotentialEnergy())*self.getUnit()


class SquareRadiusOfGyration(openmm.CustomBondForce, AbstractCollectiveVariable):
    """
    The square of the radius of gyration of a group of atoms, defined as:

    .. math::
        R_g^2 = \\frac{1}{n^2} \\sum_i \\sum_{j>i} r_{i,j}^2,

    where :math:`n` is the number of atoms in the group and :math:`r_{i,j}` is the distance between
    atoms `i` and `j`.

    Parameters
    ----------
        atoms : List[int]
            The indices of the atoms in the group.

    Example
    -------
        >>> import openmm
        >>> import cvlib
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> RgSq = cvlib.SquareRadiusOfGyration(range(model.system.getNumParticles()))
        >>> model.system.addForce(RgSq)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> RgSqValue = RgSq.evaluate(context)
        >>> print(RgSqValue)
        0.08710942354090084 nm

    """

    def __init__(self, atoms: List[int]):
        super().__init__(f'r^2/{len(atoms)**2}')
        self.setUsesPeriodicBoundaryConditions(False)
        for i, j in itertools.combinations(atoms, 2):
            self.addBond(i, j)
        self.setUnit(unit.nanometers)


class RadiusOfGyration(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    The radius of gyration of a group of atoms, defined as:

    .. math::
        R_g = \\frac{1}{n} \\sqrt{\\sum_i \\sum_{j>i} r_{i,j}^2},

    where :math:`n` is the number of atoms in the group and :math:`r_{i,j}` is the distance between
    atoms `i` and `j`.

    Parameters
    ----------
        group : list(int)
            The indices of the atoms in the group.

    Example
    -------
        >>> import openmm
        >>> import cvlib
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> Rg = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
        >>> Rg.setForceGroup(1)
        >>> model.system.addForce(Rg)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> RgValue = Rg.evaluate(context)
        >>> print(RgValue)
        0.295143056060787 nm

    """

    def __init__(self, atoms: List[int]):
        RgSq = openmm.CustomBondForce('r^2')
        RgSq.setUsesPeriodicBoundaryConditions(False)
        for i, j in itertools.combinations(atoms, 2):
            RgSq.addBond(i, j)
        super().__init__(f'sqrt(RgSq)/{len(atoms)}')
        self.addCollectiveVariable('RgSq', RgSq)
        self.setUnit(unit.nanometers)
