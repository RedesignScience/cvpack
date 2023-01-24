"""
.. module:: cvlib
   :platform: Unix, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

.. _Context: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
.. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
.. _System: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.System.html
.. _coordination: https://www.plumed.org/doc-v2.6/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
.. _PLUMED: https://www.plumed.org

"""

import re
import itertools
import math
import openmm

from openmm import unit


def _in_md_units(quantity):
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    """
    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    else:
        return quantity


class SquareRadiusOfGyration(openmm.CustomBondForce):
    """
    The square of the radius of gyration of a group of atoms, defined as:

    .. math::
        R_g^2 = \\frac{1}{n^2} \\sum_i \\sum_{j>i} r_{i,j}^2,

    where :math:`n` is the number of atoms in the group and :math:`r_{i,j}` is the distance between
    atoms `i` and `j`.

    Parameters
    ----------
        group : list(int)
            The indices of the atoms in the group.

    Example
    -------
        >>> import openmm
        >>> import ufedmm
        >>> from ufedmm import cvlib
        >>> model = ufedmm.AlanineDipeptideModel()
        >>> RgSq = cvlib.SquareRadiusOfGyration(range(model.system.getNumParticles()))
        >>> RgSq.setForceGroup(1)
        >>> model.system.addForce(RgSq)
        4
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
        0.08711416289256209

    """

    def __init__(self, group):
        super().__init__(f'r^2/{len(group)**2}')
        self.setUsesPeriodicBoundaryConditions(False)
        for i, j in itertools.combinations(group, 2):
            self.addBond(i, j)


class RadiusOfGyration(openmm.CustomCVForce):
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
        >>> import ufedmm
        >>> from ufedmm import cvlib
        >>> model = ufedmm.AlanineDipeptideModel()
        >>> Rg = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
        >>> Rg.setForceGroup(1)
        >>> model.system.addForce(Rg)
        4
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
        0.2951510848575048

    """

    def __init__(self, group):
        RgSq = openmm.CustomBondForce('r^2')
        RgSq.setUsesPeriodicBoundaryConditions(False)
        for i, j in itertools.combinations(group, 2):
            RgSq.addBond(i, j)
        super().__init__(f'sqrt(RgSq)/{len(group)}')
        self.addCollectiveVariable('RgSq', RgSq)
