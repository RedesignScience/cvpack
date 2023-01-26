"""
.. module:: cvlib
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import List

import numpy as np
import openmm
from openmm import unit as mmunit


def _in_md_units(quantity: mmunit.Quantity) -> float:
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    """
    return quantity.value_in_unit_system(mmunit.md_unit_system)


class AbstractCollectiveVariable(openmm.Force):
    """
    An abstract class with common attributes and method for all CVs.

    """

    _unit = mmunit.dimensionless

    def _getSingleForceState(
        self, context: openmm.Context, getEnergy: bool = False, getForces: bool = False
    ) -> openmm.State:
        """
        Get an OpenMM State containing the potential energy and/or force values computed from this
        single force object.

        """
        forces = context.getSystem().getForces()
        free_groups = set(range(32)) - set(f.getForceGroup() for f in forces)
        old_group = self.getForceGroup()
        new_group = next(iter(free_groups))
        self.setForceGroup(new_group)
        state = context.getState(
            getEnergy=getEnergy, getForces=getForces, groups={new_group}
        )
        self.setForceGroup(old_group)
        return state

    def setUnit(self, unit: mmunit.Unit) -> None:
        """
        Set the unit of measurement of this collective variable.

        Parameters
        ----------
            unit
                The unit of measurement of this collective variable

        """
        self._unit = unit

    def getUnit(self) -> mmunit.Unit:
        """
        Get the unit of measurement of this collective variable.

        """
        return self._unit

    def evaluateInContext(self, context: openmm.Context) -> mmunit.Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        Parameters
        ----------
            context
                The context at which this collective variable should be evaluated

        Returns
        -------
            The value of this collective variable at the given context

        """
        state = self._getSingleForceState(context, getEnergy=True)
        return _in_md_units(state.getPotentialEnergy()) * self.getUnit()

    def effectiveMassInContext(self, context: openmm.Context) -> mmunit.Quantity:
        """
        Compute the effective mass of this collective variable at a given :OpenMM:`Context`.

        The effective mass of a collective variable :math:`q(\\mathbf{r})` is defined as
        :cite:`Chipot_2007`:

        .. math::
            m_\\mathrm{eff} = \\left(
                \\sum_{j=1}^N \\frac{1}{m_j} \\left\\|\\frac{dq}{d\\mathbf{r}_j}\\right\\|^2
            \\right)^{-1}

        Parameters
        ----------
            context
                The context at which this collective variable's effective mass should be evaluated

        Returns
        -------
            The value of this collective variable's effective mass at the given context

        """
        state = self._getSingleForceState(context, getForces=True)
        force_values = _in_md_units(state.getForces(asNumpy=True))
        indices = np.arange(context.getSystem().getNumParticles())
        masses_with_units = map(context.getSystem().getParticleMass, indices)
        mass_values = np.array(list(map(_in_md_units, masses_with_units)))
        effective_mass = 1.0 / np.sum(np.sum(force_values**2, axis=1) / mass_values)
        return (
            effective_mass * mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        )


class Distance(openmm.CustomBondForce, AbstractCollectiveVariable):
    """
    The distance between two atoms.

    Parameters
    ----------
        atom1
            The index of the first atom
        atom2
            The index of the second atom

    Example:
        >>> import cvlib
        >>> import openmm as mm
        >>> system = mm.System()
        >>> list(map(system.addParticle, [1] * 2))
        [0, 1]
        >>> distance = cvlib.Distance(0, 1)
        >>> system.addForce(distance)
        0
        >>> integrator = mm.CustomIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(system, integrator, platform)
        >>> context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1, 1, 1)])
        >>> print(distance.evaluateInContext(context))
        1.7320508075688772 nm

    """

    def __init__(self, atom1: int, atom2: int) -> None:
        super().__init__("r")
        self.addBond(atom1, atom2, [])
        self.setName("Distance")
        self.setUnit(mmunit.nanometers)


class RadiusOfGyration(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    The radius of gyration of a group of atoms, defined as:

    .. math::
        R_g = \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n \\|\\mathbf{r}_i - \\mathbf{r}_{\\rm mean}\\|^2 },

    where :math:`n` is the number of atoms in the group, :math:`\\mathbf{r}_i` is the coordinate of
    atom `i`, and :math:`\\mathbf{r}_{\\rm mean}` is the centroid of the group of atoms, that is,

    .. math::
        \\mathbf{r}_{\\rm mean} = \\frac{1}{n} \\sum_{i=1}^n \\mathbf{r}_i.

    Parameters
    ----------
        atoms
            The indices of the atoms in the group

    Example
    -------
        >>> import openmm
        >>> import cvlib
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> atoms = list(range(num_atoms))
        >>> rg_cv = cvlib.RadiusOfGyration(atoms)
        >>> model.system.addForce(rg_cv)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> print(rg_cv.evaluateInContext(context))
        0.295143056060787 nm
        >>> positions = model.positions[atoms, :]
        >>> centroid = positions.mean(axis=0)
        >>> print(mmunit.sqrt(((positions - centroid) ** 2).sum() / num_atoms))
        0.295143056060787 nm

    """

    def __init__(self, atoms: List[int]) -> None:
        num_atoms = len(atoms)
        num_groups = num_atoms + 1
        rgsq = "+".join(
            [f"distance(g{i+1}, g{num_groups})^2" for i in range(num_atoms)]
        )
        super().__init__(num_groups, f"sqrt(({rgsq})/{num_atoms})")
        for atom in atoms:
            self.addGroup([atom], [1])
        self.addGroup(atoms, [1] * num_atoms)
        self.addBond(list(range(num_groups)), [])
        self.setUsesPeriodicBoundaryConditions(False)
        self.setName("RadiusOfGyration")
        self.setUnit(mmunit.nanometers)
