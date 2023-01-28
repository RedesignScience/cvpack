"""
.. module:: cvlib
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import inspect
from collections import OrderedDict
from typing import Dict, List, Tuple

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
        state = context.getState(getEnergy=getEnergy, getForces=getForces, groups={new_group})
        self.setForceGroup(old_group)
        return state

    @classmethod
    def getArguments(cls) -> Tuple[OrderedDict, OrderedDict]:
        """
        Inspect the arguments needed for constructing an instance of this collective
        variable.

        Returns
        -------
            arguments
                An ordered dictionary containing the type annotations of all arguments
            defaults
                An ordered dictionary containing default values of optional arguments

        Example
        =======
            >>> import cvlib
            >>> args, defaults = cvlib.Distance.getArguments()
            >>> print(*args.items())
            ('atom1', <class 'int'>) ('atom2', <class 'int'>)
            >>> print(*defaults.items())
            <BLANKLINE>

        Example
        =======
            >>> import cvlib
            >>> radius_of_gyration = cvlib.RadiusOfGyration([1, 2, 3])
            >>> args, _ = radius_of_gyration.getArguments()
            >>> print(*args.items())
            ('group', typing.List[int])

        """
        arguments = OrderedDict()
        defaults = OrderedDict()
        for name, parameter in inspect.signature(cls).parameters.items():
            arguments[name] = parameter.annotation
            if parameter.default is not inspect.Parameter.empty:
                defaults[name] = parameter.default
        return arguments, defaults

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

        The effective mass of a collective variable :math:`q({\\bf r})` is defined as
        :cite:`Chipot_2007`:

        .. math::

            m_\\mathrm{eff}({\\bf r}) = \\left(
                \\sum_{i=1}^N \\frac{1}{m_i} \\left\\|\\frac{dq}{d{\\bf r}_i}\\right\\|^2
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
        return effective_mass * mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2


class Distance(openmm.CustomBondForce, AbstractCollectiveVariable):
    """
    The distance between two atoms:

    .. math::

        d({\\bf r}) = \\| {\\bf r}_2 - {\\bf r}_1 \\|.

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
        >>> [system.addParticle(1) for i in range(2)]
        [0, 1]
        >>> distance = cvlib.Distance(0, 1)
        >>> system.addForce(distance)
        0
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(system, integrator, platform)
        >>> context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1, 1, 1)])
        >>> print(distance.evaluateInContext(context))
        1.7320508075688772 nm

    """

    def __init__(self, atom1: int, atom2: int) -> None:
        self._atoms = atom1, atom2
        super().__init__("r")
        self.addBond(atom1, atom2, [])
        self.setName("Distance")
        self.setUnit(mmunit.nanometers)

    def __getstate__(self) -> Dict[str, int]:
        atom1, atom2 = self._atoms
        return dict(atom1=atom1, atom2=atom2)

    def __setstate__(self, kw: Dict[str, int]) -> None:
        self.__init__(**kw)


class Angle(openmm.CustomAngleForce, AbstractCollectiveVariable):
    """
    The angle formed by three atoms:

    .. math::

        \\theta({\\bf r}) =
            \\mathrm{acos}\\left(
                \\frac{{\\bf r}_{2,1} \\cdot {\\bf r}_{2,3} }
                       {\\| {\\bf r}_{2,1} \\| \\| {\\bf r}_{2,3} \\|}
            \\right),

    where :math:`{\\bf r}_{i,j} = {\\bf r}_j - {\\bf r}_i`.

    Parameters
    ----------
        atom1
            The index of the first atom
        atom2
            The index of the second atom
        atom3
            The index of the third atom

    Example:
        >>> import cvlib
        >>> import openmm as mm
        >>> system = mm.System()
        >>> [system.addParticle(1) for i in range(3)]
        [0, 1, 2]
        >>> angle = cvlib.Angle(0, 1, 2)
        >>> system.addForce(angle)
        0
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(system, integrator, platform)
        >>> positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        >>> context.setPositions([mm.Vec3(*pos) for pos in positions])
        >>> print(angle.evaluateInContext(context))
        1.5707963267948966 rad

    """

    def __init__(self, atom1: int, atom2: int, atom3: int) -> None:
        self._atoms = atom1, atom2, atom3
        super().__init__("theta")
        self.addAngle(atom1, atom2, atom3, [])
        self.setName("Angle")
        self.setUnit(mmunit.radians)

    def __getstate__(self) -> Dict[str, int]:
        atom1, atom2, atom3 = self._atoms
        return dict(atom1=atom1, atom2=atom2, atom3=atom3)

    def __setstate__(self, kw: Dict[str, int]) -> None:
        self.__init__(**kw)


class Torsion(openmm.CustomTorsionForce, AbstractCollectiveVariable):
    """
    The torsion angle formed by four atoms:

    .. math::

        \\varphi({\\bf r}) = \\mathrm{atan2}\\left(\\frac{
            ({\\bf r}_{2,1} \\times {\\bf r}_{3,4}) \\cdot {\\bf u}_{2,3}
        }{
            {\\bf r}_{2,1} \\cdot {\\bf r}_{3,4} - ({\\bf r}_{2,1} \\cdot {\\bf u}_{2,3})
                                                   ({\\bf r}_{3,4} \\cdot {\\bf u}_{2,3})
        }\\right),

    where :math:`{\\bf r}_{i,j} = {\\bf r}_j - {\\bf r}_i`,
    :math:`{\\bf u}_{2,3} = {\\bf r}_{2,3}/\\|{\\bf r}_{2,3}\\|`,
    and `atan2 <https://en.wikipedia.org/wiki/Atan2>`_ is the 2-argument arctangent function.

    Parameters
    ----------
        atom1
            The index of the first atom
        atom2
            The index of the second atom
        atom3
            The index of the third atom
        atom4
            The index of the fourth atom

    Example:
        >>> import cvlib
        >>> import openmm as mm
        >>> system = mm.System()
        >>> [system.addParticle(1) for i in range(4)]
        [0, 1, 2, 3]
        >>> torsion = cvlib.Torsion(0, 1, 2, 3)
        >>> system.addForce(torsion)
        0
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(system, integrator, platform)
        >>> positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
        >>> context.setPositions([mm.Vec3(*pos) for pos in positions])
        >>> print(torsion.evaluateInContext(context))
        1.5707963267948966 rad

    """

    def __init__(self, atom1: int, atom2: int, atom3: int, atom4: int) -> None:
        self._atoms = atom1, atom2, atom3, atom4
        super().__init__("theta")
        self.addTorsion(atom1, atom2, atom3, atom4, [])
        self.setName("Torsion")
        self.setUnit(mmunit.radians)

    def __getstate__(self) -> Dict[str, int]:
        atom1, atom2, atom3, atom4 = self._atoms
        return dict(atom1=atom1, atom2=atom2, atom3=atom3, atom4=atom4)

    def __setstate__(self, kw: Dict[str, int]) -> None:
        self.__init__(**kw)


class RadiusOfGyration(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\\bf r}) = \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n \\left\\|{\\bf r}_i -
                                \\frac{1}{n} \\sum_{i=j}^n {\\bf r}_j \\right\\|^2 }.

    Parameters
    ----------
        group
            The indices of the atoms in the group

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> radius_of_gyration = cvlib.RadiusOfGyration(list(range(num_atoms)))
        >>> model.system.addForce(radius_of_gyration)
        5
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> integrator = mm.VerletIntegrator(0)
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(radius_of_gyration.evaluateInContext(context))
        0.295143056060787 nm

    """

    def __init__(self, group: List[int]) -> None:
        self._group = group
        num_atoms = len(group)
        num_groups = num_atoms + 1
        rgsq = "+".join([f"distance(g{i+1}, g{num_groups})^2" for i in range(num_atoms)])
        super().__init__(num_groups, f"sqrt(({rgsq})/{num_atoms})")
        for atom in group:
            self.addGroup([atom], [1])
        self.addGroup(group, [1] * num_atoms)
        self.addBond(list(range(num_groups)), [])
        self.setUsesPeriodicBoundaryConditions(False)
        self.setName("RadiusOfGyration")
        self.setUnit(mmunit.nanometers)

    def __getstate__(self) -> Dict[str, List[int]]:
        return dict(group=self._group)

    def __setstate__(self, kw: Dict[str, List[int]]) -> None:
        self.__init__(**kw)
