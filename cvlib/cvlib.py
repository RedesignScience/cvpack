"""
.. module:: cvlib
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import inspect
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openmm
from openmm import unit as mmunit

QuantityOrFloat = Union[mmunit.Quantity, float]


def _in_md_units(quantity: QuantityOrFloat) -> float:
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    Parameters
    ----------
        quantity
            The quantity to be converted

    Returns
    -------
        The numerical value of the quantity in the MD unit system

    """
    if mmunit.is_quantity(quantity):
        value = quantity.value_in_unit_system(mmunit.md_unit_system)
    else:
        value = quantity
    return value


class AbstractCollectiveVariable(openmm.Force):
    """
    An abstract class with common attributes and method for all CVs.

    """

    _unit = mmunit.dimensionless
    _args = {}

    def __getstate__(self) -> Dict[str, Any]:
        return self._args

    def __setstate__(self, kw: Dict[str, Any]) -> None:
        self.__init__(**kw)

    def _registerCV(self, unit: mmunit.Unit, *args: Any) -> None:
        """
        Register the newly created AbstractCollectiveVariable subclass instance.

        This method must always be called from Subclass.__init__.

        """
        self.setName(self.__class__.__name__)
        self.setUnit(unit)
        arguments, _ = self.getArguments()
        self._args = dict(zip(arguments, args))

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
            ('atom1', <class 'int'>) ('atom2', <class 'int'>) ('pbc', <class 'bool'>)
            >>> print(*defaults.items())
            ('pbc', True)

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

    def evaluateInContext(
        self, context: openmm.Context, digits: Optional[int] = None
    ) -> mmunit.Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        Optionally, the value of this collective variable can be rounded to a specified number of
        digits.

        Parameters
        ----------
            context
                The context at which this collective variable should be evaluated
            digits
                The number of digits to round to

        Returns
        -------
            The value of this collective variable at the given context

        """
        state = self._getSingleForceState(context, getEnergy=True)
        value = _in_md_units(state.getPotentialEnergy())
        return (round(value, digits) if digits else value) * self.getUnit()

    def effectiveMassInContext(
        self, context: openmm.Context, digits: Optional[int] = None
    ) -> mmunit.Quantity:
        """
        Compute the effective mass of this collective variable at a given :OpenMM:`Context`.

        The effective mass of a collective variable :math:`q({\\bf r})` is defined as
        :cite:`Chipot_2007`:

        .. math::

            m_\\mathrm{eff}({\\bf r}) = \\left(
                \\sum_{i=1}^N \\frac{1}{m_i} \\left\\|\\frac{dq}{d{\\bf r}_i}\\right\\|^2
            \\right)^{-1}

        Optionally, the value of the effective mass of this collective variable can be rounded to a
        specified number of digits.

        Parameters
        ----------
            context
                The context at which this collective variable's effective mass should be evaluated
            digits
                The number of digits to round to

        Returns
        -------
            The value of this collective variable's effective mass at the given context

        Example
        -------
            >>> import cvlib
            >>> import openmm as mm
            >>> from openmmtools import testsystems
            >>> model = testsystems.AlanineDipeptideImplicit()
            >>> peptide = [a.index for a in model.topology.atoms() if a.residue.name != 'HOH']
            >>> radius_of_gyration = cvlib.RadiusOfGyration(peptide)
            >>> model.system.addForce(radius_of_gyration)
            6
            >>> platform = mm.Platform.getPlatformByName('Reference')
            >>> context = mm.Context(model.system, mm.VerletIntegrator(0), platform)
            >>> context.setPositions(model.positions)
            >>> print(radius_of_gyration.effectiveMassInContext(context, 6))
            30.946932 Da

        """
        state = self._getSingleForceState(context, getForces=True)
        force_values = _in_md_units(state.getForces(asNumpy=True))
        indices = np.arange(context.getSystem().getNumParticles())
        masses_with_units = map(context.getSystem().getParticleMass, indices)
        mass_values = np.array(list(map(_in_md_units, masses_with_units)))
        effective_mass = 1.0 / np.sum(np.sum(force_values**2, axis=1) / mass_values)
        unit = mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        return (round(effective_mass, digits) if digits else effective_mass) * unit


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
        pbc
            Whether to use periodic boundary conditions

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
        >>> print(distance.evaluateInContext(context, 5))
        1.73205 nm

    """

    def __init__(self, atom1: int, atom2: int, pbc: bool = True) -> None:
        super().__init__("r")
        self.addBond(atom1, atom2, [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(mmunit.nanometers, atom1, atom2)


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
        >>> print(angle.evaluateInContext(context, 6))
        1.570796 rad

    """

    def __init__(self, atom1: int, atom2: int, atom3: int) -> None:
        super().__init__("theta")
        self.addAngle(atom1, atom2, atom3, [])
        self._registerCV(mmunit.radians, atom1, atom2, atom3)


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
        >>> print(torsion.evaluateInContext(context, 6))
        1.570796 rad

    """

    def __init__(self, atom1: int, atom2: int, atom3: int, atom4: int) -> None:
        super().__init__("theta")
        self.addTorsion(atom1, atom2, atom3, atom4, [])
        self._registerCV(mmunit.radians, atom1, atom2, atom3, atom4)


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
        >>> print(radius_of_gyration.evaluateInContext(context, 6))
        0.295143 nm

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
        self._registerCV(mmunit.nanometers, group)


class NumberOfContacts(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The number of contacts between two atom groups:

    .. math::
        N({\\bf r}) = \\sum_{i \\in {\\bf g}_1} \\sum_{j \\in {\\bf g}_2}
                        S\\left(\\frac{\\|{\\bf r}_j - {\\bf r}_i\\|}{r_0}\\right)

    where :math:`r_0` is the threshold distance for defining a contact and :math:`S(x)` is a step
    function equal to 1 if a contact is made or equal to 0 otherwise. In analysis, it is fine to
    make :math:`S(x) = H(1-x)`, where `H` is the `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. In a simulation, however,
    :math:`S(x)` should be a continuous approximation for :math:`H(1-x)`.

    Atom pairs are ignored for distances beyond a cutoff :math:`r_c`. To avoid discontinuities,
    a switching function is applied at :math:`r_s \\leq r \\leq r_c` to make :math:`S(r/r_0)`
    smoothly decay to zero.

    .. note::

        The two groups are allowed to overlap. In this case, terms with :math:`j = i`
        (self-contacts) are ignored and each combination with :math:`j \\neq i` is counted
        only once.

    Parameters
    ----------
        group1
            The indices of the atoms in the first group
        group2
            The indices of the atoms in the second group
        numAtoms
            The total number of atoms in the system (required by OpenMM)
        pbc
            Whether the system has periodic boundary conditions
        stepFunction
            The function "step(1-x)" (for analysis only) or a continuous approximation
            thereof
        thresholdDistance
            The threshold distance for considering two atoms as being in contact
        cutoffDistance
            The distance beyond which an atom pair will be ignored
        switchingDistance
            The distance beyond which a swithing function will be applied

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> carbons = [a.index for a in model.topology.atoms() if a.element == app.element.carbon]
        >>> num_atoms = model.topology.getNumAtoms()
        >>> optionals = {"pbc": False, "stepFunction": "step(1-x)"}
        >>> nc = cvlib.NumberOfContacts(carbons, carbons, num_atoms, **optionals)
        >>> model.system.addForce(nc)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> print(nc.evaluateInContext(context, 6))
        6.0 dimensionless

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: List[int],
        group2: List[int],
        numAtoms: int,
        pbc: bool = True,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: QuantityOrFloat = 0.3,
        cutoffDistance: QuantityOrFloat = 0.6,
        switchingDistance: QuantityOrFloat = 0.5,
    ) -> None:
        super().__init__(stepFunction + f"; x=r/{_in_md_units(thresholdDistance)}")
        nonbonded_method = self.CutoffPeriodic if pbc else self.CutoffNonPeriodic
        self.setNonbondedMethod(nonbonded_method)
        for _ in range(numAtoms):
            self.addParticle([])
        self.setUseSwitchingFunction(True)
        self.setCutoffDistance(cutoffDistance)
        self.setSwitchingDistance(switchingDistance)
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        self._registerCV(
            mmunit.dimensionless,
            group1,
            group2,
            numAtoms,
            pbc,
            stepFunction,
            _in_md_units(thresholdDistance),
            _in_md_units(cutoffDistance),
            _in_md_units(switchingDistance),
        )


class RootMeanSquareDeviation(openmm.RMSDForce, AbstractCollectiveVariable):
    """
    The minimum root-mean-square deviation (RMSD) between the current and reference coordinates of a
    group of `n` atoms:

    .. math::

        RMSD({\\bf r}) = \\sqrt{
            \\frac{1}{n} \\sum_{i=1}^n \\| {\\bf r}_i - {\\bf R}(\\bf r) {\\bf r}_i^{\\rm ref} \\|^2
        }

    where :math:`{\\bf R}(\\bf r)` is the rotation matrix that minimizes the RMSD.

    Parameters
    ----------
        referencePositions
            The reference coordinates. If there are ``numAtoms`` coordinates, they must refer to the
            the system atoms and be sorted accordingly. Otherwise, if there are ``n`` coordinates,
            with ``n=len(group)``, they must refer to the group atoms in the same order as they
            appear in ``group``. The first criterion has precedence when ``n == numAtoms``.
        group
            The index of the atoms in the group
        numAtoms
            The total number of atoms in the system (required by OpenMM)

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideImplicit()
        >>> num_atoms = model.topology.getNumAtoms()
        >>> group = list(range(num_atoms))
        >>> rmsd = cvlib.RootMeanSquareDeviation(model.positions, group, num_atoms)
        >>> model.system.addForce(rmsd)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> integrator.step(1000)
        >>> value = rmsd.evaluateInContext(context)
        >>> round(value/value.unit, 7)
        0.1231383

    """

    def __init__(
        self,
        referencePositions: Union[np.ndarray, List[openmm.Vec3], mmunit.Quantity],
        group: List[int],
        numAtoms: int,
    ) -> None:
        coords = _in_md_units(referencePositions)
        num_coords = coords.shape[0] if isinstance(coords, np.ndarray) else len(coords)
        assert num_coords == len(group) or num_coords == numAtoms
        if num_coords == numAtoms:
            positions = coords.copy()
            coords = np.array([positions[atom] for atom in group])
        else:
            positions = np.zeros((numAtoms, 3))
            for i, atom in enumerate(group):
                positions[atom, :] = np.array([coords[i][j] for j in range(3)])
        super().__init__(positions, group)
        self._registerCV(mmunit.nanometers, coords, group, numAtoms)
