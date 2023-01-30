"""
.. module:: cvlib
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import inspect
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

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
        super().__init__("r")
        self.addBond(atom1, atom2, [])
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
        >>> print(angle.evaluateInContext(context))
        1.5707963267948966 rad

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
        >>> print(torsion.evaluateInContext(context))
        1.5707963267948966 rad

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
        >>> print(radius_of_gyration.evaluateInContext(context))
        0.295143056060787 nm

    """

    def __init__(self, group: List[int]) -> None:
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


class PrincipalComponentAngle(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    Method for computing PCs :cite:`Sarabandi_2020`:

    Eigenvalues :cite:`Smith_1961`

    Eigenvectors :cite:`Franca_1989`

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> atoms = list(range(num_atoms))
        >>> pca = cvlib.cvlib.PCA(atoms[:num_atoms//2], atoms[num_atoms//2:])
        >>> model.system.addForce(pca)
        5
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> integrator = mm.VerletIntegrator(0)
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(round(pca.evaluateInContext(context), 10))
        0.0327870952

    """

    def __init__(self, group1: List[int], group2: List[int]) -> None:
        assert len(group1) > 1 and len(group2) > 1
        super().__init__("")
        definitions_a = self._addPrincipalComponentCalculation(0, group1, "_1")
        definitions_b = self._addPrincipalComponentCalculation(0, group2, "_2")
        energy = ";".join(["v0_1", definitions_a, definitions_b])
        print(energy)
        self.setEnergyFunction(energy)
        self._registerCV(1, group1, group2)

    def _principalComponent(self, order: int, number: int) -> Tuple[OrderedDict, OrderedDict]:
        """
        Get the variables and forces needed for computating the principal component vector of given
        order for a given number of coordinates.

        Parameters
        ----------
            order
                The order of the principal component vector. Valid values are 0, 1, and 2.
            number
                The number of coordinates

        """
        matrix = [["Cxx", "Cxy", "Cxz"], ["Cxy", "Cyy", "Cyz"], ["Cxz", "Cyz", "Czz"]]
        matrix_sq = list(
            ["+".join(f"{matrix[i][k]}*{matrix[k][j]}" for k in range(3)) for j in range(3)]
            for i in range(3)
        )
        variables = OrderedDict(
            v0=f"({(order == 0)*'b0 + '}b1*{matrix[0][order]} - b2*({matrix_sq[0][order]}))/s{order}",
            v1=f"({(order == 1)*'b0 + '}b1*{matrix[1][order]} - b2*({matrix_sq[1][order]}))/s{order}",
            v2=f"({(order == 2)*'b0 + '}b1*{matrix[2][order]} - b2*({matrix_sq[2][order]}))/s{order}",
            b0="a0*a2*b2",
            b1="(a2^2 - a1)*b2",
            b2="1/(a1*a2 - a0)",
            a0="s0*s1*s2",
            a1="s0*(s1 + s2) + s1*s2",
            a2="s0 + s1 + s2",
            s0="sqrt(m + 2*alpha)",  # sqrt(eig1)
            s1="sqrt(m - alpha + delta)",  # sqrt(eig2)
            s2="sqrt(m - alpha - delta)",  # sqrt(eig3)
            eig1="m + 2*alpha",  # delete when ready
            eig2="m - alpha + delta",  # delete when ready
            eig3="m - alpha - delta",  # delete when ready
            delta="sqrt(3*(p - alpha^2))",
            alpha="sqrt(p)*cos(phi/3)",
            phi="atan2(sqrt(p^3 - q^2), q)",
            p="(Axx^2 + Ayy^2 + Azz^2)/6 + (Cxy^2 + Cxz^2 + Cyz^2)/3",
            q="(Axx*(Ayy*Azz - Cyz^2) - Cxy*(Cxy*Azz - Cxz*Cyz) + Cxz*(Cxy*Cyz - Cxz*Ayy))/2",
            Axx="Cxx - m",
            Ayy="Cyy - m",
            Azz="Czz - m",
            m="(Cxx + Cyy + Czz)/3",
        )

        axes = ["x", "y", "z"]
        forces = OrderedDict()
        for i, ax1 in enumerate(axes):
            forces[f"S{ax1}"] = openmm.CustomExternalForce(ax1)
            for ax2 in axes[i:]:
                variables[f"C{ax1}{ax2}"] = f"({number}*S{ax1}{ax2} - S{ax1}*S{ax2})/{number**2}"
                forces[f"S{ax1}{ax2}"] = openmm.CustomExternalForce(f"{ax1}*{ax2}")

        return variables, forces

    def _addPrincipalComponentCalculation(self, order: int, group: List[int], suffix: str) -> str:
        """
        Add computation of the principal component vector of given order for a given set of
        atoms.

        Parameters
        ----------
            order
                The order of the principal component vector. Valid values are 0, 1, and 2.
            group
                The group of atoms
            suffix
                A suffix to distinguish this computation from others of the same kind

        """
        variables, forces = self._principalComponent(order, len(group))

        for var, force in forces.items():
            for i in group:
                force.addParticle(i)
            self.addCollectiveVariable(f"{var}{suffix}", force)

        definitions = ";".join(f"\n{var} = {expr}" for var, expr in variables.items())
        for var in list(variables.keys()) + list(forces.keys()):
            definitions = re.sub(rf"\b{var}\b", f"{var}{suffix}", definitions)

        return definitions
