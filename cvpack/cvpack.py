"""
.. module:: cvpack
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import collections
import functools
import inspect
import typing as t

import numpy as np
import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .unit import value_in_md_units


class SerializableResidue(mmapp.topology.Residue):
    r"""
    A class that extends OpenMM's Residue class with additional methods for
    serialization and deserialization.
    """

    def __init__(self, residue: mmapp.topology.Residue) -> None:
        super().__init__(residue.name, residue.index, None, residue.id, None)
        self._atoms = [
            mmapp.topology.Atom(atom.name, atom.element, atom.index, None, atom.id)
            for atom in residue.atoms()
        ]


class BaseCollectiveVariable(openmm.Force):
    r"""
    An abstract class with common attributes and method for all CVs.
    """

    _unit: mmunit.Unit = mmunit.dimensionless
    _mass_unit: mmunit.Unit = mmunit.dalton * mmunit.nanometers**2
    _args: t.Dict[str, t.Any] = {}

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self._args

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def _registerCV(self, unit: mmunit.Unit, *args: t.Any, **kwargs: t.Any) -> None:
        """
        Register the newly created BaseCollectiveVariable subclass instance.

        This method must always be called from Subclass.__init__.

        Parameters
        ----------
            unit
                The unit of measurement of this collective variable. It must be a unit
                in the MD unit system (mass in Da, distance in nm, time in ps,
                temperature in K, energy in kJ/mol, angle in rad).
            args
                The arguments needed to construct this collective variable
            kwargs
                The keyword arguments needed to construct this collective variable
        """
        self.setName(self.__class__.__name__)
        self.setUnit(unit)
        self._mass_unit = mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        arguments, _ = self.getArguments()
        self._args = dict(zip(arguments, args))
        self._args.update(kwargs)

    def _getSingleForceState(
        self, context: openmm.Context, getEnergy: bool = False, getForces: bool = False
    ) -> openmm.State:
        """
        Get an OpenMM State containing the potential energy and/or force values computed
        from this single force object.

        Parameters
        ----------
            context
                The context from which the state should be extracted
            getEnergy
                If True, the potential energy will be computed
            getForces
                If True, the forces will be computed

        Raises
        ------
            ValueError
                If this force is not present in the given context
        """
        forces = context.getSystem().getForces()
        if not any(force.this == self.this for force in forces):
            raise RuntimeError("This force is not present in the given context.")
        self_group = self.getForceGroup()
        other_groups = {
            force.getForceGroup() for force in forces if force.this != self.this
        }
        if self_group not in other_groups:
            return context.getState(
                getEnergy=getEnergy, getForces=getForces, groups=1 << self_group
            )
        old_group = self.getForceGroup()
        new_group = self.setUnusedForceGroup(0, context.getSystem())
        context.reinitialize(preserveState=True)
        state = context.getState(
            getEnergy=getEnergy, getForces=getForces, groups=1 << new_group
        )
        self.setForceGroup(old_group)
        context.reinitialize(preserveState=True)
        return state

    def _precisionRound(self, number: float, digits: t.Optional[int] = None) -> float:
        """
        Round a number to a specified number of precision digits (if specified).

        The number of precision digits is defined as the number of digits after the
        decimal point of the number's scientific notation representation.

        Parameters
        ----------
            number
                The number to be rounded
            digits
                The number of digits to round to. If None, the number will not be
                rounded.

        Returns
        -------
            The rounded number
        """
        if digits is None:
            return number
        power = f"{number:e}".split("e")[1]
        return round(number, -(int(power) - digits))

    @staticmethod
    def _checkUnitCompatibility(unit: mmunit.Unit) -> None:
        """
        Check if the given unit is compatible with the MD unit system.

        Parameters
        ----------
            unit
                The unit to check
        """
        if not np.isclose(
            mmunit.Quantity(1.0, unit).value_in_unit_system(mmunit.md_unit_system),
            1.0,
        ):
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")

    @classmethod
    def getArguments(cls) -> t.Tuple[collections.OrderedDict, collections.OrderedDict]:
        """
        Inspect the arguments needed for constructing an instance of this collective
        variable.

        Returns
        -------
            A dictionary with the type annotations of all arguments

            A dictionary with the default values of optional arguments

        Example
        -------
            >>> import cvpack
            >>> args, defaults = cvpack.RadiusOfGyration.getArguments()
            >>> for name, annotation in args.items():
            ...     print(f"{name}: {annotation}")
            group: typing.Sequence[int]
            pbc: <class 'bool'>
            weighByMass: <class 'bool'>
            >>> print(*defaults.items())
            ('pbc', False) ('weighByMass', False)
        """
        arguments = collections.OrderedDict()
        defaults = collections.OrderedDict()
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

    def setUnusedForceGroup(self, position: int, system: openmm.System) -> int:
        """
        Set the force group of this collective variable to the one at a given position
        in the ascending ordered list of unused force groups in an :OpenMM:`System`.

        .. note::

            Evaluating a collective variable (see :meth:`getValue`) or computing its
            effective mass (see :meth:`getEffectiveMass`) is more efficient when the
            collective variable is the only force in its own force group.

        Parameters
        ----------
            position
                The position of the force group in the ascending ordered list of unused
                force groups in the system
            system
                The system to search for unused force groups

        Returns
        -------
            The index of the force group that was set

        Raises
        ------
            RuntimeError
                If all force groups are already in use
        """
        free_groups = sorted(
            set(range(32)) - {force.getForceGroup() for force in system.getForces()}
        )
        if not free_groups:
            raise RuntimeError("All force groups are already in use.")
        new_group = free_groups[position]
        self.setForceGroup(new_group)
        return new_group

    def getValue(
        self, context: openmm.Context, digits: t.Optional[int] = None
    ) -> mmunit.Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        Optionally, the value can be rounded to a specified number of precision digits,
        which is the number of digits after the decimal point of the value in scientific
        notation.

        .. note::

            This method will be more efficient if the collective variable is the only
            force in its force group (see :OpenMM:`Force`).

        Parameters
        ----------
            context
                The context at which this collective variable should be evaluated
            digits
                The number of precision digits to round to. If None, the value will not
                be rounded.

        Returns
        -------
            The value of this collective variable at the given context
        """
        state = self._getSingleForceState(context, getEnergy=True)
        value = value_in_md_units(state.getPotentialEnergy())
        return mmunit.Quantity(self._precisionRound(value, digits), self.getUnit())

    def getEffectiveMass(
        self, context: openmm.Context, digits: t.Optional[int] = None
    ) -> mmunit.Quantity:
        r"""
        Compute the effective mass of this collective variable at a given
        :OpenMM:`Context`.

        The effective mass of a collective variable :math:`q({\bf r})` is defined as
        :cite:`Chipot_2007`:

        .. math::

            m_\mathrm{eff}({\bf r}) = \left(
                \sum_{i=1}^N \frac{1}{m_i} \left\|
                    \frac{dq}{d{\bf r}_i}
                \right\|^2
            \right)^{-1}

        Optionally, effective mass of this collective variable can be rounded to a
        specified number of precision digits, which is the number of digits after the
        decimal point of the effective mass in scientific notation.

        .. note::

            This method will be more efficient if the collective variable is the only
            force in its force group (see :OpenMM:`Force`).

        Parameters
        ----------
            context
                The context at which this collective variable's effective mass should be
                evaluated
            digits
                The number of precision digits to round to. If None, the value will not
                be rounded.

        Returns
        -------
            The effective mass of this collective variable at the given context

        Example
        -------
            >>> import cvpack
            >>> import openmm
            >>> from openmmtools import testsystems
            >>> model = testsystems.AlanineDipeptideImplicit()
            >>> peptide = [
            ...     a.index
            ...     for a in model.topology.atoms()
            ...     if a.residue.name != 'HOH'
            ... ]
            >>> radius_of_gyration = cvpack.RadiusOfGyration(peptide)
            >>> radius_of_gyration.setForceGroup(1)
            >>> radius_of_gyration.setUnusedForceGroup(0, model.system)
            1
            >>> model.system.addForce(radius_of_gyration)
            6
            >>> platform = openmm.Platform.getPlatformByName('Reference')
            >>> context = openmm.Context(
            ...     model.system,openmm.VerletIntegrator(0), platform
            ... )
            >>> context.setPositions(model.positions)
            >>> print(radius_of_gyration.getEffectiveMass(context, digits=6))
            30.94693 Da
        """
        state = self._getSingleForceState(context, getForces=True)
        # pylint: disable=protected-access,c-extension-no-member
        get_mass = functools.partial(
            openmm._openmm.System_getParticleMass, context.getSystem()
        )
        force_vectors = state.getForces(asNumpy=True)._value
        # pylint: enable=protected-access,c-extension-no-member
        squared_forces = np.sum(np.square(force_vectors), axis=1)
        nonzeros = np.nonzero(squared_forces)[0]
        if nonzeros.size == 0:
            return mmunit.Quantity(np.inf, self._mass_unit)
        mass_values = np.fromiter(map(get_mass, nonzeros), dtype=np.float64)
        effective_mass = 1.0 / np.sum(squared_forces[nonzeros] / mass_values)
        return mmunit.Quantity(
            self._precisionRound(effective_mass, digits), self._mass_unit
        )
