"""
.. module:: cvpack
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import inspect
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .unit import value_in_md_units


class SerializableResidue(mmapp.topology.Residue):
    """
    A class that extends OpenMM's Residue class with additional methods for
    serialization and deserialization.
    """

    def __init__(self, residue: mmapp.topology.Residue) -> None:
        super().__init__(residue.name, residue.index, None, residue.id, None)
        self._atoms = [
            mmapp.topology.Atom(atom.name, atom.element, atom.index, None, atom.id)
            for atom in residue.atoms()
        ]


class AbstractCollectiveVariable(openmm.Force):
    """
    An abstract class with common attributes and method for all CVs.
    """

    _unit: mmunit.Unit = mmunit.dimensionless
    _args: Dict[str, Any] = {}

    def __getstate__(self) -> Dict[str, Any]:
        return self._args

    def __setstate__(self, keywords: Dict[str, Any]) -> None:
        self.__init__(**keywords)

    def _registerCV(self, unit: mmunit.Unit, *args: Any, **kwargs: Any) -> None:
        """
        Register the newly created AbstractCollectiveVariable subclass instance.

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
        free_groups = set(range(32)) - other_groups
        old_group = self.getForceGroup()
        new_group = next(iter(free_groups))
        self.setForceGroup(new_group)
        context.reinitialize(preserveState=True)
        state = context.getState(
            getEnergy=getEnergy, getForces=getForces, groups=1 << new_group
        )
        self.setForceGroup(old_group)
        context.reinitialize(preserveState=True)
        return state

    def _precisionRound(self, number: float, digits: Optional[int] = None) -> float:
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
    def getArguments(cls) -> Tuple[OrderedDict, OrderedDict]:
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

    def getValue(
        self, context: openmm.Context, digits: Optional[int] = None
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
        self, context: openmm.Context, digits: Optional[int] = None
    ) -> mmunit.Quantity:
        """
        Compute the effective mass of this collective variable at a given
        :OpenMM:`Context`.

        The effective mass of a collective variable :math:`q({\\bf r})` is defined as
        :cite:`Chipot_2007`:

        .. math::

            m_\\mathrm{eff}({\\bf r}) = \\left(
                \\sum_{i=1}^N \\frac{1}{m_i} \\left\\|
                    \\frac{dq}{d{\\bf r}_i}
                \\right\\|^2
            \\right)^{-1}

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
            >>> model.system.addForce(radius_of_gyration)
            6
            >>> platform =openmm.Platform.getPlatformByName('Reference')
            >>> context =openmm.Context(
            ...     model.system,openmm.VerletIntegrator(0), platform
            ... )
            >>> context.setPositions(model.positions)
            >>> print(radius_of_gyration.getEffectiveMass(context, digits=6))
            30.94693 Da
        """
        state = self._getSingleForceState(context, getForces=True)
        force_values = value_in_md_units(state.getForces(asNumpy=True))
        mass_values = [
            value_in_md_units(context.getSystem().getParticleMass(i))
            for i in range(context.getSystem().getNumParticles())
        ]
        effective_mass = 1.0 / np.sum(np.sum(force_values**2, axis=1) / mass_values)
        unit = mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        return mmunit.Quantity(self._precisionRound(effective_mass, digits), unit)
