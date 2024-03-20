"""
.. module:: cvpack
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import collections
import inspect
import typing as t

import openmm
import yaml
from openmm import app as mmapp

from cvpack import unit as mmunit

from .unit import value_in_md_units
from .utils import compute_effective_mass, get_single_force_state


class SerializableAtom(yaml.YAMLObject):
    r"""
    A serializable version of OpenMM's Atom class.
    """

    yaml_tag = "!cvpack.Atom"

    def __init__(  # pylint: disable=super-init-not-called
        self, atom: t.Union[mmapp.topology.Atom, "SerializableAtom"]
    ) -> None:
        self.name = atom.name
        self.index = atom.index
        if isinstance(atom, mmapp.topology.Atom):
            self.element = atom.element.symbol
            self.residue = atom.residue.index
        else:
            self.element = atom.element
            self.residue = atom.residue
        self.id = atom.id

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self.__dict__

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__dict__.update(keywords)


yaml.SafeDumper.add_representer(SerializableAtom, SerializableAtom.to_yaml)
yaml.SafeLoader.add_constructor(SerializableAtom.yaml_tag, SerializableAtom.from_yaml)


class SerializableResidue(yaml.YAMLObject):
    r"""
    A serializable version of OpenMM's Residue class.
    """

    yaml_tag = "!cvpack.Residue"

    def __init__(  # pylint: disable=super-init-not-called
        self, residue: t.Union[mmapp.topology.Residue, "SerializableResidue"]
    ) -> None:
        self.name = residue.name
        self.index = residue.index
        if isinstance(residue, mmapp.topology.Residue):
            self.chain = residue.chain.index
        else:
            self.chain = residue.chain
        self.id = residue.id
        self._atoms = list(map(SerializableAtom, residue.atoms()))

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self.__dict__

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__dict__.update(keywords)

    def __len__(self) -> int:
        return len(self._atoms)

    def atoms(self):
        """Iterate over all Atoms in the Residue."""
        return iter(self._atoms)


yaml.SafeDumper.add_representer(SerializableResidue, SerializableResidue.to_yaml)
yaml.SafeLoader.add_constructor(
    SerializableResidue.yaml_tag, SerializableResidue.from_yaml
)


class BaseCollectiveVariable(openmm.Force, yaml.YAMLObject):
    r"""
    An abstract class with common attributes and method for all CVs.
    """

    _unit: mmunit.Unit = mmunit.dimensionless
    _mass_unit: mmunit.Unit = mmunit.dalton * mmunit.nanometers**2
    _args: t.Dict[str, t.Any] = {}
    _period: t.Optional[float] = None

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self._args

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def __copy__(self):
        return yaml.safe_load(yaml.safe_dump(self))

    def __deepcopy__(self, memo):
        return yaml.safe_load(yaml.safe_dump(self))

    def _registerCV(
        self,
        unit: mmunit.Unit,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
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
        cls = self.__class__
        self.setName(cls.__name__)
        self.setUnit(unit)
        self._mass_unit = mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        arguments, _ = self.getArguments()
        self._args = dict(zip(arguments, args))
        self._args.update(kwargs)

    def _registerPeriod(self, period: float) -> None:
        """
        Register the period of this collective variable.

        This method must called from Subclass.__init__ if the collective variable is
        periodic.

        Parameters
        ----------
            period
                The period of this collective variable
        """
        self._period = period

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
            group: typing.Iterable[int]
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

    def getPeriod(self) -> t.Optional[mmunit.SerializableQuantity]:
        """
        Get the period of this collective variable.

        Returns
        -------
            The period of this collective variable or None if it is not periodic
        """
        if self._period is None:
            return None
        return mmunit.SerializableQuantity(self._period, self.getUnit())

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

    def getValue(self, context: openmm.Context) -> mmunit.Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        .. note::

            This method will be more efficient if the collective variable is the only
            force in its force group (see :OpenMM:`Force`).

        Parameters
        ----------
            context
                The context at which this collective variable should be evaluated

        Returns
        -------
            The value of this collective variable at the given context
        """
        state = get_single_force_state(self, context, getEnergy=True)
        value = value_in_md_units(state.getPotentialEnergy())
        return mmunit.Quantity(value, self.getUnit())

    def getEffectiveMass(self, context: openmm.Context) -> mmunit.Quantity:
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

        .. note::

            This method will be more efficient if the collective variable is the only
            force in its force group (see :OpenMM:`Force`).

        Parameters
        ----------
            context
                The context at which this collective variable's effective mass should be
                evaluated

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
            >>> print(radius_of_gyration.getEffectiveMass(context))
            30.946... Da
        """
        return mmunit.Quantity(compute_effective_mass(self, context), self._mass_unit)
