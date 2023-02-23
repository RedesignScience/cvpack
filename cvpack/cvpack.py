"""
.. module:: cvpack
   :platform: Linux, MacOS, Windows
   :synopsis: Useful Collective Variables for OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import ast
import inspect
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

QuantityOrFloat = Union[mmunit.Quantity, float]
UnitOrStr = Union[mmunit.Unit, str]


def in_md_units(quantity: QuantityOrFloat) -> float:
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


def str_to_unit(unitStr: str) -> mmunit.Unit:
    """
    Returns an OpenMM unit of measurement from its string representation.

    Parameters
    ----------
        unitStr
            The string representation of the unit to be converted

    Returns
    -------
        The OpenMM unit of measurement

    """

    class NodeTransformer(ast.NodeTransformer):
        """
        A child class of ast.NodeTransformer that replaces all instances of ast.Name with
        an ast.Attribute with the value "mmunit" and the attribute name equal to the original
        id of the ast.Name.

        """

        def visit_Name(self, node: ast.Name) -> ast.Attribute:  # pylint: disable=invalid-name
            """
            Replace an instance of ast.Name with an ast.Attribute with the value "mmunit" and
            the attribute name equal to the original id of the ast.Name.

            """
            mod = ast.Name(id="mmunit", ctx=ast.Load())
            return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())

    tree = NodeTransformer().visit(ast.parse(unitStr, mode="eval"))
    return eval(  # pylint: disable=eval-used
        compile(ast.fix_missing_locations(tree), "", mode="eval")
    )


class SerializableResidue(mmapp.topology.Residue):
    """
    A class that extends OpenMM's Residue class with additional methods for serialization and
    deserialization.

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

    _unit = mmunit.dimensionless
    _args = {}

    def __getstate__(self) -> Dict[str, Any]:
        return self._args

    def __setstate__(self, keywords: Dict[str, Any]) -> None:
        self.__init__(**keywords)

    def _registerCV(self, unit: mmunit.Unit, *args: Any) -> None:
        """
        Register the newly created AbstractCollectiveVariable subclass instance.

        This method must always be called from Subclass.__init__.

        Parameters
        ----------
            unit
                The unit of measurement of this collective variable. It must be a unit in the MD
                unit system (mass in Da, distance in nm, time in ps, temperature in K, energy in
                kJ/mol, angle in rad).
            args
                The arguments needed to construct this collective variable

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
                If this force is not part of the system in the given context

        """
        forces = context.getSystem().getForces()
        if not any(force.this == self.this for force in forces):
            raise RuntimeError("This force is not part of the system in the given context.")
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
        Inspect the arguments needed for constructing an instance of this collective variable.

        Returns
        -------
            A dictionary with the type annotations of all arguments

            A dictionary with the default values of optional arguments

        Example
        -------
            >>> import cvpack
            >>> args, defaults = cvpack.RadiusOfGyration.getArguments()
            >>> print(*args.items())
            ('group', typing.Iterable[int]) ('pbc', <class 'bool'>) ('weighByMass', <class 'bool'>)
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

    def getValue(self, context: openmm.Context, digits: Optional[int] = None) -> mmunit.Quantity:
        """
        Evaluate this collective variable at a given :OpenMM:`Context`.

        Optionally, the value can be rounded to a specified number of digits.

        Parameters
        ----------
            context
                The context at which this collective variable should be evaluated
            digits
                The number of digits to round to

        Returns
        -------
        """
        state = self._getSingleForceState(context, getEnergy=True)
        value = in_md_units(state.getPotentialEnergy())
        return (round(value, digits) if digits else value) * self.getUnit()

    def getEffectiveMass(
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

        Example
        -------
            >>> import cvpack
            >>> import openmm as mm
            >>> from openmmtools import testsystems
            >>> model = testsystems.AlanineDipeptideImplicit()
            >>> peptide = [a.index for a in model.topology.atoms() if a.residue.name != 'HOH']
            >>> radius_of_gyration = cvpack.RadiusOfGyration(peptide)
            >>> model.system.addForce(radius_of_gyration)
            6
            >>> platform = mm.Platform.getPlatformByName('Reference')
            >>> context = mm.Context(model.system, mm.VerletIntegrator(0), platform)
            >>> context.setPositions(model.positions)
            >>> print(radius_of_gyration.getEffectiveMass(context, digits=6))
            30.946932 Da

        """
        state = self._getSingleForceState(context, getForces=True)
        force_values = in_md_units(state.getForces(asNumpy=True))
        indices = np.arange(context.getSystem().getNumParticles())
        masses_with_units = map(context.getSystem().getParticleMass, indices)
        mass_values = np.array(list(map(in_md_units, masses_with_units)))
        effective_mass = 1.0 / np.sum(np.sum(force_values**2, axis=1) / mass_values)
        unit = mmunit.dalton * (mmunit.nanometers / self.getUnit()) ** 2
        return (round(effective_mass, digits) if digits else effective_mass) * unit
