"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import sys
from typing import Sequence, Union, get_args

import numpy as np
import openmm
from numpy.typing import ArrayLike

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

CustomForce = Union[
    openmm.CustomBondForce,
    openmm.CustomAngleForce,
    openmm.CustomTorsionForce,
    openmm.CustomCompoundBondForce,
]


class AtomicFunction(openmm.CustomCompoundBondForce, AbstractCollectiveVariable):
    """
    A generic function of the coordinates of `m` groups of `n` atoms:

    .. math::

        f({\\bf r}) = \\sum_{i=1}^m F\\left(
            {\\bf r}_{i,1}, {\\bf r}_{i,2}, \\dots, {\\bf r}_{i,n}
        \\right)

    where :math:`F` is a user-defined function and :math:`{\\bf r}_{i,j}` is the position of the
    :math:`j`-th atom of the :math:`i`-th group.

    The function :math:`F` is defined as a string and can be any expression supported by
    :OpenMM:`CustomCompoundBondForce`. If it contains named parameters, they must be passed as
    keyword arguments to the :class:`AtomicFunction` constructor. The parameters can be scalars
    or arrays of length :math:`m`. In the latter case, each value will be assigned to the
    corresponding group of atoms.

    Parameters
    ----------
        atomsPerGroup
            The number of atoms in each group
        function
            The function to be evaluated. It must be a valid :OpenMM:`CustomCompoundBondForce`
            expression
        atoms
            The indices of the atoms in each group, passed as a 2D array-like object of shape
            `(m, n)`, where `m` is the number of groups and `n` is the number of atoms per group,
            or as a 1D array-like object of length `m*n`, where the first `n` elements are the
            indices of the atoms in the first group, the next `n` elements are the indices of the
            atoms in the second group, and so on.
        unit
            The unit of measurement of the collective variable. It must be compatible with the
            MD unit system (mass in `daltons`, distance in `nanometers`, time in `picoseconds`,
            temperature in `kelvin`, energy in `kilojoules_per_mol`, angle in `radians`). If the
            collective variables does not have a unit, use `unit.dimensionless`
        pbc
            Whether to use periodic boundary conditions
        **parameters
            The named parameters of the function. Each parameter can be a scalar quantity or a 1D
            array-like object of length `m`, where `m` is the number of groups. In the latter
            case, each entry of the array is used for the corresponding group of atoms.

    Raises
    ------
        ValueError
            If the collective variable is not compatible with the MD unit system

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> import numpy as np
        >>> from cvpack import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> angle1 = cvpack.Angle(0, 5, 10)
        >>> angle2 = cvpack.Angle(10, 15, 20)
        >>> colvar = cvpack.AtomicFunction(
        ...     3,
        ...     "(k/2)*(angle(p1, p2, p3) - theta0)^2",
        ...     [[0, 5, 10], [10, 15, 20]],
        ...     unit.kilojoules_per_mole,
        ...     k = 1000 * unit.kilojoules_per_mole/unit.radian**2,
        ...     theta0 = [np.pi/2, np.pi/3] * unit.radian,
        ... )
        >>> [model.system.addForce(f) for f in [angle1, angle2, colvar]]
        [5, 6, 7]
        >>> integrator =openmm.VerletIntegrator(0)
        >>> platform =openmm.Platform.getPlatformByName('Reference')
        >>> context =openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> theta1 = angle1.getValue(context).value_in_unit(openmm.unit.radian)
        >>> theta2 = angle2.getValue(context).value_in_unit(openmm.unit.radian)
        >>> print(round(500*((theta1 - np.pi/2)**2 + (theta2 - np.pi/3)**2), 6))
        429.479028
        >>> print(colvar.getValue(context, digits=6))
        429.479028 kJ/mol
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        atomsPerGroup: int,
        function: str,
        indices: ArrayLike,
        unit: mmunit.Unit,
        pbc: bool = False,
        **parameters: Union[mmunit.ScalarQuantity, Sequence[mmunit.ScalarQuantity]],
    ) -> None:
        groups = np.asarray(indices, dtype=np.int32).reshape(-1, atomsPerGroup)
        super().__init__(atomsPerGroup, function)
        perbond_parameters = []
        for name, data in parameters.items():
            if isinstance(data, mmunit.Quantity):
                data = data.value_in_unit_system(mmunit.md_unit_system)
            if isinstance(data, Sequence):
                self.addPerBondParameter(name)
                perbond_parameters.append(data)
            else:
                self.addGlobalParameter(name, data)
        for group, *values in zip(groups, *perbond_parameters):
            self.addBond(group, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
        if mmunit.Quantity(1, unit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")
        self._registerCV(
            unit, atomsPerGroup, function, groups, mmunit.SerializableUnit(unit), pbc, **parameters
        )

    @classmethod
    def _fromCustomForce(
        cls,
        force: CustomForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an object of :openmm:`CustomBondForce`,
        :openmm:`CustomAngleForce`, :openmm:`CustomTorsionForce`, or
        :openmm:`CustomCompoundBondForce` class.
        """
        if isinstance(force, openmm.CustomBondForce):
            number, item, definition = 2, "Bond", "; r=distance(p1, p2)"
        elif isinstance(force, openmm.CustomAngleForce):
            number, item, definition = 3, "Angle", "; theta=angle(p1, p2, p3)"
        elif isinstance(force, openmm.CustomTorsionForce):
            number, item, definition = 4, "Torsion", "; theta=dihedral(p1, p2, p3, p4)"
        elif isinstance(force, openmm.CustomCompoundBondForce):
            number, item, definition = force.getNumParticlesPerBond(), "Bond", ""
        function = force.getEnergyFunction() + definition
        parameters = {}
        for i in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(i)
            value = force.getGlobalParameterDefaultValue(i)
            parameters[name] = value
        per_item_parameter_names = []
        for i in range(getattr(force, f"getNumPer{item}Parameters")()):
            per_item_parameter_names.append(getattr(force, f"getPer{item}ParameterName")(i))
        for name in per_item_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(getattr(force, f"getNum{item}s")()):
            if isinstance(force, openmm.CustomCompoundBondForce):
                indices, per_item_parameters = force.getBondParameters(i)
            else:
                *indices, per_item_parameters = getattr(force, f"get{item}Parameters")(i)
            atoms.append(indices)
            for name, value in zip(per_item_parameter_names, per_item_parameters):
                parameters[name].append(value)
        return cls(number, function, atoms, unit, pbc, **parameters)

    @classmethod
    def _fromHarmonicBondForce(
        cls,
        force: openmm.HarmonicBondForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`HarmonicBondForce`.
        """
        parameters = {"r0": [], "k": []}
        atoms = []
        for i in range(force.getNumBonds()):
            *indices, length, k = force.getBondParameters(i)
            atoms.append(indices)
            parameters["r0"].append(length)
            parameters["k"].append(k)
        return cls(2, "(k/2)*(distance(p1, p2)-r0)^2", atoms, unit, pbc, **parameters)

    @classmethod
    def _fromHarmonicAngleForce(
        cls,
        force: openmm.HarmonicAngleForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`HarmonicAngleForce`.
        """
        parameters = {"theta0": [], "k": []}
        atoms = []
        for i in range(force.getNumAngles()):
            *indices, angle, k = force.getAngleParameters(i)
            atoms.append(indices)
            parameters["theta0"].append(angle)
            parameters["k"].append(k)
        return cls(3, "(k/2)*(angle(p1, p2, p3)-theta0)^2", atoms, unit, pbc, **parameters)

    @classmethod
    def _fromPeriodicTorsionForce(
        cls,
        force: openmm.PeriodicTorsionForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`PeriodicTorsionForce`.
        """
        parameters = {"periodicity": [], "phase": [], "k": []}
        atoms = []
        for i in range(force.getNumTorsions()):
            *indices, periodicity, phase, k = force.getTorsionParameters(i)
            atoms.append(indices)
            parameters["periodicity"].append(periodicity)
            parameters["phase"].append(phase)
            parameters["k"].append(k)
        return cls(
            4,
            "k*(1 + cos(periodicity*dihedral(p1, p2, p3, p4) - phase))",
            atoms,
            unit,
            pbc,
            **parameters,
        )

    @classmethod
    def fromOpenMMForce(cls, force: openmm.Force, unit: mmunit.Unit, pbc: bool = False) -> Self:
        """
        Create an :class:`AtomicFunction` from an :OpenMM:`Force`.

        Parameters
        ----------
            force
                The force to be converted
            unit
                The unit of measurement of the collective variable. It must be compatible with the
                MD unit system (mass in `daltons`, distance in `nanometers`, time in `picoseconds`,
                temperature in `kelvin`, energy in `kilojoules_per_mol`, angle in `radians`). If
                the collective variables does not have a unit, use `unit.dimensionless`
            pbc
                Whether to use periodic boundary conditions

        Raises
        ------
            TypeError
                If the force is not convertible to an :class:`AtomicFunction`

        Examples
        --------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app
        >>> from cvpack import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> residues = [r for r in model.topology.residues() if 59 <= r.index <= 79]
        >>> helix_content = cvpack.HelixTorsionContent(residues)
        >>> model.system.addForce(helix_content)
        6
        >>> forces = {force.getName(): force for force in model.system.getForces()}
        >>> copies = {
        ...     name: cvpack.AtomicFunction.fromOpenMMForce(force, unit.kilojoules_per_mole)
        ...     for name, force in forces.items()
        ...     if name in ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]
        ... }
        >>> copies["HelixTorsionContent"] = cvpack.AtomicFunction.fromOpenMMForce(
        ...     helix_content, unit.dimensionless
        ... )
        >>> [model.system.addForce(force) for force in copies.values()]
        [7, 8, 9, 10]
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> for name, force in copies.items():
        ...    forces[name].setForceGroup(1)
        ...    state = context.getState(getEnergy=True, groups={1})
        ...    value = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ...    print(f"{name}: original={value}, copy={force.getValue(context, digits=6)}")
        ...    forces[name].setForceGroup(0)
        HarmonicBondForce: original=2094.3124828597124, copy=2094.312483 kJ/mol
        HarmonicAngleForce: original=3239.795214705211, copy=3239.795215 kJ/mol
        PeriodicTorsionForce: original=4226.051934121031, copy=4226.051934 kJ/mol
        HelixTorsionContent: original=17.452848762681054, copy=17.452849 dimensionless
        """
        if isinstance(force, get_args(CustomForce)):
            return cls._fromCustomForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicBondForce):
            return cls._fromHarmonicBondForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicAngleForce):
            return cls._fromHarmonicAngleForce(force, unit, pbc)
        if isinstance(force, openmm.PeriodicTorsionForce):
            return cls._fromPeriodicTorsionForce(force, unit, pbc)
        raise TypeError(f"Force {force} is not a CustomCompoundBondForce.")
