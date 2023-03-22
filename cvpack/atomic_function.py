"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import openmm
from numpy.typing import ArrayLike
from openmm import unit as mmunit

from .cvpack import (
    AbstractCollectiveVariable,
    QuantityOrFloat,
    UnitOrStr,
    in_md_units,
    str_to_unit,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class AtomicFunction(openmm.CustomCompoundBondForce, AbstractCollectiveVariable):
    """
    A generic function of the coordinates of `m` groups of `n` atoms:

    .. math::

        f({\\bf r}) = \\sum_{i=1}^m F\\left(
            {\\bf r}_{i,1}, {\\bf r}_{i,2}, \\dots, {\\bf r}_{i,n}
        \\right)

    where :math:`F` is a user-defined function and :math:`{\\bf r}_{i,j}` is the position of the
    :math:`j`-th atom of the :math:`i`-th group.

    .. note::

        The function :math:`F` is defined as a string and can be any expression supported by
        :OpenMM:`CustomCompoundBondForce`. It can contain named parameters, which must be passed
        as keyword arguments to the constructor. The parameters can be scalars or arrays of length
        :math:`n`. In the latter case, each value of the array is used for the corresponding group
        of atoms.

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
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> angle1 = cvpack.Angle(0, 5, 10)
        >>> angle2 = cvpack.Angle(10, 15, 20)
        >>> colvar = cvpack.AtomicFunction(
        ...     3,
        ...     '(k/2)*(angle(p1, p2, p3) - theta0)^2',
        ...     [[0, 5, 10], [10, 15, 20]],
        ...     "kilojoules_per_mole",
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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        atomsPerGroup: int,
        function: str,
        indices: ArrayLike,
        unit: UnitOrStr,
        pbc: bool = False,
        **parameters: Union[QuantityOrFloat, Sequence[QuantityOrFloat]],
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
        cv_unit = str_to_unit(unit) if isinstance(unit, str) else unit
        if in_md_units(mmunit.Quantity(1.0, cv_unit)) != 1:
            raise ValueError(f"Unit {cv_unit} is not compatible with the MD unit system.")
        self._registerCV(cv_unit, atomsPerGroup, function, groups, str(unit), pbc)

    @classmethod
    def _fromCustomBondForce(
        cls,
        force: Union[openmm.CustomBondForce, openmm.CustomCompoundBondForce],
        unit: UnitOrStr,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`CustomBondForce` or
        :OpenMM:`CustomCompoundBondForce`.
        """
        is_compound = isinstance(force, openmm.CustomCompoundBondForce)
        atoms_per_group = force.getNumParticlesPerBond() if is_compound else 2
        function = force.getEnergyFunction()
        parameters = {}
        for i in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(i)
            value = force.getGlobalParameterDefaultValue(i)
            parameters[name] = value
        perbond_parameter_names = []
        for i in range(force.getNumPerBondParameters()):
            perbond_parameter_names.append(force.getPerBondParameterName(i))
        for name in perbond_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(force.getNumBonds()):
            if is_compound:
                indices, perbond_parameters = force.getBondParameters(i)
            else:
                *indices, perbond_parameters = force.getBondParameters(i)
            atoms.append(indices)
            for name, value in zip(perbond_parameter_names, perbond_parameters):
                parameters[name].append(value)
        return cls(atoms_per_group, function, atoms, unit, pbc, **parameters)

    @classmethod
    def _fromCustomAngleForce(
        cls,
        force: openmm.CustomAngleForce,
        unit: UnitOrStr,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`CustomAngleForce`.
        """
        function = force.getEnergyFunction()
        parameters = {}
        for i in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(i)
            value = force.getGlobalParameterDefaultValue(i)
            parameters[name] = value
        perangle_parameter_names = []
        for i in range(force.getNumPerAngleParameters()):
            perangle_parameter_names.append(force.getPerAngleParameterName(i))
        for name in perangle_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(force.getNumAngles()):
            *indices, perangle_parameters = force.getAngleParameters(i)
            atoms.append(indices)
            for name, value in zip(perangle_parameter_names, perangle_parameters):
                parameters[name].append(value)
        return cls(3, function, atoms, unit, pbc, **parameters)

    @classmethod
    def _fromCustomTorsionForce(
        cls,
        force: openmm.CustomTorsionForce,
        unit: UnitOrStr,
        pbc: bool = False,
    ) -> Self:
        """
        Create a :class:`AtomicFunction` from an :OpenMM:`CustomTorsionForce`.
        """
        function = force.getEnergyFunction()
        parameters = {}
        for i in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(i)
            value = force.getGlobalParameterDefaultValue(i)
            parameters[name] = value
        pertorsion_parameter_names = []
        for i in range(force.getNumPerTorsionParameters()):
            pertorsion_parameter_names.append(force.getPerTorsionParameterName(i))
        for name in pertorsion_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(force.getNumTorsions()):
            *indices, pertorsion_parameters = force.getTorsionParameters(i)
            atoms.append(indices)
            for name, value in zip(pertorsion_parameter_names, pertorsion_parameters):
                parameters[name].append(value)
        return cls(4, function, atoms, unit, pbc, **parameters)

    @classmethod
    def _fromHarmonicBondForce(
        cls,
        force: openmm.HarmonicBondForce,
        unit: UnitOrStr,
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
        unit: UnitOrStr,
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
        unit: UnitOrStr,
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
            "k*(1+cos(periodicity*angle(p1, p2, p3, p4)-phase))",
            atoms,
            unit,
            pbc,
            **parameters,
        )

    @classmethod
    def fromOpenMMForce(cls, force: openmm.Force, unit: UnitOrStr, pbc: bool = False) -> Self:
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
        """
        if isinstance(force, (openmm.CustomBondForce, openmm.CustomCompoundBondForce)):
            return cls._fromCustomBondForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicBondForce):
            return cls._fromHarmonicBondForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicAngleForce):
            return cls._fromHarmonicAngleForce(force, unit, pbc)
        if isinstance(force, openmm.PeriodicTorsionForce):
            return cls._fromPeriodicTorsionForce(force, unit, pbc)
        if isinstance(force, openmm.CustomAngleForce):
            return cls._fromCustomAngleForce(force, unit, pbc)
        if isinstance(force, openmm.CustomTorsionForce):
            return cls._fromCustomTorsionForce(force, unit, pbc)
        raise TypeError(f"Force {force} is not a CustomCompoundBondForce.")
