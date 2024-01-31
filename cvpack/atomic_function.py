"""
.. class:: AtomicFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the coordinates of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import typing as t

import numpy as np
import openmm
from numpy.typing import ArrayLike

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


def _add_parameters(
    force: openmm.Force,
    size: int,
    **parameters: t.Union[mmunit.ScalarQuantity, t.Sequence[mmunit.ScalarQuantity]],
) -> t.List[t.Union[float, t.Sequence[mmunit.ScalarQuantity]]]:
    perbond_parameters = []
    for name, data in parameters.items():
        if isinstance(data, mmunit.Quantity):
            data = data.value_in_unit_system(mmunit.md_unit_system)
        if isinstance(data, t.Sequence):
            if len(data) != size:
                raise ValueError(
                    f"The length of parameter {name} is {len(data)}. Should be {size}."
                )
            force.addPerBondParameter(name)
            perbond_parameters.append(data)
        else:
            force.addGlobalParameter(name, data)
    return perbond_parameters


class AtomicFunction(openmm.CustomCompoundBondForce, BaseCollectiveVariable):
    r"""
    A generic function of the coordinates of atoms split into `m` groups of `n`
    atoms each:

    .. math::

        f({\bf r}) = \sum_{i=1}^m F\left(
            {\bf r}_{i,1}, {\bf r}_{i,2}, \dots, {\bf r}_{i,n}
        \right)

    where :math:`F` is a user-defined function and :math:`{\bf r}_{i,j}` is the
    coordinate of the :math:`j`-th atom of the :math:`i`-th group.

    The function :math:`F` is defined as a string and can be any expression supported by
    :OpenMM:`CustomCompoundBondForce`. If it contains named parameters, they must be
    passed as keyword arguments to the :class:`AtomicFunction` constructor. The
    parameters can be scalars or arrays of length :math:`m`. In the latter case, each
    value will be assigned to the corresponding group of atoms.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid
        :OpenMM:`CustomCompoundBondForce` expression
    groups
        The indices of the atoms in each group, passed as a 2D array-like object of
        shape `(m, n)`, where `m` is the number of groups and `n` is the number of
        atoms per group. If a 1D object is passed, it is assumed that `m` is 1 and
        `n` is the length of the object.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `unit.dimensionless`
    pbc
        Whether to use periodic boundary conditions

    Keyword Args
    ------------
    **parameters
        The named parameters of the function. Each parameter can be a scalar
        quantity or a 1D array-like object of length `m`, where `m` is the number of
        groups. In the latter case, each entry of the array is used for the
        corresponding group of atoms.

    Raises
    ------
    ValueError
        If the groups are not specified as a 1D or 2D array-like object
    ValueError
        If the unit of the collective variable is not compatible with the MD unit system

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
        ...     "(k/2)*(angle(p1, p2, p3) - theta0)^2",
        ...     unit.kilojoules_per_mole,
        ...     [[0, 5, 10], [10, 15, 20]],
        ...     k = 1000 * unit.kilojoules_per_mole/unit.radian**2,
        ...     theta0 = [np.pi/2, np.pi/3] * unit.radian,
        ... )
        >>> [model.system.addForce(f) for f in [angle1, angle2, colvar]]
        [5, 6, 7]
        >>> integrator = openmm.VerletIntegrator(0)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> theta1 = angle1.getValue(context).value_in_unit(openmm.unit.radian)
        >>> theta2 = angle2.getValue(context).value_in_unit(openmm.unit.radian)
        >>> print(round(500*((theta1 - np.pi/2)**2 + (theta2 - np.pi/3)**2), 3))
        429.479
        >>> print(colvar.getValue(context, digits=6))
        429.479 kJ/mol
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        function: str,
        unit: mmunit.Unit,
        groups: ArrayLike,
        pbc: bool = True,
        **parameters: t.Union[mmunit.ScalarQuantity, t.Sequence[mmunit.ScalarQuantity]],
    ) -> None:
        groups = np.atleast_2d(groups)
        num_groups, atoms_per_group, *others = groups.shape
        if others:
            raise ValueError("Array `groups` cannot have more than 2 dimensions")
        super().__init__(atoms_per_group, function)
        perbond_parameters = _add_parameters(self, num_groups, **parameters)
        for group, *values in zip(groups, *perbond_parameters):
            self.addBond(group, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._checkUnitCompatibility(unit)
        unit = mmunit.SerializableUnit(unit)
        self._registerCV(unit, function, unit, groups, pbc, **parameters)

    @classmethod
    def _fromCustomForce(
        cls,
        force: t.Union[
            openmm.CustomAngleForce,
            openmm.CustomBondForce,
            openmm.CustomCompoundBondForce,
            openmm.CustomExternalForce,
            openmm.CustomTorsionForce,
        ],
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> "AtomicFunction":
        """
        Create a :class:`AtomicFunction` from an object of :openmm:`CustomBondForce`,
        :openmm:`CustomAngleForce`, :openmm:`CustomTorsionForce`, or
        :openmm:`CustomCompoundBondForce` class.
        """
        if isinstance(force, openmm.CustomExternalForce):
            number, item, definition = 1, "Particle", "; x=x1; y=y1; z=z1"
        elif isinstance(force, openmm.CustomBondForce):
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
            per_item_parameter_names.append(
                getattr(force, f"getPer{item}ParameterName")(i)
            )
        for name in per_item_parameter_names:
            parameters[name] = []
        atoms = []
        for i in range(getattr(force, f"getNum{item}s")()):
            if isinstance(force, openmm.CustomCompoundBondForce):
                indices, per_item_parameters = force.getBondParameters(i)
            else:
                *indices, per_item_parameters = getattr(
                    force,
                    f"get{item}Parameters",
                )(i)
            atoms.append(indices)
            for name, value in zip(per_item_parameter_names, per_item_parameters):
                parameters[name].append(value)
        atoms = np.asarray(atoms).reshape(-1, number)
        return cls(function, unit, atoms, pbc, **parameters)

    @classmethod
    def _fromHarmonicBondForce(
        cls,
        force: openmm.HarmonicBondForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> "AtomicFunction":
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
        return cls("(k/2)*(distance(p1, p2)-r0)^2", unit, atoms, pbc, **parameters)

    @classmethod
    def _fromHarmonicAngleForce(
        cls,
        force: openmm.HarmonicAngleForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> "AtomicFunction":
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
        return cls("(k/2)*(angle(p1, p2, p3)-theta0)^2", unit, atoms, pbc, **parameters)

    @classmethod
    def _fromPeriodicTorsionForce(
        cls,
        force: openmm.PeriodicTorsionForce,
        unit: mmunit.Unit,
        pbc: bool = False,
    ) -> "AtomicFunction":
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
            "k*(1 + cos(periodicity*dihedral(p1, p2, p3, p4) - phase))",
            unit,
            atoms,
            pbc,
            **parameters,
        )

    @classmethod
    def fromOpenMMForce(
        cls, force: openmm.Force, unit: mmunit.Unit, pbc: bool = False
    ) -> "AtomicFunction":
        """
        Create an :class:`AtomicFunction` from an :OpenMM:`Force`.

        Parameters
        ----------
            force
                The force to be converted
            unit
                The unit of measurement of the collective variable. It must be
                compatible with the MD unit system (mass in `daltons`, distance in
                `nanometers`, time in `picoseconds`, temperature in `kelvin`, energy in
                `kilojoules_per_mol`, angle in `radians`). If the collective variables
                does not have a unit, use `unit.dimensionless`
            pbc
                Whether to use periodic boundary conditions

        Raises
        ------
            TypeError
                If the force is not convertible to an :class:`AtomicFunction`

        Examples
        --------
        >>> import cvpack
        >>> import numpy as np
        >>> import openmm
        >>> from cvpack import unit
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> residues = [r for r in model.topology.residues() if 59 <= r.index <= 79]
        >>> helix_content = cvpack.HelixTorsionContent(residues)
        >>> model.system.addForce(helix_content)
        6
        >>> num_atoms = model.system.getNumParticles()
        >>> mean_x = openmm.CustomExternalForce("x/num_atoms")
        >>> mean_x.addGlobalParameter("num_atoms", num_atoms)
        0
        >>> for i in range(num_atoms):
        ...     _ = mean_x.addParticle(i, [])
        >>> model.system.addForce(mean_x)
        7
        >>> forces = {force.getName(): force for force in model.system.getForces()}
        >>> copies = {
        ...     name: cvpack.AtomicFunction.fromOpenMMForce(
        ...         force, unit.kilojoules_per_mole
        ...     )
        ...     for name, force in forces.items()
        ...     if name in [
        ...         "HarmonicBondForce",
        ...         "HarmonicAngleForce",
        ...         "PeriodicTorsionForce",
        ...         "CustomExternalForce"
        ...     ]
        ... }
        >>> copies["HelixTorsionContent"] = cvpack.AtomicFunction.fromOpenMMForce(
        ...     helix_content, unit.dimensionless
        ... )
        >>> indices = {}
        >>> for index, (name, force) in enumerate(copies.items(), start=1):
        ...     _ = model.system.addForce(force)
        ...     force.setForceGroup(index)
        ...     indices[name] = index
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> for name in copies:
        ...    state = context.getState(getEnergy=True, groups={indices[name]})
        ...    value = state.getPotentialEnergy() / unit.kilojoules_per_mole
        ...    copy_value = copies[name].getValue(context, digits=6)
        ...    print(f"{name}: original={value:.6f}, copy={copy_value}")
        HarmonicBondForce: original=2094.312483, copy=2094.312 kJ/mol
        HarmonicAngleForce: original=3239.795215, copy=3239.795 kJ/mol
        PeriodicTorsionForce: original=4226.051934, copy=4226.052 kJ/mol
        CustomExternalForce: original=5.021558, copy=5.021558 kJ/mol
        HelixTorsionContent: original=17.452849, copy=17.45285 dimensionless
        """
        if isinstance(
            force,
            (
                openmm.CustomAngleForce,
                openmm.CustomBondForce,
                openmm.CustomCompoundBondForce,
                openmm.CustomExternalForce,
                openmm.CustomTorsionForce,
            ),
        ):
            return cls._fromCustomForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicBondForce):
            return cls._fromHarmonicBondForce(force, unit, pbc)
        if isinstance(force, openmm.HarmonicAngleForce):
            return cls._fromHarmonicAngleForce(force, unit, pbc)
        if isinstance(force, openmm.PeriodicTorsionForce):
            return cls._fromPeriodicTorsionForce(force, unit, pbc)
        raise TypeError(f"Force {force} is not convertible to an AtomicFunction")
