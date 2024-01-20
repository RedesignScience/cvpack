"""
.. class:: CentroidFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the centroids of groups of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm
from numpy.typing import ArrayLike

from cvpack import unit as mmunit

from .atomic_function import _add_parameters
from .cvpack import AbstractCollectiveVariable


class CentroidFunction(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    A generic function of the centroids of :math:`m \\times n` atoms groups split
    into `m` collections of `n` groups each:

    .. math::

        f({\\bf r}) = \\sum_{i=1}^m F\\Big(
            {\\bf g}^i_1({\\bf r}),
            {\\bf g}^i_2({\\bf r}),
            \\dots,
            {\\bf g}^i_n({\\bf r})
        \\Big)

    where :math:`F` is a user-defined function and :math:`{\\bf g}^i_1({\\bf r})` is the
    centroid of the :math:`j`-th group of atoms of the :math:`i`-th collection of
    groups.

    The function :math:`F` is defined as a string and can be any expression supported
    by :OpenMM:`CustomCentroidBondForce`. If it contains named parameters, they must
    be passed as keyword arguments to the :class:`CentroidFunction` constructor. The
    parameters can be scalars or arrays of length :math:`m`. In the latter case, each
    value will be assigned to the corresponding collection of atom groups.

    The centroid of a group of atoms is defined as:

    .. math::

        {\\bf g}_j({\\bf r}) = \\frac{1}{N_j} \\sum_{k=1}^{N_j} {\\bf r}_{k,j}

    where :math:`N_j` is the number of atoms in group :math:`j` and
    :math:`{\\bf r}_{k,j}` is the coordinate of the :math:`k`-th atom of group
    :math:`j`. Optionally, the centroid can be weighted by the mass of each atom
    in the group. In this case, it is redefined as:

    .. math::

        {\\bf g}_j({\\bf r}) = \\frac{1}{M_j} \\sum_{k=1}^{N_j} m_{k,j} {\\bf r}_{k,j}

    where :math:`M_j` is the total mass of atom group :math:`j` and :math:`m_{k,j}` is
    the mass of the :math:`k`-th atom in group :math:`j`.

    Parameters
    ----------
    function
        The function to be evaluated. It must be a valid
        :OpenMM:`CustomCentroidBondForce` expression
    groups
        The groups of atoms to be used in the function. Each group must be specified
        as a list of atom indices with arbitrary length
    collections
        The indices of the groups in each collection, passed as a 2D array-like object
        of shape `(m, n)`, where `m` is the number of collections and `n` is the number
        groups per collection. If a 1D object is passed, it is assumed that `m` is 1 and
        `n` is the length of the object.
    unit
        The unit of measurement of the collective variable. It must be compatible
        with the MD unit system (mass in `daltons`, distance in `nanometers`, time
        in `picoseconds`, temperature in `kelvin`, energy in `kilojoules_per_mol`,
        angle in `radians`). If the collective variables does not have a unit, use
        `dimensionless`
    pbc
        Whether to use periodic boundary conditions
    weighByMass
        Whether to define the centroid as the center of mass of the group instead of
        the geometric center

    Keyword Args
    ------------
    **parameters
        The named parameters of the function. Each parameter can be a scalar
        quantity or a 1D array-like object of length `m`, where `m` is the number of
        group collections. In the latter case, each entry of the array is used for
        the corresponding collection of groups.

    Raises
    ------
    ValueError
        If the collections are not specified as a 1D or 2D array-like object
    ValueError
        If group indices are out of bounds
    ValueError
        If the unit of the collective variable is not compatible with the MD unit system

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> atoms = list(range(num_atoms))
        >>> groups = [[i] for i in atoms]  # Each atom is a group
        >>> groups.append(atoms)  # The whole molecule is also a group
        >>> sum_dist_sq = "+".join(
        ...     f'distance(g{i+1}, g{num_atoms+1})^2' for i in atoms
        ... )
        >>> function = f"sqrt(({sum_dist_sq})/n)"  # The radius of gyration
        >>> colvar = cvpack.CentroidFunction(
        ...     function, unit.nanometers, groups, n=num_atoms,
        ... )
        >>> model.system.addForce(colvar)
        5
        >>> integrator = openmm.VerletIntegrator(0)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(colvar.getValue(context, digits=6))
        0.2951431 nm
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        function: str,
        unit: mmunit.Unit,
        groups: t.Sequence[t.Sequence[int]],
        collections: t.Optional[ArrayLike] = None,
        pbc: bool = False,
        weighByMass: bool = False,
        **parameters: mmunit.ScalarQuantity,
    ) -> None:
        collections = np.atleast_2d(
            np.arange(len(groups)) if collections is None else collections
        )
        num_collections, groups_per_collection, *others = collections.shape
        if others:
            raise ValueError("Array `collections` cannot have more than 2 dimensions")
        num_groups = len(groups)
        if np.any(collections < 0) or np.any(collections >= num_groups):
            raise ValueError("Group index out of bounds")
        super().__init__(groups_per_collection, function)
        for group in groups:
            self.addGroup(group, None if weighByMass else [1] * len(group))
        perbond_parameters = _add_parameters(self, num_collections, **parameters)
        for collection, *values in zip(collections, *perbond_parameters):
            self.addBond(collection, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._checkUnitCompatibility(unit)
        unit = mmunit.SerializableUnit(unit)
        self._registerCV(
            unit, function, unit, groups, collections, pbc, weighByMass, **parameters
        )
