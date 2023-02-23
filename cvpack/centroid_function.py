"""
.. class:: CentroidFunction
   :platform: Linux, MacOS, Windows
   :synopsis: A generic function of the centroids of groups of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import openmm

from .cvpack import AbstractCollectiveVariable, UnitOrStr, in_md_units, str_to_unit


class CentroidFunction(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    A generic function of the centroids of `n` groups of atoms:

    .. math::

        f({\\bf r}) = F({\\bf R}_1, {\\bf R}_2, \\dots, {\\bf R}_n)

    where :math:`F` is a user-defined function and :math:`{\\bf R}_i` is the centroid of the
    :math:`i`-th group of atoms. The function :math:`F` is defined as a string and can be any valid
    :OpenMM:`CustomCentroidBondForce` expression.

    The centroid of a group of atoms is defined as:

    .. math::

        {\\bf R}_i({\\bf r}) = \\frac{1}{n_i} \\sum_{j=1}^{n_i} {\\bf r}_{j, i}

    where :math:`n_i` is the number of atoms in group :math:`i` and :math:`{\\bf r}_{j, i}` is the
    position of the :math:`j`-th atom in group :math:`i`. Optionally, the centroid can be weighted
    by the mass of each atom in the group. In this case, it is defined as:

    .. math::

        {\\bf R}_i({\\bf r}) = \\frac{1}{M_i} \\sum_{j=1}^{n_i} m_{j, i} {\\bf r}_{j, i}

    where :math:`M_i` is the total mass of group :math:`i` and :math:`m_{j, i}` is the mass of the
    :math:`j`-th atom in group :math:`i`.

    Parameters
    ----------
        function
            The function to be evaluated. It must be a valid :OpenMM:`CustomCentroidBondForce`
            expression
        groups
            The groups of atoms to be used in the function. Each group must be a list of atom
            indices
        unit
            The unit of measurement of the collective variable. It must be compatible with the
            MD unit system (mass in `daltons`, distance in `nanometers`, time in `picoseconds`,
            temperature in `kelvin`, energy in `kilojoules_per_mol`, angle in `radians`). If
            the collective variables does not have a unit, use `dimensionless`
        pbc
            Whether to use periodic boundary conditions

    Raises
    ------
        ValueError
            If the collective variable is not compatible with the MD unit system

    Example
    -------
        >>> import cvpack
        >>> import openmm as mm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> atoms = list(range(num_atoms))
        >>> rg = cvpack.RadiusOfGyration(atoms)
        >>> definitions = [f'd{i+1} = distance(g{i+1}, g{num_atoms+1})' for i in atoms]
        >>> sum_dist_sq = "+".join(f'd{i+1}^2' for i in atoms)
        >>> function = ";".join([f"sqrt(({sum_dist_sq})/{num_atoms})"] + definitions)
        >>> groups = [[i] for i in atoms] + [atoms]
        >>> colvar = cvpack.CentroidFunction(function, groups, "nanometers")
        >>> [model.system.addForce(f) for f in [rg, colvar]]
        [5, 6]
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(rg.getValue(context, digits=6))
        0.295143 nm
        >>> print(colvar.getValue(context, digits=6))
        0.295143 nm
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        function: str,
        groups: Iterable[Iterable[int]],
        unit: UnitOrStr,
        pbc: bool = False,
        weighByMass: bool = False,
    ) -> None:
        num_groups = len(groups)
        super().__init__(num_groups, function)
        for group in groups:
            self.addGroup(group, None if weighByMass else [1] * len(group))
        self.addBond(list(range(num_groups)), [])
        self.setUsesPeriodicBoundaryConditions(pbc)
        cv_unit = str_to_unit(unit) if isinstance(unit, str) else unit
        if in_md_units(1 * cv_unit) != 1:
            raise ValueError(f"Unit {cv_unit} is not compatible with the MD unit system.")
        self._registerCV(cv_unit, function, groups, str(unit), pbc)
