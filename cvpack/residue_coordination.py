"""
.. class:: Distance
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two groups of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm.app.topology import Residue

from .cvpack import unit as mmunit
from .cvpack import AbstractCollectiveVariable, SerializableResidue


class ResidueCoordinationNumber(openmm.CustomCentroidBondForce, AbstractCollectiveVariable):
    """
    The number of contacts between two atom groups:

    .. math::
        N({\\bf r}) = \\sum_{i \\in {\\bf g}_1} \\sum_{j \\in {\\bf g}_2}
                        S\\left(\\frac{\\|{\\bf r}_j - {\\bf r}_i\\|}{r_0}\\right)

    where :math:`r_0` is the threshold distance for defining a contact and :math:`S(x)`
    is a step function equal to :math:`1` if a contact is made or equal to :math:`0`
    otherwise. For trajectory analysis, it is fine to make :math:`S(x) = H(1-x)`, where
    `H` is the `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. For molecular dynamics,
    however, :math:`S(x)` should be a continuous approximation of :math:`H(1-x)` for
    :math:`x \\geq 0`. By default :cite:`Iannuzzi_2003`, the following function is used:

    .. math::

        S(x) = \\frac{1-x^6}{1-x^{12}} = \\frac{1}{1+x^6}

    In fact, a cutoff distance :math:`r_c = x_c r_0` (typically, :math:`x_c = 2`) is
    applied so that :math:`S(x) = 0` for :math:`x \\geq x_c`. To avoid discontinuities,
    there is also the option to smoothly switch off :math:`S(x)` starting from
    :math:`r_s = x_s r_0` (typically, :math:`x_s = 1.5`) instead of doing it abruptly at
    :math:`r_c`.

    .. note::

        Atoms are allowed to be in both groups. In this case, self-contacts
        (:math:`i = j`) are ignored and each pair of distinct atoms (:math:`i \\neq j`)
        is counted only once.

    Parameters
    ----------
        group1
            The indices of the atoms in the first group
        group2
            The indices of the atoms in the second group
        numAtoms
            The total number of atoms in the system (required by OpenMM)
        pbc
            Whether the system has periodic boundary conditions
        stepFunction
            The function "step(1-x)" (for analysis only) or a continuous approximation
            thereof
        thresholdDistance
            The threshold distance (:math:`r_0`) for considering two atoms as being in
            contact
        cutoffFactor
            The factor :math:`x_c` that multiplies the threshold distance to define
            the cutoff distance
        switchFactor
            The factor :math:`x_s` that multiplies the threshold distance to define
            the distance at which the step function starts switching off smoothly.
            If None, it switches off abruptly at the cutoff distance.

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> carbons = [
        ...     a.index
        ...     for a in model.topology.atoms()
        ...     if a.element == app.element.carbon
        ... ]
        >>> num_atoms = model.topology.getNumAtoms()
        >>> optionals = {"pbc": False, "stepFunction": "step(1-x)"}
        >>> nc = cvpack.NumberOfContacts(
        ...     carbons, carbons, num_atoms, **optionals
        ... )
        >>> model.system.addForce(nc)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(
        ...     model.system, openmm.CustomIntegrator(0), platform
        ... )
        >>> context.setPositions(model.positions)
        >>> print(nc.getValue(context, digits=6))
        6.0 dimensionless

    """
    @mmunit.convert_quantities
    def __init__(
        self,
        residueGroup1: t.Sequence[Residue],
        residueGroup2: t.Sequence[Residue],
        pbc: bool = True,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: mmunit.ScalarQuantity = mmunit.Quantity(1.0, mmunit.nanometers),
        normalize: bool = False,
        weighByMass: bool = True,
    ) -> None:
        nr1 = len(residueGroup1)
        nr2 = len(residueGroup2)
        energy = f"({stepFunction})/{nr1 * nr2}" if normalize else stepFunction
        super().__init__(2, energy + f"; x=distance(g1,g2)/{thresholdDistance}")
        self.setUsesPeriodicBoundaryConditions(pbc)
        for residue in residueGroup1 + residueGroup2:
            self.addGroup(
                [atom.index for atom in residue.atoms()],
                *([] if weighByMass else [[1] * len(residue)]),
            )
        for idx1 in range(nr1):
            for idx2 in range(nr1, nr1 + nr2):
                self.addBond([idx1, idx2], [])
        self._registerCV(
            mmunit.dimensionless,
            list(map(SerializableResidue, residueGroup1)),
            list(map(SerializableResidue, residueGroup2)),
            pbc,
            stepFunction,
            thresholdDistance,
            normalize,
        )
