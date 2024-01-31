"""
.. class:: Distance
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two groups of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm.app.topology import Residue

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable, SerializableResidue


class ResidueCoordination(openmm.CustomCentroidBondForce, BaseCollectiveVariable):
    r"""
    The number of contacts between two disjoint groups of residues:

    .. math::
        N({\bf r}) = \sum_{i \in {\bf G}_1} \sum_{j \in {\bf G}_2} S\left(
            \frac{\|{\bf R}_j({\bf r}) - {\bf R}_i({\bf r})\|}{r_0}
        \right)

    where :math:`{\bf G}_1` and :math:`{\bf G}_2` are the two groups of residues,
    :math:`{\bf R}_i` is the centroid of the residue :math:`i`, :math:`r_0` is the
    threshold distance for defining a contact and :math:`S(x)` is a step function equal
    to :math:`1` if a contact is made or equal to :math:`0` otherwise. For trajectory
    analysis, it is fine to make :math:`S(x) = H(1-x)`, where `H` is the `Heaviside step
    function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. For molecular
    dynamics, however, :math:`S(x)` should be a continuous approximation of
    :math:`H(1-x)` for :math:`x \geq 0`. By default :cite:`Iannuzzi_2003`, the
    following function is used:

    .. math::

        S(x) = \frac{1-x^6}{1-x^{12}} = \frac{1}{1+x^6}

    Parameters
    ----------
        residueGroup1
            The residues in the first group
        residueGroup2
            The residues in the second group
        pbc
            Whether the system has periodic boundary conditions
        stepFunction
            The function "step(1-x)" (for analysis only) or a continuous approximation
            thereof
        thresholdDistance
            The threshold distance (:math:`r_0`) for considering two residues as being
            in contact
        normalize
            Whether the number of contacts should be normalized by the total number of
            possible contacts
        weighByMass
            Whether the centroid of each residue should be weighted by the mass of the
            atoms in the residue

    Raises
    ------
        ValueError
            If the two groups of residues are not disjoint

    Example
    -------
        >>> import cvpack
        >>> import itertools as it
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> group1 = list(it.islice(model.topology.residues(), 125, 142))
        >>> print(*[r.name for r in group1])  # doctest: +ELLIPSIS
        TRP ASP GLU ... ASN GLN THR
        >>> group2 = list(it.islice(model.topology.residues(), 142, 156))
        >>> print(*[r.name for r in group2])  # doctest: +ELLIPSIS
        PRO ASN ARG ... ARG THR GLY
        >>> residue_coordination = cvpack.ResidueCoordination(
        ...     group1,
        ...     group2,
        ...     stepFunction="step(1-x)",
        ...     thresholdDistance=0.6*unit.nanometers,
        ... )
        >>> residue_coordination.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(residue_coordination)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(
        ...     model.system, openmm.CustomIntegrator(0), platform
        ... )
        >>> context.setPositions(model.positions)
        >>> print(residue_coordination.getValue(context))
        26.0 dimensionless
        >>> residue_coordination.setReferenceValue(26 * unit.dimensionless)
        >>> context.reinitialize(preserveState=True)
        >>> print(residue_coordination.getValue(context, digits=6))
        1.0 dimensionless
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residueGroup1: t.Sequence[Residue],
        residueGroup2: t.Sequence[Residue],
        pbc: bool = True,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: mmunit.ScalarQuantity = mmunit.Quantity(
            1.0, mmunit.nanometers
        ),
        normalize: bool = False,
        weighByMass: bool = True,
    ) -> None:
        nr1 = len(residueGroup1)
        nr2 = len(residueGroup2)
        self._ref_val = nr1 * nr2 if normalize else 1.0
        expression = (
            f"({stepFunction})/refval"
            f"; x=distance(g1,g2)/{thresholdDistance}"
            f"; refval={self._ref_val}"
        )
        super().__init__(2, expression)
        self.setUsesPeriodicBoundaryConditions(pbc)
        for res in residueGroup1 + residueGroup2:
            self.addGroup(
                [atom.index for atom in res.atoms()],
                *([] if weighByMass else [[1] * len(res)]),
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
            weighByMass,
        )

    def getReferenceValue(self) -> mmunit.Quantity:
        """
        Get the reference value used for normalizing the residue coordination.

        Returns
        -------
        mmunit.Quantity
            The reference value.
        """
        return self._ref_val * self.getUnit()

    @mmunit.convert_quantities
    def setReferenceValue(self, value: mmunit.ScalarQuantity) -> None:
        """
        Set the reference value used for normalizing the residue coordination.

        Parameters
        ----------
            value
                The reference value.
        """
        expression = self.getEnergyFunction()
        self.setEnergyFunction(
            expression.replace(f"refval={self._ref_val}", f"refval={value}")
        )
        self._ref_val = value
