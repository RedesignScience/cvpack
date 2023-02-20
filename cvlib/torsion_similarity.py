"""
.. class:: DihedralSimilarity
   :platform: Linux, MacOS, Windows
   :synopsis: The degree of correlation between pairs of dihedral angles.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import numpy as np
import openmm
from openmm import unit as mmunit

from .cvlib import AbstractCollectiveVariable


class TorsionSimilarity(openmm.CustomCompoundBondForce, AbstractCollectiveVariable):
    """
    The degree of similarity between `n` pairs of torsion angles:

    .. math::

        s({\\bf r}) = \\frac{n}{2} + \\frac{1}{2} \\sum_{i=1}^n \\cos\\Big(
            \\phi^{\\rm 1st}_i({\\bf r}) - \\phi^{\\rm 2nd}_i({\\bf r})
        \\Big)

    where :math:`\\phi^{\\rm kth}_i` is the torsion angle at position :math:`i` in the :math:`k`-th
    list.

    Parameters
    ----------
        firstList
            A list of :math:`n` tuples of four atom indices defining the first torsion angle in
            each pair.
        secondList
            A list of :math:`n` tuples of four atom indices defining the second torsion angle in
            each pair.

    Example
    -------
        >>> import cvlib
        >>> import mdtraj
        >>> import openmm as mm
        >>> import openmm.unit as mmunit
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> traj = mdtraj.Trajectory(model.positions, mdtraj.Topology.from_openmm(model.topology))
        >>> phi_atoms, _ = mdtraj.compute_phi(traj)
        >>> valid_atoms = traj.top.select("resid 59 to 79 and backbone")
        >>> phi_atoms = [phi for phi in phi_atoms if all(atom in valid_atoms for atom in phi)]
        >>> torsion_similarity = cvlib.TorsionSimilarity(phi_atoms[1:], phi_atoms[:-1])
        >>> model.system.addForce(torsion_similarity)
        6
        >>> integrator = mm.VerletIntegrator(0)
        >>> platform = mm.Platform.getPlatformByName("Reference")
        >>> context = mm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(torsion_similarity.evaluateInContext(context, 6))
        18.659917 dimensionless
    """

    def __init__(
        self, firstList: Iterable[Iterable[int]], secondList: Iterable[Iterable[int]]
    ) -> None:
        assert all(len(torsion) == 4 for torsion in firstList)  # each torsion must have 4 atoms
        assert all(len(torsion) == 4 for torsion in secondList)  # each torsion must have 4 atoms
        energy = f"0.5*(1 + cos(min(delta, {2*np.pi} - delta)))"
        definition = "delta = dihedral(p1, p2, p3, p4) - dihedral(p5, p6, p7, p8)"
        super().__init__(8, f"{energy}; {definition}")
        for first, second in zip(firstList, secondList):
            self.addBond([*first, *second], [])
        self._registerCV(mmunit.dimensionless, firstList, secondList)
