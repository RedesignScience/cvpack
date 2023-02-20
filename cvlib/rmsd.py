"""
.. class:: RMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Root-mean-square deviation of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable, Union

import numpy as np
import openmm
from openmm import unit as mmunit

from .cvlib import AbstractCollectiveVariable, in_md_units


class RMSD(openmm.RMSDForce, AbstractCollectiveVariable):
    """
    The minimum root-mean-square deviation (RMSD) between the current and reference coordinates of a
    group of `n` atoms:

    .. math::

        d_{\\rm rms}({\\bf r}) = \\sqrt{
            \\frac{1}{n} \\sum_{i=1}^n \\left\\|
                {\\bf r}_i - {\\bf R}({\\bf r}) {\\bf r}_i^{\\rm ref}
            \\right\\|^2
        }

    where :math:`{\\bf R}(\\bf r)` is the rotation matrix that minimizes the RMSD.

    Parameters
    ----------
        referencePositions
            The reference coordinates. If there are ``numAtoms`` coordinates, they must refer to the
            the system atoms and be sorted accordingly. Otherwise, if there are ``n`` coordinates,
            with ``n=len(group)``, they must refer to the group atoms in the same order as they
            appear in ``group``. The first criterion has precedence when ``n == numAtoms``.
        group
            The index of the atoms in the group
        numAtoms
            The total number of atoms in the system (required by OpenMM)

    Example
    -------
        >>> import cvlib
        >>> import openmm as mm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideImplicit()
        >>> num_atoms = model.topology.getNumAtoms()
        >>> group = list(range(num_atoms))
        >>> rmsd = cvlib.RMSD(model.positions, group, num_atoms)
        >>> model.system.addForce(rmsd)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> integrator.step(1000)
        >>> value = rmsd.evaluateInContext(context)
        >>> round(value/value.unit, 7)
        0.1231383

    """

    def __init__(
        self,
        referencePositions: Union[np.ndarray, Iterable[openmm.Vec3], mmunit.Quantity],
        group: Iterable[int],
        numAtoms: int,
    ) -> None:
        coords = in_md_units(referencePositions)
        num_coords = coords.shape[0] if isinstance(coords, np.ndarray) else len(coords)
        assert num_coords == len(group) or num_coords == numAtoms
        if num_coords == numAtoms:
            positions = coords.copy()
            coords = np.array([positions[atom] for atom in group])
        else:
            positions = np.zeros((numAtoms, 3))
            for i, atom in enumerate(group):
                positions[atom, :] = np.array([coords[i][j] for j in range(3)])
        super().__init__(positions, group)
        self._registerCV(mmunit.nanometers, coords, group, numAtoms)
