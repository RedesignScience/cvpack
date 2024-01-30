"""
.. class:: CompositeRMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Deviation of multiple corotating bodies from their reference structures

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import copy
import typing as t

import numpy as np
import openmm
import openmmcppforces as mmcpp

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable


class CompositeRMSD(mmcpp.CompositeRMSDForce, AbstractCollectiveVariable):
    """
    The minimum root-mean-square deviation (RMSD) between the current and reference
    coordinates of a group of `n` atoms:

    .. math::

        d_{\\rm rms}({\\bf r}) = \\sqrt{
            \\frac{1}{n} \\sum_{i=1}^n \\left\\|
                \\hat{\\bf r}_i - {\\bf A}({\\bf r}) \\hat{\\bf r}_i^{\\rm ref}
            \\right\\|^2
        }

    where :math:`\\hat{\\bf r}_i` is the position of the :math:`i`-th atom in the group
    relative to the group's center of geometry (centroid),
    :math:`\\hat{\\bf r}_i^{\\rm ref}` is the centroid-centered position of the same
    atom in a reference configuration, and :math:`{\\bf A}({\\bf r})` is the rotation
    matrix that minimizes the RMSD between the group and the reference structure.

    .. warning::

        Periodic boundary conditions are `not supported
        <https://github.com/openmm/openmm/issues/2913>`_. It atoms in the group belong
        to distinct molecules, calling :func:`getNullBondForce` and adding the resulting
        force to the system might circumvent any potential issues.

    Parameters
    ----------
        referencePositions
            The reference coordinates. If there are ``n`` coordinates,  with
            ``n=len(group)``, they must refer to the group atoms in the same order as
            they appear in ``group``. Otherwise, if there are ``numAtoms`` coordinates
            (see below), they must refer to the the system atoms and be sorted
            accordingly. The first criterion has precedence over the second when
            ``n == numAtoms``.
        group
            The index of the atoms in the group
        numAtoms
            The total number of atoms in the system (required by OpenMM)

    Raises
    ------
        ValueError
            If ``len(referencePositions)`` is neither ``numAtoms`` nor ``len(group)``

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideImplicit()
        >>> num_atoms = model.topology.getNumAtoms()
        >>> group = list(range(num_atoms))
        >>> rmsd = cvpack.RMSD(model.positions, group, num_atoms)
        >>> rmsd.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(rmsd)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> integrator.step(1000)
        >>> value = rmsd.getValue(context)
        >>> round(value/value.unit, 7)
        0.1231383

    """

    @mmunit.convert_quantities
    def __init__(
        self,
        referencePositions: t.Union[mmunit.MatrixQuantity, t.Dict[int, mmunit.VectorQuantity]],
        groups: t.Sequence[t.Sequence[int]],
        numAtoms: int = 0,
    ) -> None:
        num_atoms = numAtoms or len(referencePositions)
        all_atoms = sum(groups, [])
        if len(set(all_atoms)) != len(all_atoms):
            raise ValueError("Atom groups must be disjoint")
        defined_coords = {atom: tuple(referencePositions[atom]) for atom in all_atoms}
        all_coords = np.zeros((num_atoms, 3))
        for atom, coords in defined_coords.items():
            all_coords[atom, :] = coords
        super().__init__(all_coords)
        for group in groups:
            self.addGroup(group)
        self._registerCV(mmunit.nanometers, defined_coords, groups, 0)
