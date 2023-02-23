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

from .cvpack import AbstractCollectiveVariable, in_md_units


class RMSD(openmm.RMSDForce, AbstractCollectiveVariable):
    """
    The minimum root-mean-square deviation (RMSD) between the current and reference coordinates of a
    group of `n` atoms:

    .. math::

        d_{\\rm rms}({\\bf r}) = \\sqrt{
            \\frac{1}{n} \\sum_{i=1}^n \\left\\|
                \\hat{\\bf r}_i - {\\bf R}({\\bf r}) \\hat{\\bf r}_i^{\\rm ref}
            \\right\\|^2
        }

    where :math:`\\hat{\\bf r}_i` is the position of the :math:`i`-th atom in the group relative to
    the centroid of the group, :math:`\\hat{\\bf r}_i^{\\rm ref}` is the position of the same atom
    in the reference configuration, and :math:`{\\bf R}({\\bf r})` is the rotation matrix that
    minimizes the RMSD between the group and the reference structure.

    .. warning::

        Periodic boundary conditions are not supported (see OpenMM issue `#2392
        <https://github.com/openmm/openmm/issues/2913>`_). This is not a problem if all atoms belong
        to the same molecule. If they do not, a way to circumvent a possible error is to call
        :func:`getFakeBondForce` and add the resulting force to the system in addition to this CV.

    Parameters
    ----------
        referencePositions
            The reference coordinates. If there are ``n`` coordinates,  with ``n=len(group)``, they
            must refer to the group atoms in the same order as they appear in ``group``. Otherwise,
            if there are ``numAtoms`` coordinates (see below), they must refer to the the system
            atoms and be sorted accordingly. The first criterion has precedence over the second when
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
        >>> import openmm as mm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideImplicit()
        >>> num_atoms = model.topology.getNumAtoms()
        >>> group = list(range(num_atoms))
        >>> rmsd = cvpack.RMSD(model.positions, group, num_atoms)
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

    def __init__(
        self,
        referencePositions: Union[np.ndarray, Iterable[openmm.Vec3], mmunit.Quantity],
        group: Iterable[int],
        numAtoms: int,
    ) -> None:
        coords = in_md_units(referencePositions)
        num_coords = coords.shape[0] if isinstance(coords, np.ndarray) else len(coords)
        if num_coords == len(group):
            positions = np.zeros((numAtoms, 3))
            for i, atom in enumerate(group):
                positions[atom, :] = np.array([coords[i][j] for j in range(3)])
        elif num_coords == numAtoms:
            positions = coords.copy()
            coords = np.array([positions[atom] for atom in group])
        else:
            raise ValueError("Invalid number of coordinates")
        super().__init__(positions, group)
        self._registerCV(mmunit.nanometers, coords, group, numAtoms)

    def getFakeBondForce(self) -> openmm.HarmonicBondForce:
        """
        Get a bond force that does not contribute to the energy but, if added to the same system,
        can guarantee that this collective variable is computed correctly when the atoms in the
        group do not belong to the same molecule.

        Returns
        -------
            A null bond force that creates a connected graph with all the atoms in the group.

        Example
        -------
            >>> import cvpack
            >>> import openmm as mm
            >>> from openmm import app, unit
            >>> from openmmtools import testsystems
            >>> model = testsystems.WaterBox(box_edge=10*unit.angstroms, cutoff=5*unit.angstroms)
            >>> group = [atom.index for atom in model.topology.atoms() if atom.residue.index < 3]
            >>> rmsd = cvpack.RMSD(model.positions, group, model.topology.getNumAtoms())
            >>> [model.system.addForce(f) for f in [rmsd, rmsd.getFakeBondForce()]]
            [3, 4]
            >>> integrator = openmm.VerletIntegrator(2*unit.femtoseconds)
            >>> platform = openmm.Platform.getPlatformByName('Reference')
            >>> context = openmm.Context(model.system, integrator, platform)
            >>> context.setPositions(model.positions)
            >>> integrator.step(100)
            >>> print(rmsd.getValue(context, digits=6))
            0.104363 nm

        """
        force = openmm.HarmonicBondForce()
        group = self._args["group"]
        for i in range(len(group) - 1):
            force.addBond(group[i], group[i + 1], 0.0, 0.0)
        return force
