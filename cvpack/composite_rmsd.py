"""
.. class:: CompositeRMSD
   :platform: Linux, MacOS, Windows
   :synopsis: Deviation of multiple corotating bodies from their reference structures

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


class _Stub:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        raise ImportError(
            "CompositeRMSD requires the mdtools::openmm-cpp-forces conda package"
        )


try:
    from openmmcppforces import CompositeRMSDForce
except ImportError:
    CompositeRMSDForce = _Stub


class CompositeRMSD(CompositeRMSDForce, BaseCollectiveVariable):
    r"""
    The composite root-mean-square deviation (RMSD) between the current and reference
    coordinates of :math:`m` groups of atoms:

    .. math::

        d_{\rm crms}({\bf r}) = \sqrt{
            \frac{1}{n} \min_{
                \bf q \in \mathbb{R}^4 \atop \|{\bf q}\| = 1
            } \sum_{j=1}^m \sum_{i \in {\bf g}_j} \left\|
                {\bf A}({\bf q})\left({\bf r}_i - {\bf c}_j\right) -
                    {\bf r}_i^{\rm ref} + {\bf c}_j^{\rm ref}
            \right\|^2
        }

    where each group :math:`{\bf g}_j` is a set of :math:`n_j` atom indices,
    :math:`n = \sum_{j=1}^m n_j` is the total number of atoms in these groups,
    :math:`{\bf A}(\bf q)` is the rotation matrix corresponding to a unit quaternion
    :math:`{\bf q}`, :math:`{\bf r}_i` and :math:`{\bf r}_i^{\rm ref}` are the
    positions of atom :math:`i` in the current and reference structures, respectively,
    :math:`{\bf c}_j` is the position of atom :math:`i`, given by

    .. math::

        {\bf c}_j = \frac{1}{n_j} \sum_{i \in {\bf g}_j} {\bf r}_i

    and :math:`{\bf c}_j^{\rm ref}` is the centroid of the reference structure for
    group :math:`j`, defined analogously to :math:`{\bf c}_j`.

    .. warning::

        To use this class, you must install the `openmm-cpp-forces`_ conda package.

    .. _openmm-cpp-forces:

        https://anaconda.org/mdtools/openmm-cpp-forces

    Parameters
    ----------
    referencePositions
        The reference coordinates, which can be either a coordinate matrix or a mapping
        from atom indices to coordinate vectors. It must contain all atoms in
        ``groups``, and does not need to contain all atoms in the system. See
        ``numAtoms`` below.
    groups
        A sequence of disjoint atom groups. Each group is a sequence of atom indices.
    numAtoms
        The total number of atoms in the system, including those that are not in
        ``groups``. This argument is necessary only if ``referencePositions`` does not
        contain all atoms in the system.

    Raises
    ------
    ImportError
        If the `openmm-cpp-forces`_ conda package is not installed.
    ValueError
        If ``groups`` is not a sequence of disjoint atom groups.

    Example
    -------
    >>> import cvpack
    >>> import openmm as mm
    >>> import pytest
    >>> from openmmtools import testsystems
    >>> from cvpack import unit
    >>> model = testsystems.HostGuestVacuum()
    >>> host_atoms, guest_atoms = (
    ...     [a.index for a in r.atoms()]
    ...     for r in model.topology.residues()
    ... )
    >>> try:
    ...     composite_rmsd = cvpack.CompositeRMSD(
    ...         model.positions,
    ...         [host_atoms, guest_atoms],
    ...     )
    ... except ImportError:
    ...     pytest.skip("openmm-cpp-forces is not installed")
    >>> composite_rmsd.setUnusedForceGroup(0, model.system)
    1
    >>> model.system.addForce(composite_rmsd)
    5
    >>> context = mm.Context(
    ...     model.system,
    ...     mm.VerletIntegrator(1.0 * unit.femtoseconds),
    ...     mm.Platform.getPlatformByName('Reference'),
    ... )
    >>> context.setPositions(model.positions)
    >>> print(composite_rmsd.getValue(context))
    0.0 nm
    >>> model.positions[guest_atoms] += 1.0 * unit.nanometers
    >>> context.setPositions(model.positions)
    >>> print(composite_rmsd.getValue(context))
    0.0 nm
    """

    @mmunit.convert_quantities
    def __init__(
        self,
        referencePositions: t.Union[
            mmunit.MatrixQuantity, t.Dict[int, mmunit.VectorQuantity]
        ],
        groups: t.Sequence[t.Sequence[int]],
        numAtoms: t.Optional[int] = None,
    ) -> None:
        num_atoms = numAtoms or len(referencePositions)
        all_atoms = sum(groups, [])
        if len(set(all_atoms)) != len(all_atoms):
            raise ValueError("Atom groups must be disjoint")
        defined_coords = {atom: referencePositions[atom] for atom in all_atoms}
        all_coords = np.zeros((num_atoms, 3))
        for atom, coords in defined_coords.items():
            all_coords[atom, :] = coords
        super().__init__(all_coords)
        for group in groups:
            self.addGroup(group)
        self._registerCV(mmunit.nanometers, defined_coords, groups, num_atoms)
