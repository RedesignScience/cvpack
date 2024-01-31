"""
.. class:: RadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: The radius of gyration of a group of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from cvpack import unit as mmunit

from .base_radius_of_gyration import BaseRadiusOfGyration


class RadiusOfGyration(BaseRadiusOfGyration):
    r"""
    The radius of gyration of a group of :math:`n` atoms:

    .. math::

        r_g({\bf r}) = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left\|
            {\bf r}_i - {\bf r}_c({\bf r})
        \right\|^2 }.

    where :math:`{\bf r}_c({\bf r})` is the geometric center of the group:

    .. math::

        {\bf r}_c({\bf r}) = \frac{1}{n} \sum_{i=j}^n {\bf r}_j

    Optionally, the radius of gyration can be computed with respect to the center of
    mass of the group. In this case, the geometric center is replaced by:

    .. math::

        {\bf r}_m({\bf r}) = \frac{1}{M} \sum_{i=1}^n m_i {\bf r}_i

    where :math:`M = \sum_{i=1}^n m_i` is the total mass of the group.

    .. note::

        This collective variable lacks parallelization and might be slow when the group
        of atoms is large. In this case, :class:`RadiusOfGyrationSq` might be preferred.

    Parameters
    ----------
        group
            The indices of the atoms in the group
        pbc
            Whether to use periodic boundary conditions
        weighByMass
            Whether to use the center of mass of the group instead of its geometric
            center

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> num_atoms = model.system.getNumParticles()
        >>> radius_of_gyration = cvpack.RadiusOfGyration(list(range(num_atoms)))
        >>> radius_of_gyration.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(radius_of_gyration)
        5
        >>> platform =openmm.Platform.getPlatformByName('Reference')
        >>> integrator =openmm.VerletIntegrator(0)
        >>> context =openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(radius_of_gyration.getValue(context, digits=6))
        0.2951431 nm

    """

    def __init__(
        self, group: t.Sequence[int], pbc: bool = False, weighByMass: bool = False
    ) -> None:
        num_atoms = len(group)
        num_groups = num_atoms + 1
        sum_dist_sq = "+".join(
            [f"distance(g{i+1}, g{num_atoms + 1})^2" for i in range(num_atoms)]
        )
        super().__init__(
            num_groups, f"sqrt(({sum_dist_sq})/{num_atoms})", group, pbc, weighByMass
        )
        self.addBond(list(range(num_groups)))
        self._registerCV(mmunit.nanometers, group, pbc, weighByMass)
