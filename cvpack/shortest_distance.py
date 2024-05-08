"""
.. class:: ShortestDistance
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import unit as mmunit

from .collective_variable import CollectiveVariable
from .units import Quantity


class ShortestDistance(CollectiveVariable, openmm.CustomCVForce):
    r"""
    A smooth approximation of the shortest distance between two atom groups:

    .. math::
        r_{\rm min}({\bf r}) = \frac{
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2}
                r_{ij} e^{-\frac{r_{ij}^2}{2 \sigma^2}}
        }{
            \sum_{i \in {\bf g}_1} \sum_{j \in {\bf g}_2}
                e^{-\frac{r_{ij}^2}{2 \sigma^2}}
        }

    where :math:`r_{ij} = \|{\bf r}_j - {\bf r}_i\|` is the distance between atoms
    :math:`i` and :math:`j` and :math:`\sigma` is a parameter that controls the
    degree of approximation. The smaller the value of :math:`\sigma`, the closer the
    approximation to the true shortest distance.

    In practice, a cutoff distance :math:`r_c` is applied to the atom pairs so that
    the collective variable is computed only for pairs of atoms separated by a distance
    less than :math:`r_c`. A switching function is also applied to smoothly turn off
    the collective variable starting from a distance :math:`r_s < r_c`.

    .. note::

        Atoms are allowed to be in both groups. In this case, terms for which
        :math:`i = j` are ignored.

    Parameters
    ----------
    group1
        The indices of the atoms in the first group.
    group2
        The indices of the atoms in the second group.
    numAtoms
        The total number of atoms in the system, including those that are not in any
        of the groups.
    sigma
        The parameter that controls the degree of approximation.
    magnitude
        The expected order of magnitude of the shortest distance. This parameter does
        not affect the value of the collective variable, but a good estimate can
        improve numerical stability.
    cutoffDistance
        The cutoff distance for evaluating atom pairs.
    switchDistance
        The distance at which the switching function starts to be applied.
    pbc
        Whether to consider periodic boundary conditions in distance calculations.
    name
        The name of the collective variable.

    Example
    -------
    >>> import cvpack
    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestVacuum()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     group = group1 if residue.name == "B2" else group2
    ...     group.extend(atom.index for atom in residue.atoms())
    >>> coords = model.positions.value_in_unit(mmunit.nanometers)
    >>> np.linalg.norm(
    ...     coords[group1, None, :] - coords[None, group2, :],
    ...     axis=-1,
    ... ).min()
    0.17573...
    >>> num_atoms = model.system.getNumParticles()
    >>> min_dist = cvpack.ShortestDistance(group1, group2, num_atoms)
    >>> min_dist.addToSystem(model.system)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> min_dist.getValue(context)
    0.17573... nm
    """

    def __init__(
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        numAtoms: int,
        sigma: mmunit.Quantity = Quantity(0.01 * mmunit.nanometers),
        magnitude: mmunit.Quantity = Quantity(0.2 * mmunit.nanometers),
        cutoffDistance: mmunit.Quantity = Quantity(0.5 * mmunit.nanometers),
        switchDistance: mmunit.Quantity = Quantity(0.4 * mmunit.nanometers),
        pbc: bool = True,
        name: str = "shortest_distance",
    ) -> None:
        if mmunit.is_quantity(sigma):
            sigma = sigma.value_in_unit(mmunit.nanometers)
        if mmunit.is_quantity(magnitude):
            magnitude = magnitude.value_in_unit(mmunit.nanometers)
        weight = f"exp(-0.5*(r^2 - {magnitude**2})/{sigma**2})"
        forces = {
            "numerator": openmm.CustomNonbondedForce(f"r*{weight}"),
            "denominator": openmm.CustomNonbondedForce(weight),
        }
        super().__init__("numerator/denominator")
        for cv, force in forces.items():
            force.setNonbondedMethod(
                force.CutoffPeriodic if pbc else force.CutoffNonPeriodic
            )
            force.setCutoffDistance(cutoffDistance)
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(switchDistance)
            force.setUseLongRangeCorrection(False)
            for _ in range(numAtoms):
                force.addParticle([])
            force.addInteractionGroup(group1, group2)
            self.addCollectiveVariable(cv, force)
        self._registerCV(
            name,
            mmunit.nanometers,
            group1=group1,
            group2=group2,
            numAtoms=numAtoms,
            sigma=sigma,
            magnitude=magnitude,
            cutoffDistance=cutoffDistance,
            switchDistance=switchDistance,
            pbc=pbc,
        )


ShortestDistance.registerTag("!cvpack.ShortestDistance")
