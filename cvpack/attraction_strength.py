"""
.. class:: AttractionStrength
   :platform: Linux, MacOS, Windows
   :synopsis: The strength of the attraction between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Dict, Sequence

import openmm

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable


class _NonbondedForce(openmm.NonbondedForce):
    def __getstate__(self) -> Dict[str, str]:
        return {"xml": openmm.XmlSerializer.serialize(self)}

    def __setstate__(self, state: Dict[str, str]) -> None:
        self.__init__(openmm.XmlSerializer.deserialize(state["xml"]))


class AttractionStrength(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The strength of the attraction between two atom groups:

    .. math::

        U({\\bf r}) = \\sum_{i \\in {\\bf g}_1} \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ j \\neq i}}
        \\left\\{
            4 \\epsilon_{ij} \\left(\\frac{1}{y_{ij}} - \\frac{1}{y_{ij}^2}\\right) +
            \\frac{q_{ij}^2}{4 \\pi \\varepsilon_0 r_{\\rm c}}
                \\left(\\frac{1}{x_{ij}} + \\frac{x_{ij}^2 - 3}{2}\\right)
        \\right\\}

    where :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are the two atom groups, :math:`r_{\\rm c}`
    is the cutoff distance, and :math:`\\varepsilon_0` is the permittivity of empty space. The
    charge product :math:`q_{ij}^2` is zero if atoms i and j repell each other and is equal to
    :math:`-q_i q_j` otherwise, i.e. :math:`q_{ij}^2 = \\max(0, -q_i q_j)`. The Lennard-Jones
    parameters are given by Lorentz-Berthelot mixing rules, i.e. :math:`\\epsilon_{ij} =
    \\sqrt{\\epsilon_i \\epsilon_j}`, and :math:`\\sigma_{ij} = (\\sigma_i + \\sigma_j)/2`.
    The remaining symbols are defined as:

    .. math::

        x_{ij} = \\left\\|\\frac{{\\bf r}_i - {\\bf r}_j}{r_{\\rm c}}\\right\\|

    and :math:`y_{ij} = \\max(z_{ij}, 4 - z_{ij})`, where

    .. math::

        z_{ij} = \\left\\|\\frac{{\\bf r}_i - {\\bf r}_j}{\\sigma_{ij}}\\right\\|^6

    The Lennard-Jones parameters, atomic charges, cutoff distance, boundary conditions, as well as
    whether to use a switching function and its corresponding switching distance, are taken from
    :openmm:`NonbondedForce` object. Any non-exclusion exceptions involving atoms in
    :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are turned into exclusions.

    Parameters
    ----------
    group1
        The first atom group.
    group2
        The second atom group.
    nonbonded_force
        The :openmm:`NonbondedForce` object from which to take the necessary parameters.

    Examples
    --------
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestExplicit()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     if residue.name != "HOH":
    ...         group = group1 if residue.name == "B2" else group2
    ...         group.extend(atom.index for atom in residue.atoms())
    >>> forces = {f.getName(): f for f in model.system.getForces()}
    >>> attraction_strength = AttractionStrength(
    ...     group1, group2, forces["NonbondedForce"]
    ... )
    >>> model.system.addForce(attraction_strength)
    5
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> print(attraction_strength.getValue(context, 6))
    4912.514 kJ/mol
    >>> print(attraction_strength.getEffectiveMass(context, 6))
    2.163946e-07 nm**2 mol**2 Da/(kJ**2)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: Sequence[int],
        group2: Sequence[int],
        nonbonded_force: openmm.NonbondedForce,
    ) -> None:
        one_4pi_eps0 = 138.93545764438198
        cutoff = mmunit.value_in_md_units(nonbonded_force.getCutoffDistance())
        super().__init__(
            f"4*epsilon*(1/y - 1/y^2) + {one_4pi_eps0}*q12sq*(1/x + (x^2 - 3)/2)"
            f"; x = r/{cutoff}"
            "; y = max(z, 4 - z)"
            "; z = (r/sigma)^6"
            "; q12sq = max(0, -charge1*charge2)"
            "; epsilon = sqrt(epsilon1*epsilon2)"
            "; sigma = (sigma1 + sigma2)/2"
        )
        if nonbonded_force.usesPeriodicBoundaryConditions():
            self.setNonbondedMethod(self.CutoffPeriodic)
        else:
            self.setNonbondedMethod(self.CutoffNonPeriodic)
        self.setCutoffDistance(cutoff)
        for parameter in ("charge", "sigma", "epsilon"):
            self.addPerParticleParameter(parameter)
        for atom in range(nonbonded_force.getNumParticles()):
            self.addParticle(nonbonded_force.getParticleParameters(atom))
        self.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
        self.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        self._registerCV(
            mmunit.kilojoules_per_mole,
            group1,
            group2,
            _NonbondedForce(nonbonded_force),
        )
