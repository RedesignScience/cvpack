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

        S_{\\rm attr}({\\bf r}) =
            &-\\sum_{i \\in {\\bf g}_1} \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ j \\neq i}}
                \\epsilon_{ij} u_{\\rm disp}\\left(
                    \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{\\sigma_{ij}}
                \\right) \\\\
            &-\\sum_{i \\in {\\bf g}_1} \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ q_jq_i < 0}}
            \\frac{q_i q_j}{4 \\pi \\varepsilon_0 r_{\\rm c}} u_{\\rm elec}\\left(
                \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{r_{\\rm c}}
            \\right)

    where :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are the two atom groups, :math:`r_{\\rm c}`
    is the cutoff distance, :math:`\\varepsilon_0` is the permittivity of empty space, and
    :math:`q_i` is the charge of atom :math:`i`. The Lennard-Jones parameters are given by the
    Lorentz-Berthelot mixing rule, i.e. :math:`\\epsilon_{ij} = \\sqrt{\\epsilon_i \\epsilon_j}`,
    and :math:`\\sigma_{ij} = (\\sigma_i + \\sigma_j)/2`.

    The function :math:`u_{\\rm disp}(x)` is a Lennard-Jones-like reduced potential with a
    highly softened repulsion part, defined as

    .. math::

        u_{\\rm disp}(x) = 4\\left(\\frac{1}{y^2} - \\frac{1}{y}\\right),
        \\quad \\text{where} \\quad
        y = |x^6 - 2| + 2

    The function :math:`u_{\\rm elec}(x)` provides a Coulomb-like decay with reaction-field
    screening, defined as

    .. math::

        u_{\\rm elec}(x) = \\frac{1}{x} + \\frac{x^2 - 3}{2}

    The screening considers a perfect conductor as the surrounding medium :cite:`Correa_2022`.

    .. note::

        Only attractive electrostatic interactions are considered (:math:`q_i q_i < 0`). This makes
        :math:`S_{\\rm attr}({\\bf r})` exclusively non-negative. Its upper bound depends on the
        system and the chosen groups of atoms.

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
        The :openmm:`NonbondedForce` object from which to collect the necessary parameters.

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
    4910.382 kJ/mol
    >>> print(attraction_strength.getEffectiveMass(context, 6))
    2.169263e-07 nm**2 mol**2 Da/(kJ**2)
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
            "; y = abs((r/sigma)**6 - 2) + 2"
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
