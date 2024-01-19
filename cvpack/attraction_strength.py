"""
.. class:: AttractionStrength
   :platform: Linux, MacOS, Windows
   :synopsis: The strength of the attraction between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
import numbers

import openmm
import xmltodict

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable

ONE_4PI_EPS0 = 138.93545764438198


class _NonbondedForce(openmm.NonbondedForce):
    def __getstate__(self) -> t.Dict[str, str]:
        return xmltodict.parse(openmm.XmlSerializer.serialize(self))

    def __setstate__(self, state: t.Dict[str, str]) -> None:
        self.__init__(openmm.XmlSerializer.deserialize(xmltodict.unparse(state)))


class AttractionStrength(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The strength of the attraction between two atom groups:

    .. math::

        S_{\\rm attr}({\\bf r}) =
            &-\\sum_{i \\in {\\bf g}_1}
                \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ j \\neq i}}
                    \\epsilon_{ij} u_{\\rm disp}\\left(
                        \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{\\sigma_{ij}}
                    \\right) \\\\
            &-\\sum_{i \\in {\\bf g}_1}
            \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ q_jq_i < 0}}
                \\frac{q_i q_j}{4 \\pi \\varepsilon_0 r_{\\rm c}} u_{\\rm elec}\\left(
                    \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{r_{\\rm c}}
                \\right)

    where :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are the two atom groups,
    :math:`r_{\\rm c}` is the cutoff distance, :math:`\\varepsilon_0` is the
    permittivity of empty space, and :math:`q_i` is the charge of atom :math:`i`. The
    Lennard-Jones parameters are given by the Lorentz-Berthelot mixing rule, i.e.
    :math:`\\epsilon_{ij} = \\sqrt{\\epsilon_i \\epsilon_j}`, and
    :math:`\\sigma_{ij} = (\\sigma_i + \\sigma_j)/2`.

    The function :math:`u_{\\rm disp}(x)` is a Lennard-Jones-like reduced potential with
    a highly softened repulsion part, defined as

    .. math::

        u_{\\rm disp}(x) = 4\\left(\\frac{1}{y^2} - \\frac{1}{y}\\right),
        \\quad \\text{where} \\quad
        y = |x^6 - 2| + 2

    The function :math:`u_{\\rm elec}(x)` provides a Coulomb-like decay with
    reaction-field screening, defined as

    .. math::

        u_{\\rm elec}(x) = \\frac{1}{x} + \\frac{x^2 - 3}{2}

    The screening considers a perfect conductor as the surrounding medium
    :cite:`Correa_2022`.

    .. note::

        Only attractive electrostatic interactions are considered (:math:`q_i q_i < 0`).
        This makes :math:`S_{\\rm attr}({\\bf r})` exclusively non-negative. Its upper
        bound depends on the system and the chosen groups of atoms.

    The Lennard-Jones parameters, atomic charges, cutoff distance, boundary conditions,
    as well as whether to use a switching function and its corresponding switching
    distance, are taken from :openmm:`NonbondedForce` object. Any non-exclusion
    exceptions involving atoms in :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are turned
    into exclusions.

    Parameters
    ----------
    group1
        The first atom group.
    group2
        The second atom group.
    nonbondedForce
        The :openmm:`NonbondedForce` object from which to collect the necessary
        parameters.
    reference
        A reference value (in energy units per mole) to which the collective variable
        should be normalized. One can also provide an :OpenMM:`Context` object from
        which to obtain the reference value. If either is provided, the collective
        variable will become dimensionless.

    Examples
    --------
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestExplicit()
    >>> group1, group2 = [], []
    >>> for residue in model.topology.residues():
    ...     if residue.name != "HOH":
    ...         group = group1 if residue.name == "B2" else group2
    ...         group.extend(atom.index for atom in residue.atoms())
    >>> forces = {f.getName(): f for f in model.system.getForces()}
    >>> cv1 = cvpack.AttractionStrength(group1, group2, forces["NonbondedForce"])
    >>> model.system.addForce(cv1)
    5
    >>> cv2 = cvpack.AttractionStrength(
    ...     group1, group2, forces["NonbondedForce"], 100*unit.kilojoules_per_mole,
    ... )
    >>> model.system.addForce(cv2)
    6
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> print(cv1.getValue(context, 4))
    4912.5 kJ/mol
    >>> print(cv1.getEffectiveMass(context, 4))
    2.1639e-07 nm**2 mol**2 Da/(kJ**2)
    >>> print(cv2.getValue(context, 4))
    49.125 dimensionless
    >>> print(cv2.getEffectiveMass(context, 4))
    0.0021639 nm**2 Da
    >>> cv3 = cvpack.AttractionStrength(
    ...     group1, group2, forces["NonbondedForce"], context,
    ... )
    >>> model.system.addForce(cv3)
    7
    >>> context.reinitialize(preserveState=True)
    >>> print(cv3.getValue(context, 4))
    1.0 dimensionless
    >>> print(cv3.getEffectiveMass(context, 4))
    5.2222 nm**2 Da
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        nonbondedForce: openmm.NonbondedForce,
        reference: t.Union[mmunit.ScalarQuantity, openmm.Context, None] = None,
    ) -> None:
        cutoff = mmunit.value_in_md_units(nonbondedForce.getCutoffDistance())
        expression = (
            "-(lj + coul)/refval"
            "; lj = 4*epsilon*(1/y^2 - 1/y)"
            f"; coul = {ONE_4PI_EPS0}*q1q2*(1/x + (x^2 - 3)/2)"
            f"; x = r/{cutoff}"
            "; y = abs((r/sigma)^6 - 2) + 2"
            "; q1q2 = min(0, charge1*charge2)"
            "; epsilon = sqrt(epsilon1*epsilon2)"
            "; sigma = (sigma1 + sigma2)/2"
            "; refval = 1"
        )
        super().__init__(expression)
        if nonbondedForce.usesPeriodicBoundaryConditions():
            self.setNonbondedMethod(self.CutoffPeriodic)
        else:
            self.setNonbondedMethod(self.CutoffNonPeriodic)
        self.setCutoffDistance(cutoff)
        for parameter in ("charge", "sigma", "epsilon"):
            self.addPerParticleParameter(parameter)
        for atom in range(nonbondedForce.getNumParticles()):
            self.addParticle(nonbondedForce.getParticleParameters(atom))
        for exception in range(nonbondedForce.getNumExceptions()):
            i, j, *_ = nonbondedForce.getExceptionParameters(exception)
            self.addExclusion(i, j)
        self.setUseSwitchingFunction(nonbondedForce.getUseSwitchingFunction())
        self.setSwitchingDistance(nonbondedForce.getSwitchingDistance())
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        if isinstance(reference, openmm.Context):
            reference = self._get_value(reference)
        if isinstance(reference, numbers.Number):
            self.setEnergyFunction(
                expression.replace("refval = 1", f"refval = {reference}")
            )
        self._registerCV(
            mmunit.kilojoules_per_mole if reference is None else mmunit.dimensionless,
            group1,
            group2,
            _NonbondedForce(nonbondedForce),
            reference,
        )

    def _get_value(self, context: openmm.Context) -> float:
        system = openmm.System()
        for _ in range(context.getSystem().getNumParticles()):
            system.addParticle(1.0)
        system.addForce(openmm.CustomNonbondedForce(self))
        state = context.getState(getPositions=True)
        context = openmm.Context(system, openmm.VerletIntegrator(1.0))
        context.setPositions(state.getPositions())
        context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        state = context.getState(getEnergy=True)
        return mmunit.value_in_md_units(state.getPotentialEnergy())
