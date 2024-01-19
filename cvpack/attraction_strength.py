"""
.. class:: AttractionStrength
   :platform: Linux, MacOS, Windows
   :synopsis: The strength of the attraction between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable

ONE_4PI_EPS0 = 138.93545764438198


class _NonbondedForceSurrogate:  # pylint: disable=too-many-instance-attributes
    """A surrogate class for the NonbondedForce class in OpenMM."""

    def __init__(self, other: openmm.NonbondedForce) -> None:
        self._cutoff = other.getCutoffDistance()
        self._uses_pbc = other.usesPeriodicBoundaryConditions()
        self._num_particles = other.getNumParticles()
        self._particle_parameters = list(
            map(other.getParticleParameters, range(self._num_particles))
        )
        self._num_exceptions = other.getNumExceptions()
        self._exception_parameters = list(
            map(other.getExceptionParameters, range(self._num_exceptions))
        )
        self._use_switching_function = other.getUseSwitchingFunction()
        self._switching_distance = other.getSwitchingDistance()

    def __getstate__(self) -> t.Dict[str, str]:
        return {
            "cutoff": self.getCutoffDistance(),
            "uses_pbc": self.usesPeriodicBoundaryConditions(),
            "num_particles": self.getNumParticles(),
            "particle_parameters": [
                self.getParticleParameters(i) for i in range(self.getNumParticles())
            ],
            "num_exceptions": self.getNumExceptions(),
            "exception_parameters": [
                self.getExceptionParameters(i) for i in range(self.getNumExceptions())
            ],
            "use_switching_function": self.getUseSwitchingFunction(),
            "switching_distance": self.getSwitchingDistance(),
        }

    def __setstate__(self, state: t.Dict[str, str]) -> None:
        self._cutoff = state["cutoff"]
        self._uses_pbc = state["uses_pbc"]
        self._num_particles = state["num_particles"]
        self._particle_parameters = state["particle_parameters"]
        self._num_exceptions = state["num_exceptions"]
        self._exception_parameters = state["exception_parameters"]
        self._use_switching_function = state["use_switching_function"]
        self._switching_distance = state["switching_distance"]

    def getCutoffDistance(self) -> float:
        """Get the cutoff distance."""
        return mmunit.value_in_md_units(self._cutoff)

    def usesPeriodicBoundaryConditions(self) -> bool:
        """Return whether periodic boundary conditions are used."""
        return self._uses_pbc

    def getNumParticles(self) -> int:
        """Get the number of particles."""
        return self._num_particles

    def getParticleParameters(self, index: int) -> t.Tuple[float, float, float]:
        """Get the parameters of a particle at the given index."""
        return tuple(map(mmunit.value_in_md_units, self._particle_parameters[index]))

    def getNumExceptions(self):
        """Get the number of exceptions."""
        return self._num_exceptions

    def getExceptionParameters(
        self, index: int
    ) -> t.Tuple[int, int, float, float, float]:
        """Get the parameters of an exception at the given index."""
        i, j, *params = self._exception_parameters[index]
        return i, j, *map(mmunit.value_in_md_units, params)

    def getUseSwitchingFunction(self) -> bool:
        """Return whether a switching function is used."""
        return self._use_switching_function

    def getSwitchingDistance(self) -> float:
        """Get the switching distance."""
        return mmunit.value_in_md_units(self._switching_distance)


class AttractionStrength(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The strength of the attraction between two atom groups:

    .. math::

        S_{\\rm attr}({\\bf r}) = &-\\frac{1}{E_{\\rm ref}} \\Bigg[
            \\sum_{i \\in {\\bf g}_1}
                \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ j \\neq i}}
                    \\epsilon_{ij} u_{\\rm disp}\\left(
                        \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{\\sigma_{ij}}
                    \\right) \\\\
            &+\\sum_{i \\in {\\bf g}_1}
            \\sum_{\\substack{j \\in {\\bf g}_2 \\\\ q_jq_i < 0}}
                \\frac{q_i q_j}{4 \\pi \\varepsilon_0 r_{\\rm c}} u_{\\rm elec}\\left(
                    \\frac{\\|{\\bf r}_i - {\\bf r}_j\\|}{r_{\\rm c}}
                \\right) \\Bigg]

    where :math:`{\\bf g}_1` and :math:`{\\bf g}_2` are the two atom groups,
    :math:`r_{\\rm c}` is the cutoff distance, :math:`\\varepsilon_0` is the
    permittivity of empty space, :math:`q_i` is the charge of atom :math:`i`, and
    :math:`E_{\\rm ref}` is a reference value (in energy units per mole).
    The Lennard-Jones parameters are given by the Lorentz-Berthelot mixing rule, i.e.
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

        Only attractive electrostatic interactions are considered (:math:`q_i q_i < 0`),
        which gives :math:`S_{\\rm attr}({\\bf r})` a lower bound of zero. The upper
        bound will depends on the system details, the chosen groups of atoms, and the
        adopted reference value.

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
        which to obtain the reference value.

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
    4912.5 dimensionless
    >>> print(cv1.getEffectiveMass(context, 4))
    2.1639e-07 nm**2 Da
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
        reference: t.Union[mmunit.ScalarQuantity, openmm.Context] = mmunit.Quantity(
            1.0, mmunit.kilojoule_per_mole
        ),
    ) -> None:
        nonbondedForce = _NonbondedForceSurrogate(nonbondedForce)
        cutoff = nonbondedForce.getCutoffDistance()
        expression = (
            "-(lj + coul)/ref"
            "; lj = 4*epsilon*(1/y^2 - 1/y)"
            f"; coul = {ONE_4PI_EPS0}*q1q2*(1/x + (x^2 - 3)/2)"
            f"; x = r/{cutoff}"
            "; y = abs((r/sigma)^6 - 2) + 2"
            "; q1q2 = min(0, charge1*charge2)"
            "; epsilon = sqrt(epsilon1*epsilon2)"
            "; sigma = (sigma1 + sigma2)/2"
            "; ref = 1"
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
            reference = self._getValue(reference)
        self.setEnergyFunction(expression.replace("ref = 1", f"ref = {reference}"))
        self._registerCV(
            mmunit.dimensionless, group1, group2, nonbondedForce, reference
        )

    def _getValue(self, context: openmm.Context) -> float:
        system = openmm.System()
        for _ in range(context.getSystem().getNumParticles()):
            system.addParticle(1.0)
        system.addForce(openmm.CustomNonbondedForce(self))
        state = context.getState(getPositions=True)
        context = openmm.Context(system, openmm.VerletIntegrator(1.0))
        context.setPositions(state.getPositions())
        context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        # pylint: disable=unexpected-keyword-arg # to avoid false positive
        state = context.getState(getEnergy=True)
        # pylint: enable=unexpected-keyword-arg
        return mmunit.value_in_md_units(state.getPotentialEnergy())
