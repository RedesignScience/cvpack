"""
.. class:: AttractionStrength
   :platform: Linux, MacOS, Windows
   :synopsis: The strength of the attraction between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable
from .utils import NonbondedForceSurrogate, evaluate_in_context

ONE_4PI_EPS0 = 138.93545764438198


class AttractionStrength(openmm.CustomNonbondedForce, BaseCollectiveVariable):
    r"""
    The strength of the attraction between two atom groups:

    .. math::

        S_{\rm attr}({\bf r}) = S_{1,2}({\bf r}) = &-\frac{1}{E_{\rm ref}} \Bigg[
            \sum_{i \in {\bf g}_1}
                \sum_{\substack{j \in {\bf g}_2 \\ j \neq i}}
                    \epsilon_{ij} u_{\rm disp}\left(
                        \frac{\|{\bf r}_i - {\bf r}_j\|}{\sigma_{ij}}
                    \right) \\
            &+\sum_{i \in {\bf g}_1}
            \sum_{\substack{j \in {\bf g}_2 \\ q_jq_i < 0}}
                \frac{q_i q_j}{4 \pi \varepsilon_0 r_{\rm c}} u_{\rm elec}\left(
                    \frac{\|{\bf r}_i - {\bf r}_j\|}{r_{\rm c}}
                \right) \Bigg]

    where :math:`{\bf g}_1` and :math:`{\bf g}_2` are the two atom groups,
    :math:`r_{\rm c}` is the cutoff distance, :math:`\varepsilon_0` is the
    permittivity of empty space, :math:`q_i` is the charge of atom :math:`i`, and
    :math:`E_{\rm ref}` is a reference value (in energy units per mole).
    The Lennard-Jones parameters are given by the Lorentz-Berthelot mixing rule, i.e.
    :math:`\epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}`, and
    :math:`\sigma_{ij} = (\sigma_i + \sigma_j)/2`.

    Optionally, one can provide a third atom group :math:`{\bf g}_3` to contrast
    the attraction strength between :math:`{\bf g}_1` and :math:`{\bf g}_2` with
    that between :math:`{\bf g}_1` and :math:`{\bf g}_3`. One can also provide
    a scaling factor :math:`\alpha` to balance the contributions of the two
    interactions. In this case, the collective variable becomes

    .. math::

        S_{\rm attr}({\bf r}) = S_{1,2}({\bf r}) - \alpha S_{1,3}({\bf r})

    .. note::

        Groups :math:`{\bf g}_1` and :math:`{\bf g}_2` can overlap or even be the
        same, in which case the collective variable will measure the strength of
        the self-attraction of :math:`{\bf g}_1`. On the other hand, the contrast
        group :math:`{\bf g}_3` cannot overlap with neither :math:`{\bf g}_1` nor
        :math:`{\bf g}_2`.

    The function :math:`u_{\rm disp}(x)` is a Lennard-Jones-like reduced potential with
    a highly softened repulsion part, defined as

    .. math::

        u_{\rm disp}(x) = 4\left(\frac{1}{y^2} - \frac{1}{y}\right),
        \quad \text{where} \quad
        y = |x^6 - 2| + 2

    The function :math:`u_{\rm elec}(x)` provides a Coulomb-like decay with
    reaction-field screening, defined as

    .. math::

        u_{\rm elec}(x) = \frac{1}{x} + \frac{x^2 - 3}{2}

    The screening considers a perfect conductor as the surrounding medium
    :cite:`Correa_2022`.

    .. note::

        Only attractive electrostatic interactions are considered (:math:`q_i q_i < 0`),
        which gives :math:`S_{\rm attr}({\bf r})` a lower bound of zero. The upper
        bound will depends on the system details, the chosen groups of atoms, and the
        adopted reference value.

    The Lennard-Jones parameters, atomic charges, cutoff distance, boundary conditions,
    as well as whether to use a switching function and its corresponding switching
    distance, are taken from :openmm:`NonbondedForce` object.

    .. note::
        Any non-exclusion exceptions involving an atom in :math:`{\bf g}_1` and an atom
        in either :math:`{\bf g}_2` or :math:`{\bf g}_3` are turned into exclusions in
        this collective variable.

    Parameters
    ----------
    group1
        The first atom group.
    group2
        The second atom group.
    nonbondedForce
        The :class:`openmm.NonbondedForce` object from which to collect the necessary
        parameters.
    contrastGroup
        An optional third atom group to contrast the attraction strength between
        :math:`{\bf g}_1` and :math:`{\bf g}_2` with that between :math:`{\bf g}_1`
        and :math:`{\bf g}_3`.
    reference
        A reference value (in energy units per mole) to which the collective variable
        should be normalized. One can also provide an :OpenMM:`Context` object from
        which to extract a reference attraction strength. The extracted value will be
        :math:`S_{1,2}({\bf r})` for :math:`E_{\rm ref} = 1`, regardless of whether
        `contrastGroup` is provided or not.

    Examples
    --------
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.HostGuestExplicit()
    >>> guest = [a.index for a in model.topology.atoms() if a.residue.name == "B2"]
    >>> host = [a.index for a in model.topology.atoms() if a.residue.name == "CUC"]
    >>> forces = {f.getName(): f for f in model.system.getForces()}
    >>> cv1 = cvpack.AttractionStrength(guest, host, forces["NonbondedForce"])
    >>> _ = cv1.setUnusedForceGroup(0, model.system)
    >>> _ = model.system.addForce(cv1)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> integrator = openmm.VerletIntegrator(1.0 * mmunit.femtoseconds)
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> context.setPositions(model.positions)
    >>> print(cv1.getValue(context, 4))
    4912.5 dimensionless

    >>> water = [a.index for a in model.topology.atoms() if a.residue.name == "HOH"]
    >>> cv2 = cvpack.AttractionStrength(guest, water, forces["NonbondedForce"])
    >>> _ = cv2.setUnusedForceGroup(0, model.system)
    >>> _ = model.system.addForce(cv2)
    >>> context.reinitialize(preserveState=True)
    >>> print(cv2.getValue(context, 4))
    2063.3 dimensionless

    >>> cv3 = cvpack.AttractionStrength(guest, host, forces["NonbondedForce"], water)
    >>> _ = cv3.setUnusedForceGroup(0, model.system)
    >>> _ = model.system.addForce(cv3)
    >>> context.reinitialize(preserveState=True)
    >>> print(cv3.getValue(context, 4))
    2849.2 dimensionless
    >>> print(cv1.getValue(context, 4) - cv2.getValue(context, 4))
    2849.2 dimensionless

    >>> cv4 = cvpack.AttractionStrength(
    ...     guest, host, forces["NonbondedForce"], water, contrastScaling=0.5
    ... )
    >>> _ = cv4.setUnusedForceGroup(0, model.system)
    >>> _ = model.system.addForce(cv4)
    >>> context.reinitialize(preserveState=True)
    >>> print(cv4.getValue(context, 4))
    3880.8 dimensionless
    >>> print(1 * cv1.getValue(context, 4) - 0.5 * cv2.getValue(context, 4))
    3880.8...
    """

    yaml_tag = "!cvpack.AttractionStrength"

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: t.Iterable[int],
        group2: t.Iterable[int],
        nonbondedForce: openmm.NonbondedForce,
        contrastGroup: t.Optional[t.Iterable[int]] = None,
        reference: t.Union[mmunit.ScalarQuantity, openmm.Context] = mmunit.Quantity(
            1.0, mmunit.kilojoule_per_mole
        ),
        contrastScaling: float = 1.0,
    ) -> None:
        group1 = list(group1)
        group2 = list(group2)
        contrasting = contrastGroup is not None
        if contrasting:
            contrastGroup = list(contrastGroup)
        nonbondedForce = NonbondedForceSurrogate(nonbondedForce)
        cutoff = nonbondedForce.getCutoffDistance()
        expression = (
            f"-(lj + coul){'*sign1*sign2' if contrasting else ''}/ref"
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
        for parameter in ("charge", "sigma", "epsilon") + ("sign",) * contrasting:
            self.addPerParticleParameter(parameter)
        for atom in range(nonbondedForce.getNumParticles()):
            charge, sigma, epsilon = nonbondedForce.getParticleParameters(atom)
            if contrasting:
                sign = -1 if atom in contrastGroup else 1
                scale = contrastScaling if atom in contrastGroup else 1
                self.addParticle([charge * scale, sigma, epsilon * scale**2, sign])
            else:
                self.addParticle([charge, sigma, epsilon])
        for exception in range(nonbondedForce.getNumExceptions()):
            i, j, *_ = nonbondedForce.getExceptionParameters(exception)
            self.addExclusion(i, j)
        self.setUseSwitchingFunction(nonbondedForce.getUseSwitchingFunction())
        self.setSwitchingDistance(nonbondedForce.getSwitchingDistance())
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        if isinstance(reference, openmm.Context):
            reference = evaluate_in_context(self, reference)
        self.setEnergyFunction(expression.replace("ref = 1", f"ref = {reference}"))
        if contrasting:
            self.addInteractionGroup(group1, contrastGroup)
        # pylint: disable=duplicate-code
        self._registerCV(
            mmunit.dimensionless,
            group1,
            group2,
            nonbondedForce,
            contrastGroup,
            reference,
            contrastScaling,
        )
