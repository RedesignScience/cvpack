"""
.. class:: NumberOfContacts
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable


class NumberOfContacts(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The number of contacts between two atom groups:

    .. math::
        N({\\bf r}) = \\sum_{i \\in {\\bf g}_1} \\sum_{j \\in {\\bf g}_2}
                        S\\left(\\frac{\\|{\\bf r}_j - {\\bf r}_i\\|}{r_0}\\right)

    where :math:`r_0` is the threshold distance for defining a contact and :math:`S(x)`
    is a step function equal to :math:`1` if a contact is made or equal to :math:`0`
    otherwise. For trajectory analysis, it is fine to make :math:`S(x) = H(1-x)`, where
    `H` is the `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. For molecular dynamics,
    however, :math:`S(x)` should be a continuous approximation of :math:`H(1-x)` for
    :math:`x \\geq 0`. By default :cite:`Iannuzzi_2003`, the following function is used:

    .. math::

        S(x) = \\frac{1-x^6}{1-x^{12}} = \\frac{1}{1+x^6}

    In fact, a cutoff distance :math:`r_c = x_c r_0` (typically, :math:`x_c = 2`) is
    applied so that :math:`S(x) = 0` for :math:`x \\geq x_c`. To avoid discontinuities,
    there is also the option to smoothly switch off :math:`S(x)` starting from
    :math:`r_s = x_s r_0` (typically, :math:`x_s = 1.5`) instead of doing it abruptly at
    :math:`r_c`.

    .. note::

        Atoms are allowed to be in both groups. In this case, self-contacts
        (:math:`i = j`) are ignored and each pair of distinct atoms (:math:`i \\neq j`)
        is counted only once.

    Parameters
    ----------
        group1
            The indices of the atoms in the first group
        group2
            The indices of the atoms in the second group
        numAtoms
            The total number of atoms in the system (required by OpenMM)
        pbc
            Whether the system has periodic boundary conditions
        stepFunction
            The function "step(1-x)" (for analysis only) or a continuous approximation
            thereof
        thresholdDistance
            The threshold distance (:math:`r_0`) for considering two atoms as being in
            contact
        cutoffFactor
            The factor :math:`x_c` that multiplies the threshold distance to define
            the cutoff distance
        switchFactor
            The factor :math:`x_s` that multiplies the threshold distance to define
            the distance at which the step function starts switching off smoothly.
            If None, it switches off abruptly at the cutoff distance.

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> carbons = [
        ...     a.index
        ...     for a in model.topology.atoms()
        ...     if a.element == app.element.carbon
        ... ]
        >>> num_atoms = model.topology.getNumAtoms()
        >>> optionals = {"pbc": False, "stepFunction": "step(1-x)"}
        >>> nc = cvpack.NumberOfContacts(
        ...     carbons, carbons, num_atoms, **optionals
        ... )
        >>> model.system.addForce(nc)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(
        ...     model.system, openmm.CustomIntegrator(0), platform
        ... )
        >>> context.setPositions(model.positions)
        >>> print(nc.getValue(context, digits=6))
        6.0 dimensionless

    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: t.Sequence[int],
        group2: t.Sequence[int],
        numAtoms: int,
        pbc: bool,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: mmunit.ScalarQuantity = mmunit.Quantity(
            0.3, mmunit.nanometers
        ),
        cutoffFactor: float = 2.0,
        switchFactor: t.Optional[float] = 1.5,
    ) -> None:
        super().__init__(stepFunction + f"; x=r/{thresholdDistance}")
        nonbonded_method = self.CutoffPeriodic if pbc else self.CutoffNonPeriodic
        self.setNonbondedMethod(nonbonded_method)
        for _ in range(numAtoms):
            self.addParticle([])
        self.setCutoffDistance(cutoffFactor * thresholdDistance)
        use_switching_function = switchFactor is not None
        self.setUseSwitchingFunction(use_switching_function)
        if use_switching_function:
            self.setSwitchingDistance(switchFactor * thresholdDistance)
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        self._registerCV(
            mmunit.dimensionless,
            group1,
            group2,
            numAtoms,
            pbc,
            stepFunction,
            thresholdDistance,
            cutoffFactor,
            switchFactor,
        )
