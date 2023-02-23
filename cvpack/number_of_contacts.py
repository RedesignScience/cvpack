"""
.. class:: NumberOfContacts
   :platform: Linux, MacOS, Windows
   :synopsis: The number of contacts between two atom groups

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable

import openmm
from openmm import unit as mmunit

from .cvpack import AbstractCollectiveVariable, QuantityOrFloat, in_md_units


class NumberOfContacts(openmm.CustomNonbondedForce, AbstractCollectiveVariable):
    """
    The number of contacts between two atom groups:

    .. math::
        N({\\bf r}) = \\sum_{i \\in {\\bf g}_1} \\sum_{j \\in {\\bf g}_2}
                        S\\left(\\frac{\\|{\\bf r}_j - {\\bf r}_i\\|}{r_0}\\right)

    where :math:`r_0` is the threshold distance for defining a contact and :math:`S(x)` is a step
    function equal to 1 if a contact is made or equal to 0 otherwise. In analysis, it is fine to
    make :math:`S(x) = H(1-x)`, where `H` is the `Heaviside step function
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_. In a simulation, however,
    :math:`S(x)` should continuously approximate :math:`H(1-x)` for :math:`x \\geq 0`. By default
    :cite:`Iannuzzi_2003`,

    .. math::

        S(x) = \\frac{1-x^6}{1-x^{12}} = \\frac{1}{1+x^6}

    Atom pairs are ignored for distances beyond a cutoff :math:`r_c`. To avoid discontinuities,
    a switching function is applied at :math:`r_s \\leq r \\leq r_c` to make :math:`S(r/r_0)`
    smoothly decay to zero.

    .. note::

        The two groups are allowed to overlap. In this case, terms with :math:`j = i`
        (self-contacts) are ignored and each combination with :math:`j \\neq i` is counted
        only once.

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
            The threshold distance for considering two atoms as being in contact
        cutoffDistance
            The distance beyond which an atom pair will be ignored
        switchingDistance
            The distance beyond which a swithing function will be applied

    Example
    -------
        >>> import cvpack
        >>> import openmm as mm
        >>> from openmm import app
        >>> from openmmtools import testsystems
        >>> model = testsystems.AlanineDipeptideVacuum()
        >>> carbons = [a.index for a in model.topology.atoms() if a.element == app.element.carbon]
        >>> num_atoms = model.topology.getNumAtoms()
        >>> optionals = {"pbc": False, "stepFunction": "step(1-x)"}
        >>> nc = cvpack.NumberOfContacts(carbons, carbons, num_atoms, **optionals)
        >>> model.system.addForce(nc)
        5
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
        >>> context.setPositions(model.positions)
        >>> print(nc.getValue(context, digits=6))
        6.0 dimensionless

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        group1: Iterable[int],
        group2: Iterable[int],
        numAtoms: int,
        pbc: bool = False,
        stepFunction: str = "1/(1+x^6)",
        thresholdDistance: QuantityOrFloat = 0.3,
        cutoffDistance: QuantityOrFloat = 0.6,
        switchingDistance: QuantityOrFloat = 0.5,
    ) -> None:
        super().__init__(stepFunction + f"; x=r/{in_md_units(thresholdDistance)}")
        nonbonded_method = self.CutoffPeriodic if pbc else self.CutoffNonPeriodic
        self.setNonbondedMethod(nonbonded_method)
        for _ in range(numAtoms):
            self.addParticle([])
        self.setUseSwitchingFunction(True)
        self.setCutoffDistance(cutoffDistance)
        self.setSwitchingDistance(switchingDistance)
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
        self._registerCV(
            mmunit.dimensionless,
            group1,
            group2,
            numAtoms,
            pbc,
            stepFunction,
            in_md_units(thresholdDistance),
            in_md_units(cutoffDistance),
            in_md_units(switchingDistance),
        )
