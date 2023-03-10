"""
.. class:: HelixHBondContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix hydrogen-bond content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import re as regex
from typing import Iterable, Pattern

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from .cvpack import (
    AbstractCollectiveVariable,
    QuantityOrFloat,
    SerializableResidue,
    in_md_units,
)


class HelixHBondContent(openmm.CustomBondForce, AbstractCollectiveVariable):
    """
    The alpha-helix hydrogen-bond content of a sequence of `n` residues:

    .. math::

        \\alpha_{\\rm HB}({\\bf r}) = \\sum_{i=5}^n B_m\\left(
            \\frac{\\| {\\bf r}^{\\rm H}_i - {\\bf r}^{\\rm O}_{i-4} \\|}{d_{\\rm HB}}
        \\right)

    where :math:`{\\bf r}^{\\rm H}_k` and :math:`{\\bf r}^{\\rm O}_k` are the positions of the
    hydrogen and oxygen atoms bonded, respectively, to the backbone nitrogen and carbon atoms of
    residue :math:`k`. In addition, :math:`d_{\\rm HB}` is the threshold distance for a hydrogen
    bond and :math:`B_m(x)` is a smooth step function given by

    .. math::
        B_m(x) = \\frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from the N- to
        the C-terminus. Due to an OpenMM limitation, the maximum supported number of residues is 37.

    Parameters
    ----------
        residues
            The residues in the sequence
        pbc
            Whether to use periodic boundary conditions
        thresholdDistance
            The threshold distance for a hydrogen bond
        halfExponent
            The parameter :math:`m` of the step function.

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> residues = [r for r in model.topology.residues() if 59 <= r.index <= 79]
        >>> print(*[r.name for r in residues])
        LYS ASP GLU ALA GLU LYS LEU PHE ASN GLN ASP VAL ASP ALA ALA VAL ARG GLY ILE LEU ARG
        >>> helix_content = cvpack.HelixHBondContent(residues)
        >>> model.system.addForce(helix_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(helix_content.getValue(context, digits=6))
        15.88038 dimensionless
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: Iterable[mmapp.topology.Residue],
        pbc: bool = False,
        thresholdDistance: QuantityOrFloat = 0.33 * mmunit.nanometers,
        halfExponent: int = 3,
        normalize: bool = False,
    ) -> None:
        def find_atom(residue: mmapp.topology.Residue, pattern: Pattern) -> int:
            for atom in residue.atoms():
                if regex.match(pattern, atom.name):
                    return atom.index
            raise ValueError(
                f"Could not find atom matching regex "
                f"'{pattern.pattern}'"
                f" in residue {residue.name}{residue.id}"
            )

        threshold = in_md_units(thresholdDistance)
        numerator = 1 / (len(residues) - 4) if normalize else 1
        super().__init__(f"{numerator}/(1+x^{2*halfExponent}); x=r/{threshold}")
        hydrogen_pattern = regex.compile("\\b(H|1H|HN1|HT1|H1|HN)\\b")
        oxygen_pattern = regex.compile("\\b(O|OCT1|OC1|OT1|O1)\\b")
        for i in range(4, len(residues)):
            self.addBond(
                find_atom(residues[i - 4], oxygen_pattern),
                find_atom(residues[i], hydrogen_pattern),
                [],
            )
        self.setUsesPeriodicBoundaryConditions(pbc)
        res = [SerializableResidue(r) for r in residues]
        self._registerCV(mmunit.dimensionless, res, pbc, threshold, halfExponent, normalize)
