"""
.. class:: SheetRMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Beta-sheet RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import app as mmapp

from cvpack import unit as mmunit

from .cvpack import SerializableResidue
from .rmsd_content import RMSDContent

# pylint: disable=protected-access
PARABETA_POSITIONS = RMSDContent._loadPositions("ideal_parallel_beta_sheet.csv")
ANTIBETA_POSITIONS = RMSDContent._loadPositions("ideal_antiparallel_beta_sheet.csv")
# pylint: enable=protected-access


class SheetRMSDContent(RMSDContent):
    """
    The alpha-helix RMSD content of a sequence of `n` residues :cite:`Pietrucci_2009`:

    .. math::

        \\alpha_{\\rm rmsd}({\\bf r}) = \\sum_{i=1}^{n-5} S\\left(
            \\frac{r_{\\rm rmsd}({\\bf g}_i, {\\bf g}_{\\rm ref})}{r_0}
        \\right)

    where :math:`{\\bf g}_i` represents a group of atoms selected from residues
    :math:`i` to :math:`i+5` of the sequence, :math:`{\\bf g}_{\\rm ref}` represents
    the same atoms in an ideal alpha-helix configuration,
    :math:`r_{\\rm rmsd}({\\bf g}_i, {\\bf g}_{\\rm ref})` is the root-mean-square
    distance (RMSD) between :math:`{\\bf g}_i` and :math:`{\\bf g}_{\\rm ref}`,
    :math:`r_0` is a threshold RMSD value, and :math:`S(x)` is a smooth step function
    whose default form is

    .. math::
        S(x) = \\frac{1 + x^4}{1 + x^4 + x^8}

    Each group :math:`{\\bf g}_i` is formed by the N, :math:`{\\rm C}_\\alpha`,
    :math:`{\\rm C}_\\beta`, C, and O atoms of consecutive residues from :math:`i`
    to :math:`i+5`, thus comprising a total of 30 atoms.  In the case glycine, the
    missing :math:`{\\rm C}_\\beta` is replaced by the corresponding H atom. The
    root-mean-square distance is then defined as

    .. math::

        r_{\\rm rmsd}({\\bf g}_i, {\\bf g}_{\\rm ref}) =
            \\sqrt{\\frac{1}{30} \\sum_{j=1}^{30} \\left\\|
                \\hat{\\bf r}_j({\\bf g}_i) -
                {\\bf A}({\\bf g}_i)\\hat{\\bf r}_j({\\bf g}_{\\rm ref})
            \\right\\|^2}

    where :math:`\\hat{\\bf r}_j({\\bf g})` is the position of the :math:`j`-th atom in
    a group :math:`{\\bf g}` relative to the group's center of geometry (centroid),
    :math:`{\\bf A}({\\bf g})` is the rotation matrix that minimizes the RMSD between
    :math:`{\\bf g}` and :math:`{\\bf g}_{\\rm ref}`.

    Optionally, alpha-helix RMSD content can be normalized to the range :math:`[0, 1]`.

    This collective variable was introduced in Ref. :cite:`Pietrucci_2009`. The default
    step function shown above is identical to the one in the original paper, but written
    in a numerically safe form. The ideal alpha-helix configuration is the same used for
    the collective variable `ALPHARMSD`_ in `PLUMED v2.8.1
    <https://github.com/plumed/plumed2>`_ .

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from
        the N-terminus to the C-terminus. This implementation is limited to a minimum
        of 6 and a maximum of 1029 residues.

    .. _ALPHARMSD: https://www.plumed.org/doc-v2.8/user-doc/html/_a_l_p_h_a_r_m_s_d.html

    Parameters
    ----------
        residues
            The residues to be used in the calculation.
        numAtoms
            The total number of atoms in the system (required by OpenMM).
        parallel
            Whether to consider a parallel beta sheet instead of an antiparallel one.
        thresholdRMSD
            The threshold RMSD value for considering a group of residues as matching an
            alpha-helix.
        stepFunction
            The form of the step function :math:`S(x)`.
        normalize
            Whether to normalize the collective variable to the range :math:`[0, 1]`.

    Example
    -------
        >>> import itertools as it
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.SrcImplicit()
        >>> residues = list(it.islice(model.topology.residues(), 8, 41))
        >>> print(*[r.name for r in residues])
        SER LEU ARG ... THR LEU LYS
        >>> sheet_content = cvpack.SheetRMSDContent(
        ...     residues, model.system.getNumParticles()
        ... )
        >>> sheet_content.getNumResidueBlocks()
        325
        >>> model.system.addForce(sheet_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(sheet_content.getValue(context, digits=4))
        4.1366 dimensionless
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        numAtoms: int,
        parallel: bool = False,
        thresholdRMSD: mmunit.ScalarQuantity = mmunit.Quantity(0.08, mmunit.nanometers),
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
    ) -> None:
        min_separation = 3 if parallel else 2
        residue_blocks = [
            [i, i + 1, i + 2, j, j + 1, j + 2]
            for i in range(len(residues) - 6 - min_separation)
            for j in range(i + 3 + min_separation, len(residues) - 3)
        ]
        # pylint: disable=duplicate-code
        super().__init__(
            residue_blocks,
            PARABETA_POSITIONS if parallel else ANTIBETA_POSITIONS,
            residues,
            numAtoms,
            thresholdRMSD,
            stepFunction,
            normalize,
        )
        self._registerCV(
            mmunit.dimensionless,
            list(map(SerializableResidue, residues)),
            numAtoms,
            parallel,
            thresholdRMSD,
            stepFunction,
            normalize,
        )
