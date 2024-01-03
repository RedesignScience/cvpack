"""
.. class:: HelixRMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .cvpack import SerializableResidue
from .rmsd import RMSD
from .rmsd_content import RMSDContent


class HelixRMSDContent(RMSDContent):
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
        the N-terminus to the C-terminus. The minimum and maximum numbers of residues
        supported in this implementation are 6 and 1029, respectively.

    .. _ALPHARMSD: https://www.plumed.org/doc-v2.8/user-doc/html/_a_l_p_h_a_r_m_s_d.html

    Parameters
    ----------
        residues
            The residues to be used in the calculation.
        numAtoms
            The total number of atoms in the system (required by OpenMM).
        thresholdRMSD
            The threshold RMSD value for considering a group of residues as matching an
            alpha-helix.
        stepFunction
            The form of the step function :math:`S(x)`.
        normalize
            Whether to normalize the collective variable to the range :math:`[0, 1]`.

    Example
    -------
        >>> import cvpack
        >>> import openmm
        >>> from openmm import app, unit
        >>> from openmmtools import testsystems
        >>> model = testsystems.LysozymeImplicit()
        >>> residues = [
        ...     r
        ...     for r in model.topology.residues()
        ...     if 59 <= r.index <= 79
        ... ]
        >>> print(*[r.name for r in residues])
        LYS ASP GLU ... ILE LEU ARG
        >>> helix_content = cvpack.HelixRMSDContent(
        ...     residues, model.system.getNumParticles()
        ... )
        >>> model.system.addForce(helix_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(helix_content.getValue(context, digits=4))
        15.981 dimensionless
    """

    _ideal_helix_positions = RMSDContent.load_positions("ideal_alpha_helix.csv")

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: mmunit.ScalarQuantity = mmunit.Quantity(0.08, mmunit.nanometers),
        stepFunction: str = "(1+x^4)/(1+x^4+x^8)",
        normalize: bool = False,
    ) -> None:
        assert (
            6 <= len(residues) <= 1029
        ), "The number of residues must be between 6 and 1029"
        num_residue_blocks = len(residues) - 5
        atoms = list(map(self.getAtomList, residues))
        positions = [openmm.Vec3(*x) for x in self._ideal_helix_positions]

        def expression(start, end):
            summands = []
            definitions = []
            for i in range(start, min(end, num_residue_blocks)):
                summands.append(stepFunction.replace("x", f"x{i}"))
                definitions.append(f"x{i}=rmsd{i}/{thresholdRMSD}")
            return ";".join(["+".join(summands)] + definitions)

        if num_residue_blocks <= 32:
            summation = expression(0, num_residue_blocks)
            force = self
        else:
            summation = "+".join(
                f"chunk{i}" for i in range((num_residue_blocks + 31) // 32)
            )
        super().__init__(
            f"({summation})/{num_residue_blocks}" if normalize else summation
        )
        for index in range(num_residue_blocks):
            if num_residue_blocks > 32 and index % 32 == 0:
                force = openmm.CustomCVForce(expression(index, index + 32))
                self.addCollectiveVariable(f"chunk{index//32}", force)
            force.addCollectiveVariable(
                f"rmsd{index}",
                RMSD(positions, sum(atoms[index : index + 6], []), numAtoms),
            )

        self._registerCV(  # pylint: disable=duplicate-code
            mmunit.dimensionless,
            list(map(SerializableResidue, residues)),
            numAtoms,
            thresholdRMSD,
            stepFunction,
            normalize,
        )
