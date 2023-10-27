"""
.. class:: HelixTorsionContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import sys
from typing import List, Sequence

import numpy as np
import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable, SerializableResidue
from .rmsd import RMSD

if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources


class HelixRMSDContent(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    The alpha-helix RMSD content of a sequence of `n` residues :cite:`Pietrucci_2009`:

    .. math::

        \\alpha_{\\rm rmsd}({\\bf r}) = \\sum_{i=1}^{n-5} B_m\\left(
            \\sqrt{\\frac{1}{30} \\sum_{j=1}^{30} \\left\\|
                \\hat{\\bf r}_j({\\bf g}_i) -
                {\\bf A}({\\bf g}_i)\\hat{\\bf r}_j({\\bf g}_{\\rm ref})
            \\right\\|^2}
        \\right)

    where :math:`{\\bf g}_i` represents a group of 30 atoms selected from residues :math:`i` to
    :math:`i+5` of the sequence, :math:`{\\bf g}_{\\rm ref}` represents the same 30 atoms in an
    ideal alpha-helix configuration, :math:`\\hat{\\bf r}_j({\\bf g})` is the position of the
    :math:`j`-th atom in a group :math:`{\\bf g}` relative to the group's center of geometry
    (centroid), :math:`{\\bf A}({\\bf g})` is the rotation matrix that minimizes the RMSD between
    :math:`{\\bf g}` and :math:`{\\bf g}_{\\rm ref}`, and :math:`B_m(x)` is a smooth step function
    given by

    .. math::
        B_m(x) = \\frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness.

    Each group :math:`{\\bf g}_i` is formed by the N, :math:`{\\rm C}_\\alpha`, C, and O atoms of
    the backbone, as well as the :math:`{\\rm C}_\\beta` atoms of the six consecutive residues
    starting from residue :math:`i`. In the case of glycine, the missing :math:`{\\rm C}_\\beta` is
    replaced by the corresponding H atom.

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from the N- to
        the C-terminus. Due to an OpenMM limitation, the maximum supported number of residues is 37.

    This collective variable was introduced in Ref. :cite:`Pietrucci_2009` with a slightly
    different step function. The ideal alpha-helix configuration is the same used in `PLUMED v2.8.1
    <https://github.com/plumed/plumed2>`_ for its collective variable `ALPHARMSD`_. By setting
    :math:`{\\rm NN}=2m` and :math:`{\\rm MM}=4m`, PLUMED's ALPHARMSD will match the collective
    variable implemented here.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. _ALPHARMSD: https://www.plumed.org/doc-v2.8/user-doc/html/_a_l_p_h_a_r_m_s_d.html

    .. warning::

        Periodic boundary conditions are `not supported
        <https://github.com/openmm/openmm/issues/2913>`_.

    Parameters
    ----------
        residues
            The residues to be used in the calculation.
        numAtoms
            The total number of atoms in the system (required by OpenMM).
        thresholdRMSD
            The threshold RMSD value for the step function.
        halfExponent
            The parameter :math:`m` of the step function.
        normalize
            Whether to normalize the collective variable to the range :math:`[0, 1]`.

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
        >>> helix_content = cvpack.HelixRMSDContent(residues, model.system.getNumParticles())
        >>> model.system.addForce(helix_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(helix_content.getValue(context, digits=7))
        15.915198 dimensionless
    """

    _ideal_helix_positions = 0.1 * np.loadtxt(
        str(
            resources.files("cvpack").joinpath("data").joinpath("ideal_alpha_helix.csv")
        ),
        delimiter=",",
    )

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: Sequence[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: mmunit.ScalarQuantity = mmunit.Quantity(0.08, mmunit.nanometers),
        halfExponent: int = 3,
        normalize: bool = False,
    ) -> None:
        assert (
            6 <= len(residues) <= 37
        ), "The number of residues must be between 6 and 37"

        def step_function(i):
            return f"1/(1 + (rmsd{i+1}/{thresholdRMSD})^{2*halfExponent})"

        def atoms_list(residue: mmapp.topology.Residue) -> List[int]:
            indices = {}
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "CB", "C", "O"]:
                    indices[atom.name] = atom.index
                elif residue.name == "GLY" and atom.name == "HA2":
                    indices["CB"] = atom.index
            if len(indices) != 5:
                raise ValueError(
                    f"Could not find all atoms in residue {residue.name}{residue.id}"
                )
            return [indices[atom] for atom in ["N", "CA", "CB", "C", "O"]]

        num_residue_blocks = len(residues) - 5
        function = " + ".join(map(step_function, range(num_residue_blocks)))
        super().__init__(
            f"({function})/{num_residue_blocks}" if normalize else function
        )
        atoms = [atoms_list(r) for r in residues]
        positions = [openmm.Vec3(*x) for x in self._ideal_helix_positions]
        for i in range(num_residue_blocks):
            group = sum(atoms[i : i + 6], [])
            self.addCollectiveVariable(f"rmsd{i+1}", RMSD(positions, group, numAtoms))

        self._registerCV(  # pylint: disable=duplicate-code
            mmunit.dimensionless,
            list(map(SerializableResidue, residues)),
            numAtoms,
            thresholdRMSD,
            halfExponent,
            normalize,
        )
