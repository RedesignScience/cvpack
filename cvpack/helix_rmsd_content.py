"""
.. class:: HelixTorsionContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Iterable, List

import numpy as np
import openmm

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from openmm import app as mmapp
from openmm import unit as mmunit

from .cvpack import (
    AbstractCollectiveVariable,
    QuantityOrFloat,
    SerializableResidue,
    in_md_units,
)
from .rmsd import RMSD


class HelixRMSDContent(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    The alpha-helix RMSD content of a sequence of `n` residues :cite:`Pietrucci_2009`:

    .. math::

        \\alpha_{\\rm rmsd}({\\bf r}) = \\sum_{i=1}^{n-5} B_m\\left(
            \\sqrt{\\frac{1}{30} \\sum_{j=1}^{30} \\left\\|
                {\\bf r}_j({\\bf g}_i) - {\\bf R}({\\bf g}_i){\\bf r}_j({\\bf g}_{\\rm ideal})
            \\right\\|^2}
        \\right)

    where :math:`{\\bf g}_i` represents a group of 30 atoms selected from residues :math:`i` to
    :math:`i+5` of the sequence, :math:`{\\bf g}_{\\rm ideal}` represents the same 30 atoms in an
    ideal alpha-helix configuration, :math:`{\\bf r}_j({\\bf g})` is the position of the
    :math:`j`-th atom in a group :math:`{\\bf g}` relative to the centroid of the group,
    :math:`{\\bf R}({\\bf g})` is the rotation matrix that minimizes the RMSD between
    :math:`{\\bf g}` and :math:`{\\bf g}_{\\rm ideal}`, and :math:`B_m(x)` is a smooth step function
    given by

    .. math::
        B_m(x) = \\frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness.

    Each group :math:`{\\bf g}_i` is formed by the N, :math:`{\\rm C}_\\alpha`, C, and O atoms of
    the backbone, as well as the :math:`{\\rm C}_\\beta` atoms of the six consecutive residues
    starting from residue :math:`i`. In the case of glycine, the missing :math:`{\\rm C}_\\beta` is
    replaced by the corresponding H atom.

    This collective variable was introduced in :cite:`Pietrucci_2009`, but with a slightly
    different step function. The ideal alpha-helix configuration used here is the same used in
    `PLUMED v2.8.1 <https://github.com/plumed/plumed2>`_ to compute the collective variable
    `ALPHARMSD`_, which can match the present implementation by setting :math:`{\\rm NN}=m` and
    :math:`{\\rm MM}=2m`.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. _ALPHARMSD: https://www.plumed.org/doc-v2.8/user-doc/html/_a_l_p_h_a_r_m_s_d.html

    .. warning::

        Periodic boundary conditions are not supported (see OpenMM issue `#2392
        <https://github.com/openmm/openmm/issues/2913>`_).

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
        >>> import openmm as mm
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
        files("cvpack").joinpath("data").joinpath("ideal_alpha_helix.csv"),
        delimiter=",",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: Iterable[mmapp.topology.Residue],
        numAtoms: int,
        thresholdRMSD: QuantityOrFloat = 0.08 * mmunit.nanometers,
        halfExponent: int = 3,
        normalize: bool = False,
    ) -> None:
        assert len(residues) >= 6, "Must have at least 6 residues"

        def step_function(i):
            return f"1/(1 + (rmsd{i+1}/{in_md_units(thresholdRMSD)})^{2*halfExponent})"

        def atoms_list(residue: mmapp.topology.Residue) -> List[int]:
            indices = {}
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "CB", "C", "O"]:
                    indices[atom.name] = atom.index
                elif residue.name == "GLY" and atom.name == "HA2":
                    indices["CB"] = atom.index
            if len(indices) != 5:
                raise ValueError(f"Could not find all atoms in residue {residue.name}{residue.id}")
            return [indices[atom] for atom in ["N", "CA", "CB", "C", "O"]]

        num_residue_blocks = len(residues) - 5
        function = " + ".join(map(step_function, range(num_residue_blocks)))
        super().__init__(f"({function})/{num_residue_blocks}" if normalize else function)
        atoms = [atoms_list(r) for r in residues]
        positions = [openmm.Vec3(*x) for x in self._ideal_helix_positions]
        for i in range(num_residue_blocks):
            group = sum(atoms[i : i + 6], [])
            self.addCollectiveVariable(f"rmsd{i+1}", RMSD(positions, group, numAtoms))

        self._registerCV(
            mmunit.dimensionless,
            [SerializableResidue(r) for r in residues],
            numAtoms,
            in_md_units(thresholdRMSD),
            halfExponent,
            normalize,
        )
