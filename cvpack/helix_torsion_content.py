"""
.. class:: HelixTorsionContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix torsion content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable, SerializableResidue


class HelixTorsionContent(openmm.CustomTorsionForce, BaseCollectiveVariable):
    r"""
    The alpha-helix Ramachandran content of a sequence of `n` residues:

    .. math::

        \alpha_{\phi,\psi}({\bf r}) = \frac{1}{2} \sum_{i=2}^{n-1} \left[
            B_m\left(
                \frac{\phi_i({\bf r}) - \phi_{\rm ref}}{\theta_{\rm tol}}
            \right) +
            B_m\left(
                \frac{\psi_i({\bf r}) - \psi_{\rm ref}}{\theta_{\rm tol}}
            \right)
        \right]

    where :math:`\phi_i({\bf r})` and :math:`\psi_i({\bf r})` are the Ramachandran
    dihedral angles of residue :math:`i`, :math:`\phi_{\rm ref}` and
    :math:`\psi_{\rm ref}` are their reference values in an alpha helix
    :cite:`Hovmoller_2002`, and :math:`\theta_{\rm tol}` is the threshold tolerance
    around these refenrences. :math:`B_m(x)` is a smooth boxcar function given by

    .. math::
        B_m(x) = \frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness. Note that
    :math:`x` needs to be elevated to an even power for :math:`B_m(x)` to be an even
    function.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. note::

        The :math:`\phi` and :math:`\psi` angles of the first and last residues are
        not considered. They are used to compute the dihedral angles of the second and
        penultimate residues, respectively.

        The residues must be a contiguous sequence from a single chain, ordered from the
        N- to the C-terminus. Due to an OpenMM limitation, the maximum supported number
        of residues is 37.

    Parameters
    ----------
        residues
            The residues in the sequence
        pbc
            Whether to use periodic boundary conditions
        phiReference
            The reference value of the phi dihedral angle in an alpha helix
        psiReference
            The reference value of the psi dihedral angle in an alpha helix
        tolerance
            The threshold tolerance around the reference values
        halfExponent
            The parameter :math:`m` of the boxcar function
        normalize
            Whether to normalize the collective variable to the range :math:`[0, 1]`

    Raises
    ------
        ValueError
            If some residue does not contain a :math:`\phi` or :math:`\psi` angle

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
        >>> helix_content = cvpack.HelixTorsionContent(residues)
        >>> helix_content.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(helix_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(helix_content.getValue(context, digits=6))
        17.45285 dimensionless
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        pbc: bool = False,
        phiReference: mmunit.ScalarQuantity = mmunit.Quantity(-63.8, mmunit.degrees),
        psiReference: mmunit.ScalarQuantity = mmunit.Quantity(-41.1, mmunit.degrees),
        tolerance: mmunit.ScalarQuantity = mmunit.Quantity(25, mmunit.degrees),
        halfExponent: int = 3,
        normalize: bool = False,
    ) -> None:
        def find_atom(residue: mmapp.topology.Residue, name: str) -> int:
            for atom in residue.atoms():
                if atom.name == name:
                    return atom.index
            raise ValueError(
                f"Could not find atom {name} in residue {residue.name}{residue.id}"
            )

        numerator = 1 / (2 * (len(residues) - 2)) if normalize else 1 / 2
        super().__init__(
            f"{numerator}/(1+x^{2*halfExponent}); x=(theta-theta_ref)/{tolerance}"
        )
        self.addPerTorsionParameter("theta_ref")
        for i in range(1, len(residues) - 1):
            self.addTorsion(
                find_atom(residues[i - 1], "C"),
                find_atom(residues[i], "N"),
                find_atom(residues[i], "CA"),
                find_atom(residues[i], "C"),
                [phiReference],
            )
            self.addTorsion(
                find_atom(residues[i], "N"),
                find_atom(residues[i], "CA"),
                find_atom(residues[i], "C"),
                find_atom(residues[i + 1], "N"),
                [psiReference],
            )
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(  # pylint: disable=duplicate-code
            mmunit.dimensionless,
            list(map(SerializableResidue, residues)),
            pbc,
            phiReference,
            psiReference,
            tolerance,
            halfExponent,
        )
