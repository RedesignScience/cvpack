"""
.. class:: HelixAngleContent
   :platform: Linux, MacOS, Windows
   :synopsis: Alpha-helix angle content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import app as mmapp

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable, SerializableResidue


class HelixAngleContent(openmm.CustomAngleForce, BaseCollectiveVariable):
    r"""
    The alpha-helix angle content of a sequence of `n` residues:

    .. math::

        \alpha_{\theta}({\bf r}) = \sum_{i=2}^{n-1} B_m\left(
            \frac{\theta^\alpha_i({\bf r}) - \theta_{\rm ref}}{\theta_{\rm tol}}
        \right)

    where :math:`\theta^\alpha_i` is the angle formed by the alpha-carbon atoms of
    residues :math:`i-1`, :math:`i`, and :math:`i+1`, :math:`\theta_{\rm ref}` is its
    reference value in an alpha helix, and :math:`\theta_{\rm tol}` is the threshold
    tolerance around this reference. :math:`B_m(x)` is a smooth boxcar function given by

    .. math::
        B_m(x) = \frac{1}{1 + x^{2m}}

    where :math:`m` is an integer parameter that controls its steepness. Note that
    :math:`x` needs to be elevated to an even power for :math:`B_m(x)` to be an even
    function.

    Optionally, this collective variable can be normalized to the range :math:`[0, 1]`.

    .. note::

        The residues must be a contiguous sequence from a single chain, ordered from the
        N- to the C-terminus. Due to an OpenMM limitation, the maximum supported number
        of residues is 37.

    Parameters
    ----------
        residues
            The residues in the sequence
        pbc
            Whether to use periodic boundary conditions
        thetaReference
            The reference value of the
            :math:`{\rm C}_\alpha{\rm -C}_\alpha{\rm -C}_\alpha` angle in an alpha
            helix
        tolerance
            The threshold tolerance around the reference values
        halfExponent
            The parameter :math:`m` of the boxcar function
        normalize
            Whether to normalize the collective variable to the range :math:`[0, 1]`

    Raises
    ------
        ValueError
            If some residue does not contain a :math:`{\rm C}_\alpha` atom

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
        >>> print(*[r.name for r in residues])  # doctest: +ELLIPSIS
        LYS ASP GLU ... ILE LEU ARG
        >>> helix_content = cvpack.HelixAngleContent(residues)
        >>> helix_content.setUnusedForceGroup(0, model.system)
        1
        >>> model.system.addForce(helix_content)
        6
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> integrator = openmm.VerletIntegrator(0)
        >>> context = openmm.Context(model.system, integrator, platform)
        >>> context.setPositions(model.positions)
        >>> print(helix_content.getValue(context, digits=6))
        18.76058 dimensionless
    """

    @mmunit.convert_quantities
    def __init__(  # pylint: disable=too-many-arguments
        self,
        residues: t.Sequence[mmapp.topology.Residue],
        pbc: bool = False,
        thetaReference: mmunit.ScalarQuantity = mmunit.Quantity(88, mmunit.degrees),
        tolerance: mmunit.ScalarQuantity = mmunit.Quantity(15, mmunit.degrees),
        halfExponent: int = 3,
        normalize: bool = False,
    ) -> None:
        def find_alpha_carbon(residue: mmapp.topology.Residue) -> int:
            for atom in residue.atoms():
                if atom.name == "CA":
                    return atom.index
            raise ValueError(
                f"Could not find atom CA in residue {residue.name}{residue.id}"
            )

        num_angles = len(residues) - 2
        numerator = 1 / num_angles if normalize else 1
        super().__init__(
            f"{numerator}/(1+x^{2*halfExponent})"
            f"; x=(theta-{thetaReference})/{tolerance}"
        )
        for i in range(1, len(residues) - 1):
            self.addAngle(
                find_alpha_carbon(residues[i - 1]),
                find_alpha_carbon(residues[i]),
                find_alpha_carbon(residues[i + 1]),
                [],
            )
        self.setUsesPeriodicBoundaryConditions(pbc)
        self._registerCV(  # pylint: disable=duplicate-code
            mmunit.dimensionless,
            list(map(SerializableResidue, residues)),
            pbc,
            thetaReference,
            tolerance,
            halfExponent,
            normalize,
        )
