"""
.. class:: BaseRadiusOfGyration
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for the radius of gyration of an atom group

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm

from .cvpack import BaseCollectiveVariable


class BaseRadiusOfGyration(openmm.CustomCentroidBondForce, BaseCollectiveVariable):
    """
    Abstract class for the radius of gyration of a group of `n` atoms.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_groups: int,
        expression: str,
        group: t.Sequence[int],
        pbc: bool = False,
        weighByMass: bool = False,
    ) -> None:
        super().__init__(num_groups, expression)
        for atom in group:
            self.addGroup([atom])
        if weighByMass:
            self.addGroup(group)
        else:
            self.addGroup(group, [1] * len(group))
        self.setUsesPeriodicBoundaryConditions(pbc)
