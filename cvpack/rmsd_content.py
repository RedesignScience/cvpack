"""
.. class:: RMSDContent
   :platform: Linux, MacOS, Windows
   :synopsis: Secondary-structure RMSD content of a sequence of residues

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from importlib import resources

import numpy as np
import openmm
from openmm import app as mmapp

from .cvpack import AbstractCollectiveVariable


class RMSDContent(openmm.CustomCVForce, AbstractCollectiveVariable):
    """
    Abstract class for secondary-structure RMSD content of a sequence of `n` residues.
    """

    @classmethod
    def load_positions(cls, filename: str) -> np.ndarray:
        return 0.1 * np.loadtxt(
            str(resources.files("cvpack").joinpath("data").joinpath(filename)),
            delimiter=",",
        )

    @staticmethod
    def getAtomList(residue: mmapp.topology.Residue) -> t.List[int]:
        residue_atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY":
            residue_atoms["CB"] = residue_atoms["HA2"]
        atom_list = []
        for atom in ("N", "CA", "CB", "C", "O"):
            try:
                atom_list.append(residue_atoms[atom])
            except KeyError as error:
                raise ValueError(
                    f"Atom {atom} not found in residue {residue.name}{residue.id}"
                ) from error
        return atom_list
