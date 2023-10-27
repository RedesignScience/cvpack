"""
.. class:: VolumeOverlap
   :platform: Linux, MacOS, Windows

   :synopsis: The volume overlap between two groups of atoms

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import itertools as it
import typing as t

import numpy as np
import openmm as omm
from openff.toolkit.topology import Topology
from openmm import app as mmapp
from rdkit.Chem import AllChem as allchem
from rdkit.Chem import rdchem, rdmolops

from cvpack import unit as mmunit

from .cvpack import AbstractCollectiveVariable


class Fragment(rdchem.Mol):
    """
    A fragment of a molecule, defined as a heavy atom together with the hydrogen atoms
    bonded to it, if any.

    Parameters
    ----------
    frag
        An RDKit mol representing the fragment. All atoms must return a positive
        value through the ``GetAtomMapNum()`` method.

    Raises
    ------
    ValueError
        If any of the fragment atoms have a non-positive atom map number.
    """

    def __init__(self, frag: rdchem.Mol) -> None:
        super().__init__(frag)
        if any(atom.GetAtomMapNum() < 1 for atom in self.GetAtoms()):
            raise ValueError("Fragment atoms must return AtomMapNum > 0")

    @property
    def atom_indices(self) -> t.List[int]:
        """The atom indices of the fragment."""
        return tuple(atom.GetAtomMapNum() - 1 for atom in self.GetAtoms())

    @property
    def num_atoms(self) -> int:
        """The number of atoms in the fragment."""
        return self.GetNumAtoms()

    @property
    def radius(self) -> float:
        """The effective radius of the fragment in nanometers."""
        return (allchem.ComputeMolVolume(self) * 3 / (4 * np.pi)) ** (1 / 3) / 10


class TupletForce(omm.CustomCentroidBondForce):
    """
    A force that calculates the volume overlap of fragment pairs, triplets,
    quadruplets, etc.

    Parameters
    ----------
    size
        The number of fragments in each tuplet.
    fragments
        A list of all the fragments from which the tuplets are formed.
    tuplets
        A list of lists of indices of the fragments in each tuplet.
    """

    def __init__(
        self,
        size: int,
        fragments: t.Iterable[Fragment],
        tuplets: t.Iterable[t.List[int]],
    ) -> None:
        indices = list(range(1, size + 1))
        sumalpha = "+".join(f"alpha{i}" for i in indices)
        exponent = (
            "-("
            + "+".join(
                f"alpha{i}*alpha{j}*distance(g{i}, g{j})^2"
                for i, j in it.combinations(indices, 2)
            )
            + f")/({sumalpha})"
        )
        super().__init__(size, f"({np.pi}/({sumalpha}))^(3/2)*exp({exponent}+{size})")
        for i in indices:
            self.addPerBondParameter(f"alpha{i}")
        for fragment in fragments:
            self.addGroup(fragment.atom_indices)
        kappa = np.pi / (1.5514 ** (2 / 3))
        alphas = [kappa / fragment.radius**2 for fragment in fragments]
        for tuplet in tuplets:
            self.addBond(tuplet, [alphas[index] for index in tuplet])


class VolumeOverlap(omm.CustomCVForce, AbstractCollectiveVariable):
    """A force that estimates the overlapping volume of two residues.

    Parameters
    ----------
    mol
        The probing mol.
    ref
        The reference mol.

    Examples
    --------
    >>> import cvpack
    >>> import mdtraj
    >>> import numpy as np
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> single_top = mdtraj.Topology.from_openmm(model.topology)
    >>> single_pos = model.positions / model.positions.unit
    >>> topology = single_top.join(single_top).to_openmm()
    >>> positions = np.vstack([single_pos, single_pos])
    >>> forcefield = openmm.app.ForceField('amber/ff14SB.xml')
    >>> system = forcefield.createSystem(topology)
    >>> chain1, chain2 = list(topology.chains())
    >>> force = cvpack.VolumeOverlap(
    ...     topology,
    ...     positions,
    ...     [atom.index for atom in chain1.atoms()],
    ...     [atom.index for atom in chain2.atoms()],
    ... )
    """

    def __init__(
        self,
        topology: Topology,
        positions: mmunit.MatrixQuantity,
        group1: t.Iterable[int],
        group2: t.Iterable[int],
        *,
        max_tuplet_size: int = 5,
    ) -> None:
        tuplet_sizes = list(range(2, max_tuplet_size + 1))
        energy_terms = [f"overlap{n}" for n in tuplet_sizes]
        signs = it.cycle(["+", "-"])
        super().__init__("".join(sum(zip(signs, energy_terms), ())))

        mols = [
            self._atomsTomol(topology, positions, group) for group in (group1, group2)
        ]
        mol_frags = [self._molToFragments(mol) for mol in mols]
        fragments = sum(mol_frags, [])
        indices = list(range(len(fragments)))
        tuplets = [[i] for i in indices[: len(mol_frags[0])]]
        for size in tuplet_sizes:
            tuplets = [
                sorted(tuplet + [i])
                for tuplet, i in it.product(tuplets, indices)
                if i not in tuplet
            ]
            self.addCollectiveVariable(
                f"overlap{size}", TupletForce(size, fragments, tuplets)
            )

    @staticmethod
    def _atomsTomol(
        topology: mmapp.Topology,
        positions: mmunit.MatrixQuantity,
        atomIndices: t.Iterable[int],
    ) -> rdchem.Mol:
        rdmol = rdchem.RWMol()
        indices = {itop: imol for imol, itop in enumerate(atomIndices)}
        index_set = set(indices)
        atoms = [atom for atom in topology.atoms() if atom.index in index_set]
        for atom in atoms:
            rdatom = rdchem.Atom(atom.element.atomic_number)
            rdatom.SetAtomMapNum(atom.index + 1)
            rdmol.AddAtom(rdatom)
        for bond in topology.bonds():
            i, j = bond.atom1.index, bond.atom2.index
            if set([i, j]).issubset(index_set):
                rdmol.AddBond(indices[i], indices[j])
        conformer = rdchem.Conformer(len(atoms))
        for i, atom in enumerate(atoms):
            conformer.SetAtomPosition(i, positions[atom.index] * 10)
        rdmol.AddConformer(conformer)
        return rdchem.Mol(rdmol)

    @staticmethod
    def _molToFragments(mol: rdchem.Mol) -> t.List[Fragment]:
        bonds_to_break = [
            bond.GetIdx()
            for bond in mol.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() > 1
            and bond.GetEndAtom().GetAtomicNum() > 1
        ]
        fragments = rdmolops.GetMolFrags(
            rdmolops.FragmentOnBonds(mol, bonds_to_break, addDummies=False), asMols=True
        )
        return [Fragment(frag) for frag in fragments]
