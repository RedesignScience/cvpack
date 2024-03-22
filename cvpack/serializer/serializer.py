"""
.. module:: serializer
   :platform: Linux, MacOS, Windows
   :synopsis: Collective Variable Serialization

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import yaml
from openmm import app as mmapp


class Serializable(yaml.YAMLObject):
    """
    A mixin class that allows serialization and deserialization of objects with PyYAML.
    """

    @classmethod
    def registerTag(cls, tag: str) -> None:
        """
        Register a class for serialization and deserialization with PyYAML.

        Parameters
        ----------
        tag
            The YAML tag to be used for this class.
        """
        cls.yaml_tag = tag
        yaml.SafeDumper.add_representer(cls, cls.to_yaml)
        yaml.SafeLoader.add_constructor(tag, cls.from_yaml)


class SerializableAtom(Serializable):
    r"""
    A serializable version of OpenMM's Atom class.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self, atom: t.Union[mmapp.topology.Atom, "SerializableAtom"]
    ) -> None:
        self.name = atom.name
        self.index = atom.index
        if isinstance(atom, mmapp.topology.Atom):
            self.element = atom.element.symbol
            self.residue = atom.residue.index
        else:
            self.element = atom.element
            self.residue = atom.residue
        self.id = atom.id

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self.__dict__

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__dict__.update(keywords)


SerializableAtom.registerTag("!cvpack.Atom")


class SerializableResidue(Serializable):
    r"""A serializable version of OpenMM's Residue class."""

    def __init__(  # pylint: disable=super-init-not-called
        self, residue: t.Union[mmapp.topology.Residue, "SerializableResidue"]
    ) -> None:
        self.name = residue.name
        self.index = residue.index
        if isinstance(residue, mmapp.topology.Residue):
            self.chain = residue.chain.index
        else:
            self.chain = residue.chain
        self.id = residue.id
        self._atoms = list(map(SerializableAtom, residue.atoms()))

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return self.__dict__

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__dict__.update(keywords)

    def __len__(self) -> int:
        return len(self._atoms)

    def atoms(self):
        """Iterate over all Atoms in the Residue."""
        return iter(self._atoms)


SerializableResidue.registerTag("!cvpack.Residue")


def serialize(obj: t.Any, iostream: t.IO) -> None:
    """
    Serializes a cvpack object.

    Parameters
    ----------
    obj
        The cvpack object to be serialized
    iostream
        A text stream in write mode

    Example
    =======
    >>> import cvpack
    >>> import io
    >>> from cvpack import serializer
    >>> radius_of_gyration = cvpack.RadiusOfGyration([0, 1, 2])
    >>> iostream = io.StringIO()
    >>> serializer.serialize(radius_of_gyration, iostream)
    >>> print(iostream.getvalue())
    !cvpack.RadiusOfGyration
    group:
    - 0
    - 1
    - 2
    pbc: false
    weighByMass: false
    <BLANKLINE>
    """
    iostream.write(yaml.safe_dump(obj))


def deserialize(iostream: t.IO) -> t.Any:
    """
    Deserializes a cvpack object.

    Parameters
    ----------
    iostream
        A text stream in read mode containing the object to be deserialized

    Returns
    -------
    t.Any
        An instance of any cvpack class

    Example
    -------
    >>> import cvpack
    >>> import io
    >>> from cvpack import serializer
    >>> radius_of_gyration = cvpack.RadiusOfGyration([0, 1, 2])
    >>> iostream = io.StringIO()
    >>> serializer.serialize(radius_of_gyration, iostream)
    >>> iostream.seek(0)
    0
    >>> new_object = serializer.deserialize(iostream)
    >>> type(new_object)
    <class 'cvpack.radius_of_gyration.RadiusOfGyration'>
    """
    return yaml.safe_load(iostream.read())
