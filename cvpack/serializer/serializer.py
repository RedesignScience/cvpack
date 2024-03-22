"""
.. module:: serializer
   :platform: Linux, MacOS, Windows
   :synopsis: Collective Variable Serialization

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import yaml


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
        An instance of any cvpack class


    Example
    =======
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
