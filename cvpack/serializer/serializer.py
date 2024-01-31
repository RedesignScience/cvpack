"""
.. module:: serializer
   :platform: Linux, MacOS, Windows
   :synopsis: Collective Variable Serialization

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import yaml


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
        !!python/object:cvpack.radius_of_gyration.RadiusOfGyration
        group:
        - 0
        - 1
        - 2
        pbc: false
        weighByMass: false
        <BLANKLINE>

    """
    iostream.write(yaml.dump(obj, Dumper=yaml.CDumper))


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
        >>> print(type(new_object))
        <class 'cvpack.radius_of_gyration.RadiusOfGyration'>

    """
    return yaml.load(iostream.read(), Loader=yaml.CLoader)
