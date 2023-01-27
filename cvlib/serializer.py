"""
.. module:: serializer
   :platform: Linux, MacOS, Windows
   :synopsis: Collective Variable Serialization

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import IO, Any

import yaml


def serialize(obj: Any, iostream: IO) -> None:
    """
    Serializes a cvlib object.

    Parameters
    ----------
        obj
            The cvlib object to be serialized
        iostream
            A text stream in write mode

    Example
    =======
        >>> import cvlib
        >>> import io
        >>> from cvlib import serializer
        >>> radius_of_gyration = cvlib.RadiusOfGyration([0, 1, 2])
        >>> iostream = io.StringIO()
        >>> serializer.serialize(radius_of_gyration, iostream)
        >>> print(iostream.getvalue())
        !!python/object:cvlib.cvlib.RadiusOfGyration
        atoms:
        - 0
        - 1
        - 2
        <BLANKLINE>

    """
    iostream.write(yaml.dump(obj, Dumper=yaml.CDumper))


def deserialize(iostream: IO) -> Any:
    """
    Deserializes a cvlib object.

    Parameters
    ----------
        iostream
            A text stream in read mode containing the object to be deserialized

    Returns
    -------
        An instance of any cvlib class


    Example
    =======
        >>> import cvlib
        >>> import io
        >>> from cvlib import serializer
        >>> radius_of_gyration = cvlib.RadiusOfGyration([0, 1, 2])
        >>> iostream = io.StringIO()
        >>> serializer.serialize(radius_of_gyration, iostream)
        >>> iostream.seek(0)
        0
        >>> new_object = serializer.deserialize(iostream)
        >>> print(type(new_object))
        <class 'cvlib.cvlib.RadiusOfGyration'>

    """
    return yaml.load(iostream.read(), Loader=yaml.CLoader)
