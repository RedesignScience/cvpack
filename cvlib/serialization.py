"""
.. module:: serialization
   :platform: Linux, MacOS, Windows
   :synopsis: Collective Variable Serialization

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import IO, Any

import yaml


def serialize(obj: Any, file: IO) -> None:
    """
    Serializes a cvlib object.

    """
    file.write(yaml.dump(obj, Dumper=yaml.CDumper))


def deserialize(file: IO) -> Any:
    """
    Deserializes a cvlib object.

    """
    return yaml.load(file.read(), Loader=yaml.CLoader)
