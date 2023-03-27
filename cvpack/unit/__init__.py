"""
.. module:: unit
    :platform: Linux, MacOS
    :synopsis: Explicitly expose the contents of openmm.unit for type hinting purposes

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .unit import SerializableUnit, convert_quantities, in_md_units  # noqa: F401
