"""
.. module:: unit
    :platform: Linux, MacOS
    :synopsis: A module for unit conversion and serialization of openmm.unit objects

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import ast
import inspect
from functools import wraps

from openmm import unit


def in_md_units(val):
    """
    Return the numerical value of a quantity in the MD unit system (e.g. mass in Da, distance in nm,
    time in ps, temperature in K, energy in kJ/mol, angle in rad).
    """
    if isinstance(val, unit.Quantity):
        return val.value_in_unit_system(unit.md_unit_system)
    return val


def convert_quantities(func):
    """
    A decorator that converts all instances of openmm.unit.Quantity in a function's list of
    arguments to their numerical values in the MD unit system (e.g. mass in Da, distance in nm,
    time in ps, charge in e, temperature in K, angle in rad, energy in kJ/mol).
    """
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            bound.arguments[name] = in_md_units(value)
        return func(*bound.args, **bound.kwargs)

    return wrapper


class _NodeTransformer(ast.NodeTransformer):
    """
    A child class of ast.NodeTransformer that replaces all instances of ast.Name with
    an ast.Attribute with the value "unit" and the attribute name equal to the original
    id of the ast.Name.
    """

    def visit_Name(self, node: ast.Name) -> ast.Attribute:  # pylint: disable=invalid-name
        """
        Replace an instance of ast.Name with an ast.Attribute with the value "unit" and
        the attribute name equal to the original id of the ast.Name.
        """
        mod = ast.Name(id="unit", ctx=ast.Load())
        return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())


class SerializableUnit(unit.Unit):
    """
    A child class of openmm.unit.Unit that can be serialized and deserialized.
    """

    def __init__(self, base_or_scaled_units):
        if isinstance(base_or_scaled_units, unit.Unit):
            self.__dict__ = base_or_scaled_units.__dict__
        elif isinstance(base_or_scaled_units, str):
            tree = _NodeTransformer().visit(ast.parse(base_or_scaled_units, mode="eval"))
            self.__dict__ = eval(  # pylint: disable=eval-used
                compile(ast.fix_missing_locations(tree), "", mode="eval")
            ).__dict__
        else:
            super().__init__(base_or_scaled_units)

    def __getstate__(self):
        return {"description": self.__repr__()}

    def __setstate__(self, keywords) -> None:
        self.__init__(keywords["description"])
