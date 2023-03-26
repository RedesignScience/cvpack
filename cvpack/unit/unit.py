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

from openmm import unit as _unit


def in_md_units(quantity):
    """
    Return the numerical value of a quantity in the MD unit system (e.g. mass in Da, distance in nm,
    time in ps, temperature in K, energy in kJ/mol, angle in rad).
    """
    if isinstance(quantity, _unit.Quantity):
        return quantity.value_in_unit_system(_unit.md_unit_system)
    return quantity


def with_values_in_md_units(func):
    """
    A decorator that converts all instances of openmm.unit.Quantity in a function's arguments
    to their numerical values in the MD unit system (e.g. mass in Da, distance in nm, time in ps,
    temperature in K, energy in kJ/mol, angle in rad).
    """
    # create a new signature with converted Quantity defaults and update the function's signature
    sig = inspect.signature(func)
    func.__signature__ = sig.replace(
        parameters=[
            parameter.replace(default=in_md_units(parameter.default))
            for parameter in sig.parameters.values()
        ]
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        converted_args = map(in_md_units, args)
        converted_kwargs = dict(zip(kwargs.keys(), map(in_md_units, kwargs.values())))
        return func(*converted_args, **converted_kwargs)

    return wrapper


class _NodeTransformer(ast.NodeTransformer):
    """
    A child class of ast.NodeTransformer that replaces all instances of ast.Name with
    an ast.Attribute with the value "mmunit" and the attribute name equal to the original
    id of the ast.Name.
    """

    def visit_Name(self, node: ast.Name) -> ast.Attribute:  # pylint: disable=invalid-name
        """
        Replace an instance of ast.Name with an ast.Attribute with the value "mmunit" and
        the attribute name equal to the original id of the ast.Name.
        """
        mod = ast.Name(id="_unit", ctx=ast.Load())
        return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())


class Unit(_unit.Unit):
    """
    A child class of openmm.unit.Unit that allows for serialization and deserialization.
    """

    def __getstate__(self):
        return {"description": str(self)}

    def __setstate__(self, keywords) -> None:
        tree = _NodeTransformer().visit(ast.parse(keywords["description"], mode="eval"))
        self.__dict__ = eval(  # pylint: disable=eval-used
            compile(ast.fix_missing_locations(tree), "", mode="eval")
        ).__dict__


class Quantity(_unit.Quantity):
    """
    A child class of openmm.unit.Quantity that allows for serialization and deserialization.
    """

    def __getstate__(self):
        return {"value": self._value, "unit": self._unit}

    def __setstate__(self, keywords):
        self.__init__(keywords["value"], keywords["unit"])
