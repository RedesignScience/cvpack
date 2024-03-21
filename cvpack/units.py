"""
.. module:: units
   :platform: Linux, MacOS, Windows
   :synopsis: Units of measurement for CVPack.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import ast
import functools
import inspect
import typing as t
from numbers import Real

import numpy as np
import openmm
from openmm import unit as mmunit

from .serializer import Serializable

ScalarQuantity = t.Union[mmunit.Quantity, Real]
VectorQuantity = t.Union[mmunit.Quantity, np.ndarray, openmm.Vec3]
MatrixQuantity = t.Union[
    mmunit.Quantity, np.ndarray, t.Sequence[openmm.Vec3], t.Sequence[np.ndarray]
]


class Unit(mmunit.Unit, Serializable, ast.NodeTransformer):
    """
    Extension of the OpenMM Unit class to allow serialization and deserialization.

    Parameters
    ----------
    data
        The data to be used to create the unit.
    """

    def __init__(self, data: t.Union[str, mmunit.Unit, dict]) -> None:
        if isinstance(data, str):
            expression = self.visit(ast.parse(data, mode="eval"))
            code = compile(ast.fix_missing_locations(expression), "", mode="eval")
            data = eval(code)  # pylint: disable=eval-used
        if isinstance(data, mmunit.Unit):
            data = dict(data.iter_base_or_scaled_units())
        super().__init__(data)

    def __repr__(self) -> str:
        return self.get_symbol()

    def __getstate__(self) -> t.Dict[str, str]:
        return {"data": str(self)}

    def __setstate__(self, keywords: t.Dict[str, str]) -> None:
        self.__init__(keywords["data"])

    def visit_Name(  # pylint: disable=invalid-name
        self, node: ast.Name
    ) -> ast.Attribute:
        """
        Visit a Name node and transform it into an Attribute node.

        Parameters
        ----------
        node
            The node to be visited and transformed.
        """
        return ast.Attribute(
            value=ast.Name(id="mmunit", ctx=ast.Load()), attr=node.id, ctx=ast.Load()
        )


Unit.registerTag("!cvpack.Unit")


class Quantity(mmunit.Quantity, Serializable):
    """
    Extension of the OpenMM Quantity class to allow serialization and deserialization.
    """

    def __init__(self, *args: t.Any) -> None:
        if len(args) == 1 and mmunit.is_quantity(args[0]):
            super().__init__(args[0].value_in_unit(args[0].unit), Unit(args[0].unit))
        else:
            super().__init__(*args)

    def __repr__(self):
        return str(self)

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {"value": self.value, "unit": str(self.unit)}

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(keywords["value"], Unit(keywords["unit"]))

    @property
    def value(self) -> t.Any:
        """The value of the quantity."""
        return self._value

    def value_in_md_units(self) -> t.Any:  # pylint: disable=invalid-name
        """The value of the quantity in MD units."""
        return self.value_in_unit_system(mmunit.md_unit_system)


Quantity.registerTag("!cvpack.Quantity")


def preprocess_units(func: t.Callable) -> t.Callable:
    """
    A decorator that converts instances of openmm.unit.Unit and openmm.unit.Quantity
    into openxps.units.Unit and openxps.units.Quantity, respectively.

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Example
    -------
    >>> from cvpack import units
    >>> from openmm import unit as mmunit
    >>> @units.preprocess_units
    ... def function(data):
    ...     return data
    >>> assert isinstance(function(mmunit.angstrom), units.Unit)
    >>> assert isinstance(function(5 * mmunit.angstrom), units.Quantity)
    >>> seq = [mmunit.angstrom, mmunit.nanometer]
    >>> assert isinstance(function(seq), list)
    >>> assert all(isinstance(item, units.Unit) for item in function(seq))
    >>> dct = {"length": 3 * mmunit.angstrom, "time": 2 * mmunit.picosecond}
    >>> assert isinstance(function(dct), dict)
    >>> assert all(isinstance(item, units.Quantity) for item in function(dct).values())
    """
    signature = inspect.signature(func)

    def convert(data: t.Any) -> t.Any:
        if isinstance(data, mmunit.Quantity):
            return Quantity(data)
        if isinstance(data, mmunit.Unit):
            return Unit(data)
        if isinstance(data, str):
            return data
        if isinstance(data, t.Sequence):
            return type(data)(map(convert, data))
        if isinstance(data, t.Dict):
            return type(data)((key, convert(value)) for key, value in data.items())
        return data

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for name, data in bound.arguments.items():
            bound.arguments[name] = convert(data)
        return func(*bound.args, **bound.kwargs)

    return wrapper


def value_in_md_units(  # pylint: disable=redefined-outer-name
    quantity: t.Union[ScalarQuantity, VectorQuantity, MatrixQuantity]
) -> t.Union[float, np.ndarray, openmm.Vec3, t.List[openmm.Vec3], t.List[np.ndarray]]:
    """
    Return the numerical value of a quantity in the MD unit system (e.g. mass in Da,
    distance in nm, time in ps, temperature in K, energy in kJ/mol, angle in rad).

    Parameters
    ----------
        quantity
            The quantity to be converted.

    Returns
    -------
        The numerical value of the quantity in the MD unit system.

    Raises
    ------
        TypeError
            If the quantity cannot be converted to the MD unit system.

    Examples
    --------
        >>> from cvpack.units import value_in_md_units
        >>> from openmm import Vec3
        >>> from openmm.unit import angstrom, femtosecond, degree
        >>> from numpy import array
        >>> value_in_md_units(1.0)
        1.0
        >>> value_in_md_units(1.0*femtosecond)
        0.001
        >>> value_in_md_units(1.0*degree)
        0.017453292519943295
        >>> value_in_md_units(array([1, 2, 3])*angstrom)
        array([0.1, 0.2, 0.3])
        >>> value_in_md_units([Vec3(1, 2, 4), Vec3(5, 8, 9)]*angstrom)
        [Vec3(x=0.1, y=0.2, z=0.4), Vec3(x=0.5, y=0.8, z=0.9)]
        >>> try:
        ...     value_in_md_units([1, 2, 3]*angstrom)
        ... except TypeError as error:
        ...     print(error)
        Cannot convert [1, 2, 3] A to MD units
    """
    if isinstance(quantity, mmunit.Quantity):
        value = quantity.value_in_unit_system(mmunit.md_unit_system)
    else:
        value = quantity
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, t.Sequence):
        if isinstance(value[0], openmm.Vec3):
            return [openmm.Vec3(*vec) for vec in value]
        if isinstance(value[0], np.ndarray):
            return [np.array(vec) for vec in value]
    if isinstance(value, (np.ndarray, openmm.Vec3)):
        return value
    raise TypeError(f"Cannot convert {quantity} to MD units")


def convert_quantities(func):
    """
    A decorator that converts all instances of openmm.unit.Quantity in a function's list
    of arguments to their numerical values in the MD unit system (e.g. mass in Da,
    distance in nm, time in ps, charge in e, temperature in K, angle in rad, energy in
    kJ/mol).

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Examples
    --------
        >>> from openmm import unit
        >>> from openmm.unit import femtosecond, degree
        >>> from numpy import array
        >>> @convert_quantities
        ... def func(a, b, c):
        ...     return a, b, c
        >>> func(1.0, 1.0*femtosecond, 1.0*degree)
        (1.0, 0.001, 0.017453292519943295)
    """
    sig = inspect.signature(func)

    def remove_unit(value):
        if isinstance(value, mmunit.Quantity):
            return value.value_in_unit_system(mmunit.md_unit_system)
        if isinstance(value, (np.ndarray, openmm.Vec3)):
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(map(remove_unit, value))
        if isinstance(value, dict):
            return {key: remove_unit(val) for key, val in value.items()}
        return value

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            bound.arguments[name] = remove_unit(value)
        return func(*bound.args, **bound.kwargs)

    return wrapper
