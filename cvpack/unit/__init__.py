"""
.. module:: unit
   :platform: Linux, MacOS, Windows
   :synopsis: This module explicitly exports OpenMM's unit module classes and
                constants for type annotation purposes.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from __future__ import annotations

import ast
import inspect
import typing as t
from functools import wraps
from numbers import Real

import numpy as np
import openmm
from openmm import unit as _mmunit

ScalarQuantity = t.Union[_mmunit.Quantity, Real]
VectorQuantity = t.Union[_mmunit.Quantity, np.ndarray, openmm.Vec3]
MatrixQuantity = t.Union[
    _mmunit.Quantity, np.ndarray, t.Sequence[openmm.Vec3], t.Sequence[np.ndarray]
]


class _NodeTransformer(ast.NodeTransformer):
    r"""
    A child class of ast.NodeTransformer that replaces all instances of ast.Name with
    an ast.Attribute with the value "unit" and the attribute name equal to the original
    id of the ast.Name.
    """

    def visit_Name(  # pylint: disable=invalid-name
        self, node: ast.Name
    ) -> ast.Attribute:
        """
        Replace an instance of ast.Name with an ast.Attribute with the value "unit" and
        the attribute name equal to the original id of the ast.Name.
        """
        mod = ast.Name(id="_mmunit", ctx=ast.Load())
        return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())


class SerializableUnit(_mmunit.Unit):
    r"""
    A child class of openmm.unit.Unit that allows for serialization/deserialization.

    Parameters
    ----------
        base_or_scaled_units
            The base or scaled units.

    Examples
    --------
        >>> import yaml
        >>> from cvpack.unit import SerializableUnit
        >>> from openmm.unit import nanometer, picosecond
        >>> SerializableUnit(nanometer)  # doctest: +ELLIPSIS
        Unit(..., name="nanometer", symbol="nm"): 1.0})
        >>> dump = yaml.dump(SerializableUnit(nanometer/picosecond))
        >>> print(dump)
        !!python/object:cvpack.unit.SerializableUnit
        description: nanometer/picosecond
        version: 1
        <BLANKLINE>
        >>> 2*yaml.load(dump, Loader=yaml.CLoader)
        Quantity(value=2, unit=nanometer/picosecond)
    """

    def __init__(self, base_or_scaled_units):
        if isinstance(base_or_scaled_units, _mmunit.Unit):
            self.__dict__ = base_or_scaled_units.__dict__
        elif isinstance(base_or_scaled_units, str):
            tree = _NodeTransformer().visit(
                ast.parse(base_or_scaled_units, mode="eval")
            )
            self.__dict__ = eval(  # pylint: disable=eval-used
                compile(ast.fix_missing_locations(tree), "", mode="eval")
            ).__dict__
        else:
            super().__init__(base_or_scaled_units)

    def __getstate__(self):
        return {"version": 1, "description": str(self)}

    def __setstate__(self, kwds) -> None:
        if kwds["version"] != 1:
            raise ValueError("Unknown version")
        self.__init__(kwds["description"])


class SerializableQuantity(_mmunit.Quantity):
    r"""
    A child class of openmm.unit.Quantity that allows for serialization/deserialization.

    Parameters
    ----------
        value
            Another quantity or the numerical value of a quantity.
        unit
            A unit of measurement or None if value is a quantity.

    Examples
    --------
        >>> import yaml
        >>> from cvpack.unit import SerializableQuantity
        >>> from openmm.unit import nanometer
        >>> SerializableQuantity(1.0, nanometer)
        Quantity(value=1.0, unit=nanometer)
        >>> dump = yaml.dump(SerializableQuantity(1.0, nanometer))
        >>> print(dump)
        !!python/object:cvpack.unit.SerializableQuantity
        unit: !!python/object:cvpack.unit.SerializableUnit
          description: nanometer
          version: 1
        value: 1.0
        version: 1
        <BLANKLINE>
        >>> yaml.load(dump, Loader=yaml.CLoader)
        Quantity(value=1.0, unit=nanometer)
    """

    def __init__(self, value, unit=None):  # pylint: disable=redefined-outer-name
        if unit is None:
            super().__init__(value._value, SerializableUnit(value.unit))
        else:
            super().__init__(value, SerializableUnit(unit))

    def __getstate__(self):
        return {"version": 1, "value": self._value, "unit": self.unit}

    def __setstate__(self, kwds) -> None:
        if kwds["version"] != 1:
            raise ValueError("Unknown version")
        self.__init__(kwds["value"], kwds["unit"])

    def value_in_md_units(self):  # pylint: disable=invalid-name
        """
        Return the numerical value of the quantity in the MD unit system (e.g. mass in
        Da, distance in nm, time in ps, temperature in K, energy in kJ/mol, angle in
        rad).
        """
        return value_in_md_units(self)


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
        >>> from cvpack.unit import value_in_md_units
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
    if isinstance(quantity, _mmunit.Quantity):
        value = quantity.value_in_unit_system(_mmunit.md_unit_system)
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
        >>> from cvpack import unit
        >>> from openmm.unit import femtosecond, degree
        >>> from numpy import array
        >>> @unit.convert_quantities
        ... def func(a, b, c):
        ...     return a, b, c
        >>> func(1.0, 1.0*femtosecond, 1.0*degree)
        (1.0, 0.001, 0.017453292519943295)
    """
    sig = inspect.signature(func)

    def remove_unit(value):
        if isinstance(value, _mmunit.Quantity):
            return value.value_in_unit_system(_mmunit.md_unit_system)
        if isinstance(value, (np.ndarray, openmm.Vec3)):
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(map(remove_unit, value))
        if isinstance(value, dict):
            return {key: remove_unit(val) for key, val in value.items()}
        return value

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            bound.arguments[name] = remove_unit(value)
        return func(*bound.args, **bound.kwargs)

    return wrapper


Unit = _mmunit.Unit
Quantity = _mmunit.Quantity

AVOGADRO_CONSTANT_NA = _mmunit.AVOGADRO_CONSTANT_NA
BOLTZMANN_CONSTANT_kB = _mmunit.BOLTZMANN_CONSTANT_kB
BaseDimension = _mmunit.BaseDimension
BaseUnit = _mmunit.BaseUnit
GRAVITATIONAL_CONSTANT_G = _mmunit.GRAVITATIONAL_CONSTANT_G
MOLAR_GAS_CONSTANT_R = _mmunit.MOLAR_GAS_CONSTANT_R
SPEED_OF_LIGHT_C = _mmunit.SPEED_OF_LIGHT_C
ScaledUnit = _mmunit.ScaledUnit
SiPrefix = _mmunit.SiPrefix
UnitSystem = _mmunit.UnitSystem
absolute_import = _mmunit.absolute_import
acos = _mmunit.acos
acosh = _mmunit.acosh
amount_dimension = _mmunit.amount_dimension
amp = _mmunit.amp
ampere = _mmunit.ampere
amperes = _mmunit.amperes
amps = _mmunit.amps
amu = _mmunit.amu
amus = _mmunit.amus
angle_dimension = _mmunit.angle_dimension
angstrom = _mmunit.angstrom
angstroms = _mmunit.angstroms
arcminute = _mmunit.arcminute
arcminutes = _mmunit.arcminutes
arcsecond = _mmunit.arcsecond
arcseconds = _mmunit.arcseconds
asin = _mmunit.asin
asinh = _mmunit.asinh
atan = _mmunit.atan
atan2 = _mmunit.atan2
atanh = _mmunit.atanh
atmosphere = _mmunit.atmosphere
atmospheres = _mmunit.atmospheres
atom_mass_units = _mmunit.atom_mass_units
atomic_mass_unit = _mmunit.atomic_mass_unit
# atto = _mmunit.atto
# attocalorie = _mmunit.attocalorie
# attocalories = _mmunit.attocalories
# attogram = _mmunit.attogram
# attograms = _mmunit.attograms
# attojoule = _mmunit.attojoule
# attojoules = _mmunit.attojoules
# attoliter = _mmunit.attoliter
# attoliters = _mmunit.attoliters
# attometer = _mmunit.attometer
# attometers = _mmunit.attometers
# attomolar = _mmunit.attomolar
# attomolars = _mmunit.attomolars
# attonewton = _mmunit.attonewton
# attonewtons = _mmunit.attonewtons
# attopascal = _mmunit.attopascal
# attopascals = _mmunit.attopascals
# attosecond = _mmunit.attosecond
# attoseconds = _mmunit.attoseconds
ban = _mmunit.ban
bans = _mmunit.bans
bar = _mmunit.bar  # pylint: disable=disallowed-name
bars = _mmunit.bars
basedimension = _mmunit.basedimension
baseunit = _mmunit.baseunit
binary_prefixes = _mmunit.binary_prefixes
bit = _mmunit.bit
bits = _mmunit.bits
bohr = _mmunit.bohr
bohrs = _mmunit.bohrs
byte = _mmunit.byte
bytes = _mmunit.bytes  # pylint: disable=redefined-builtin
calorie = _mmunit.calorie
calories = _mmunit.calories
candela = _mmunit.candela
candelas = _mmunit.candelas
centi = _mmunit.centi
centicalorie = _mmunit.centicalorie
centicalories = _mmunit.centicalories
centigram = _mmunit.centigram
centigrams = _mmunit.centigrams
centijoule = _mmunit.centijoule
centijoules = _mmunit.centijoules
centiliter = _mmunit.centiliter
centiliters = _mmunit.centiliters
centimeter = _mmunit.centimeter
centimeters = _mmunit.centimeters
centimolar = _mmunit.centimolar
centimolars = _mmunit.centimolars
centinewton = _mmunit.centinewton
centinewtons = _mmunit.centinewtons
centipascal = _mmunit.centipascal
centipascals = _mmunit.centipascals
centisecond = _mmunit.centisecond
centiseconds = _mmunit.centiseconds
centuries = _mmunit.centuries
century = _mmunit.century
centurys = _mmunit.centurys
cgs_unit_system = _mmunit.cgs_unit_system
charge_dimension = _mmunit.charge_dimension
constants = _mmunit.constants
cos = _mmunit.cos
cosh = _mmunit.cosh
coulomb = _mmunit.coulomb
coulombs = _mmunit.coulombs
dalton = _mmunit.dalton
daltons = _mmunit.daltons
day = _mmunit.day
days = _mmunit.days
debye = _mmunit.debye
debyes = _mmunit.debyes
# deca = _mmunit.deca
# decacalorie = _mmunit.decacalorie
# decacalories = _mmunit.decacalories
# decagram = _mmunit.decagram
# decagrams = _mmunit.decagrams
# decajoule = _mmunit.decajoule
# decajoules = _mmunit.decajoules
# decaliter = _mmunit.decaliter
# decaliters = _mmunit.decaliters
# decameter = _mmunit.decameter
# decameters = _mmunit.decameters
# decamolar = _mmunit.decamolar
# decamolars = _mmunit.decamolars
# decanewton = _mmunit.decanewton
# decanewtons = _mmunit.decanewtons
# decapascal = _mmunit.decapascal
# decapascals = _mmunit.decapascals
# decasecond = _mmunit.decasecond
# decaseconds = _mmunit.decaseconds
# deci = _mmunit.deci
# decicalorie = _mmunit.decicalorie
# decicalories = _mmunit.decicalories
# decigram = _mmunit.decigram
# decigrams = _mmunit.decigrams
# decijoule = _mmunit.decijoule
# decijoules = _mmunit.decijoules
# deciliter = _mmunit.deciliter
# deciliters = _mmunit.deciliters
# decimeter = _mmunit.decimeter
# decimeters = _mmunit.decimeters
# decimolar = _mmunit.decimolar
# decimolars = _mmunit.decimolars
# decinewton = _mmunit.decinewton
# decinewtons = _mmunit.decinewtons
# decipascal = _mmunit.decipascal
# decipascals = _mmunit.decipascals
# decisecond = _mmunit.decisecond
# deciseconds = _mmunit.deciseconds
define_prefixed_units = _mmunit.define_prefixed_units
degree = _mmunit.degree
degrees = _mmunit.degrees
# deka = _mmunit.deka
# dekacalorie = _mmunit.dekacalorie
# dekacalories = _mmunit.dekacalories
# dekagram = _mmunit.dekagram
# dekagrams = _mmunit.dekagrams
# dekajoule = _mmunit.dekajoule
# dekajoules = _mmunit.dekajoules
# dekaliter = _mmunit.dekaliter
# dekaliters = _mmunit.dekaliters
# dekameter = _mmunit.dekameter
# dekameters = _mmunit.dekameters
# dekamolar = _mmunit.dekamolar
# dekamolars = _mmunit.dekamolars
# dekanewton = _mmunit.dekanewton
# dekanewtons = _mmunit.dekanewtons
# dekapascal = _mmunit.dekapascal
# dekapascals = _mmunit.dekapascals
# dekasecond = _mmunit.dekasecond
# dekaseconds = _mmunit.dekaseconds
dimensionless = _mmunit.dimensionless
dit = _mmunit.dit
dits = _mmunit.dits
division = _mmunit.division
dot = _mmunit.dot
dyne = _mmunit.dyne
dynes = _mmunit.dynes
elementary_charge = _mmunit.elementary_charge
elementary_charges = _mmunit.elementary_charges
erg = _mmunit.erg
ergs = _mmunit.ergs
# exa = _mmunit.exa
# exacalorie = _mmunit.exacalorie
# exacalories = _mmunit.exacalories
# exagram = _mmunit.exagram
# exagrams = _mmunit.exagrams
# exajoule = _mmunit.exajoule
# exajoules = _mmunit.exajoules
# exaliter = _mmunit.exaliter
# exaliters = _mmunit.exaliters
# exameter = _mmunit.exameter
# exameters = _mmunit.exameters
# examolar = _mmunit.examolar
# examolars = _mmunit.examolars
# exanewton = _mmunit.exanewton
# exanewtons = _mmunit.exanewtons
# exapascal = _mmunit.exapascal
# exapascals = _mmunit.exapascals
# exasecond = _mmunit.exasecond
# exaseconds = _mmunit.exaseconds
exbi = _mmunit.exbi
farad = _mmunit.farad
farads = _mmunit.farads
feet = _mmunit.feet
femto = _mmunit.femto
femtocalorie = _mmunit.femtocalorie
femtocalories = _mmunit.femtocalories
femtogram = _mmunit.femtogram
femtograms = _mmunit.femtograms
femtojoule = _mmunit.femtojoule
femtojoules = _mmunit.femtojoules
femtoliter = _mmunit.femtoliter
femtoliters = _mmunit.femtoliters
femtometer = _mmunit.femtometer
femtometers = _mmunit.femtometers
femtomolar = _mmunit.femtomolar
femtomolars = _mmunit.femtomolars
femtonewton = _mmunit.femtonewton
femtonewtons = _mmunit.femtonewtons
femtopascal = _mmunit.femtopascal
femtopascals = _mmunit.femtopascals
femtosecond = _mmunit.femtosecond
femtoseconds = _mmunit.femtoseconds
foot = _mmunit.foot
fortnight = _mmunit.fortnight
fortnights = _mmunit.fortnights
furlong = _mmunit.furlong
furlongs = _mmunit.furlongs
gauss = _mmunit.gauss
gibi = _mmunit.gibi
giga = _mmunit.giga
gigacalorie = _mmunit.gigacalorie
gigacalories = _mmunit.gigacalories
gigagram = _mmunit.gigagram
gigagrams = _mmunit.gigagrams
gigajoule = _mmunit.gigajoule
gigajoules = _mmunit.gigajoules
gigaliter = _mmunit.gigaliter
gigaliters = _mmunit.gigaliters
gigameter = _mmunit.gigameter
gigameters = _mmunit.gigameters
gigamolar = _mmunit.gigamolar
gigamolars = _mmunit.gigamolars
giganewton = _mmunit.giganewton
giganewtons = _mmunit.giganewtons
gigapascal = _mmunit.gigapascal
gigapascals = _mmunit.gigapascals
gigasecond = _mmunit.gigasecond
gigaseconds = _mmunit.gigaseconds
gram = _mmunit.gram
grams = _mmunit.grams
hartley = _mmunit.hartley
hartleys = _mmunit.hartleys
hartree = _mmunit.hartree
hartrees = _mmunit.hartrees
# hecto = _mmunit.hecto
# hectocalorie = _mmunit.hectocalorie
# hectocalories = _mmunit.hectocalories
# hectogram = _mmunit.hectogram
# hectograms = _mmunit.hectograms
# hectojoule = _mmunit.hectojoule
# hectojoules = _mmunit.hectojoules
# hectoliter = _mmunit.hectoliter
# hectoliters = _mmunit.hectoliters
# hectometer = _mmunit.hectometer
# hectometers = _mmunit.hectometers
# hectomolar = _mmunit.hectomolar
# hectomolars = _mmunit.hectomolars
# hectonewton = _mmunit.hectonewton
# hectonewtons = _mmunit.hectonewtons
# hectopascal = _mmunit.hectopascal
# hectopascals = _mmunit.hectopascals
# hectosecond = _mmunit.hectosecond
# hectoseconds = _mmunit.hectoseconds
henries = _mmunit.henries
henry = _mmunit.henry
henrys = _mmunit.henrys
hour = _mmunit.hour
hours = _mmunit.hours
inch = _mmunit.inch
inches = _mmunit.inches
information_dimension = _mmunit.information_dimension
is_quantity = _mmunit.is_quantity
is_unit = _mmunit.is_unit
item = _mmunit.item
items = _mmunit.items
joule = _mmunit.joule
joules = _mmunit.joules
kelvin = _mmunit.kelvin
kelvins = _mmunit.kelvins
kibi = _mmunit.kibi
kilo = _mmunit.kilo
kilocalorie = _mmunit.kilocalorie
kilocalorie_per_mole = _mmunit.kilocalorie_per_mole
kilocalories = _mmunit.kilocalories
kilocalories_per_mole = _mmunit.kilocalories_per_mole
kilogram = _mmunit.kilogram
kilograms = _mmunit.kilograms
kilojoule = _mmunit.kilojoule
kilojoule_per_mole = _mmunit.kilojoule_per_mole
kilojoules = _mmunit.kilojoules
kilojoules_per_mole = _mmunit.kilojoules_per_mole
kiloliter = _mmunit.kiloliter
kiloliters = _mmunit.kiloliters
kilometer = _mmunit.kilometer
kilometers = _mmunit.kilometers
kilomolar = _mmunit.kilomolar
kilomolars = _mmunit.kilomolars
kilonewton = _mmunit.kilonewton
kilonewtons = _mmunit.kilonewtons
kilopascal = _mmunit.kilopascal
kilopascals = _mmunit.kilopascals
kilosecond = _mmunit.kilosecond
kiloseconds = _mmunit.kiloseconds
length_dimension = _mmunit.length_dimension
liter = _mmunit.liter
liters = _mmunit.liters
litre = _mmunit.litre
litres = _mmunit.litres
luminous_intensity_dimension = _mmunit.luminous_intensity_dimension
mass_dimension = _mmunit.mass_dimension
math = _mmunit.math
md_kilocalorie = _mmunit.md_kilocalorie
md_kilocalories = _mmunit.md_kilocalories
md_kilojoule = _mmunit.md_kilojoule
md_kilojoule_raw = _mmunit.md_kilojoule_raw
md_kilojoules = _mmunit.md_kilojoules
md_unit_system = _mmunit.md_unit_system
mebi = _mmunit.mebi
mega = _mmunit.mega
megacalorie = _mmunit.megacalorie
megacalories = _mmunit.megacalories
megagram = _mmunit.megagram
megagrams = _mmunit.megagrams
megajoule = _mmunit.megajoule
megajoules = _mmunit.megajoules
megaliter = _mmunit.megaliter
megaliters = _mmunit.megaliters
megameter = _mmunit.megameter
megameters = _mmunit.megameters
megamolar = _mmunit.megamolar
megamolars = _mmunit.megamolars
meganewton = _mmunit.meganewton
meganewtons = _mmunit.meganewtons
megapascal = _mmunit.megapascal
megapascals = _mmunit.megapascals
megasecond = _mmunit.megasecond
megaseconds = _mmunit.megaseconds
meter = _mmunit.meter
meters = _mmunit.meters
micro = _mmunit.micro
microcalorie = _mmunit.microcalorie
microcalories = _mmunit.microcalories
microgram = _mmunit.microgram
micrograms = _mmunit.micrograms
microjoule = _mmunit.microjoule
microjoules = _mmunit.microjoules
microliter = _mmunit.microliter
microliters = _mmunit.microliters
micrometer = _mmunit.micrometer
micrometers = _mmunit.micrometers
micromolar = _mmunit.micromolar
micromolars = _mmunit.micromolars
micronewton = _mmunit.micronewton
micronewtons = _mmunit.micronewtons
micropascal = _mmunit.micropascal
micropascals = _mmunit.micropascals
microsecond = _mmunit.microsecond
microseconds = _mmunit.microseconds
mile = _mmunit.mile
miles = _mmunit.miles
millenia = _mmunit.millenia
millenium = _mmunit.millenium
milleniums = _mmunit.milleniums
milli = _mmunit.milli
millicalorie = _mmunit.millicalorie
millicalories = _mmunit.millicalories
milligram = _mmunit.milligram
milligrams = _mmunit.milligrams
millijoule = _mmunit.millijoule
millijoules = _mmunit.millijoules
milliliter = _mmunit.milliliter
milliliters = _mmunit.milliliters
millimeter = _mmunit.millimeter
millimeters = _mmunit.millimeters
millimolar = _mmunit.millimolar
millimolars = _mmunit.millimolars
millinewton = _mmunit.millinewton
millinewtons = _mmunit.millinewtons
millipascal = _mmunit.millipascal
millipascals = _mmunit.millipascals
millisecond = _mmunit.millisecond
milliseconds = _mmunit.milliseconds
minute = _mmunit.minute
minutes = _mmunit.minutes
mmHg = _mmunit.mmHg
molal = _mmunit.molal
molar = _mmunit.molar
mole = _mmunit.mole
moles = _mmunit.moles
mymatrix = _mmunit.mymatrix
nano = _mmunit.nano
nanocalorie = _mmunit.nanocalorie
nanocalories = _mmunit.nanocalories
nanogram = _mmunit.nanogram
nanograms = _mmunit.nanograms
nanojoule = _mmunit.nanojoule
nanojoules = _mmunit.nanojoules
nanoliter = _mmunit.nanoliter
nanoliters = _mmunit.nanoliters
nanometer = _mmunit.nanometer
nanometers = _mmunit.nanometers
nanomolar = _mmunit.nanomolar
nanomolars = _mmunit.nanomolars
nanonewton = _mmunit.nanonewton
nanonewtons = _mmunit.nanonewtons
nanopascal = _mmunit.nanopascal
nanopascals = _mmunit.nanopascals
nanosecond = _mmunit.nanosecond
nanoseconds = _mmunit.nanoseconds
nat = _mmunit.nat
nats = _mmunit.nats
nepit = _mmunit.nepit
nepits = _mmunit.nepits
newton = _mmunit.newton
newtons = _mmunit.newtons
nit = _mmunit.nit
nits = _mmunit.nits
norm = _mmunit.norm
ohm = _mmunit.ohm
ohms = _mmunit.ohms
pascal = _mmunit.pascal
pascals = _mmunit.pascals
pebi = _mmunit.pebi
# peta = _mmunit.peta
# petacalorie = _mmunit.petacalorie
# petacalories = _mmunit.petacalories
# petagram = _mmunit.petagram
# petagrams = _mmunit.petagrams
# petajoule = _mmunit.petajoule
# petajoules = _mmunit.petajoules
# petaliter = _mmunit.petaliter
# petaliters = _mmunit.petaliters
# petameter = _mmunit.petameter
# petameters = _mmunit.petameters
# petamolar = _mmunit.petamolar
# petamolars = _mmunit.petamolars
# petanewton = _mmunit.petanewton
# petanewtons = _mmunit.petanewtons
# petapascal = _mmunit.petapascal
# petapascals = _mmunit.petapascals
# petasecond = _mmunit.petasecond
# petaseconds = _mmunit.petaseconds
pico = _mmunit.pico
picocalorie = _mmunit.picocalorie
picocalories = _mmunit.picocalories
picogram = _mmunit.picogram
picograms = _mmunit.picograms
picojoule = _mmunit.picojoule
picojoules = _mmunit.picojoules
picoliter = _mmunit.picoliter
picoliters = _mmunit.picoliters
picometer = _mmunit.picometer
picometers = _mmunit.picometers
picomolar = _mmunit.picomolar
picomolars = _mmunit.picomolars
piconewton = _mmunit.piconewton
piconewtons = _mmunit.piconewtons
picopascal = _mmunit.picopascal
picopascals = _mmunit.picopascals
picosecond = _mmunit.picosecond
picoseconds = _mmunit.picoseconds
planck_unit_system = _mmunit.planck_unit_system
pound_force = _mmunit.pound_force
pound_mass = _mmunit.pound_mass
pounds_force = _mmunit.pounds_force
pounds_mass = _mmunit.pounds_mass
prefix = _mmunit.prefix
print_function = _mmunit.print_function
psi = _mmunit.psi
quantity = _mmunit.quantity
radian = _mmunit.radian
radians = _mmunit.radians
second = _mmunit.second
seconds = _mmunit.seconds
si_prefixes = _mmunit.si_prefixes
si_unit_system = _mmunit.si_unit_system
sin = _mmunit.sin
sinh = _mmunit.sinh
sqrt = _mmunit.sqrt
standard_dimensions = _mmunit.standard_dimensions
stone = _mmunit.stone
stones = _mmunit.stones
sum = _mmunit.sum  # pylint: disable=redefined-builtin
sys = _mmunit.sys
tan = _mmunit.tan
tanh = _mmunit.tanh
tebi = _mmunit.tebi
temperature_dimension = _mmunit.temperature_dimension
# tera = _mmunit.tera
# teracalorie = _mmunit.teracalorie
# teracalories = _mmunit.teracalories
# teragram = _mmunit.teragram
# teragrams = _mmunit.teragrams
# terajoule = _mmunit.terajoule
# terajoules = _mmunit.terajoules
# teraliter = _mmunit.teraliter
# teraliters = _mmunit.teraliters
# terameter = _mmunit.terameter
# terameters = _mmunit.terameters
# teramolar = _mmunit.teramolar
# teramolars = _mmunit.teramolars
# teranewton = _mmunit.teranewton
# teranewtons = _mmunit.teranewtons
# terapascal = _mmunit.terapascal
# terapascals = _mmunit.terapascals
# terasecond = _mmunit.terasecond
# teraseconds = _mmunit.teraseconds
tesla = _mmunit.tesla
teslas = _mmunit.teslas
time_dimension = _mmunit.time_dimension
torr = _mmunit.torr
unit = _mmunit.unit
unit_definitions = _mmunit.unit_definitions
unit_math = _mmunit.unit_math
unit_operators = _mmunit.unit_operators
volt = _mmunit.volt
volts = _mmunit.volts
watt = _mmunit.watt
watts = _mmunit.watts
week = _mmunit.week
weeks = _mmunit.weeks
yard = _mmunit.yard
yards = _mmunit.yards
year = _mmunit.year
years = _mmunit.years
yobi = _mmunit.yobi
# yotta = _mmunit.yotta
# yottacalorie = _mmunit.yottacalorie
# yottacalories = _mmunit.yottacalories
# yottagram = _mmunit.yottagram
# yottagrams = _mmunit.yottagrams
# yottajoule = _mmunit.yottajoule
# yottajoules = _mmunit.yottajoules
# yottaliter = _mmunit.yottaliter
# yottaliters = _mmunit.yottaliters
# yottameter = _mmunit.yottameter
# yottameters = _mmunit.yottameters
# yottamolar = _mmunit.yottamolar
# yottamolars = _mmunit.yottamolars
# yottanewton = _mmunit.yottanewton
# yottanewtons = _mmunit.yottanewtons
# yottapascal = _mmunit.yottapascal
# yottapascals = _mmunit.yottapascals
# yottasecond = _mmunit.yottasecond
# yottaseconds = _mmunit.yottaseconds
# yotto = _mmunit.yotto
# yottocalorie = _mmunit.yottocalorie
# yottocalories = _mmunit.yottocalories
# yottogram = _mmunit.yottogram
# yottograms = _mmunit.yottograms
# yottojoule = _mmunit.yottojoule
# yottojoules = _mmunit.yottojoules
# yottoliter = _mmunit.yottoliter
# yottoliters = _mmunit.yottoliters
# yottometer = _mmunit.yottometer
# yottometers = _mmunit.yottometers
# yottomolar = _mmunit.yottomolar
# yottomolars = _mmunit.yottomolars
# yottonewton = _mmunit.yottonewton
# yottonewtons = _mmunit.yottonewtons
# yottopascal = _mmunit.yottopascal
# yottopascals = _mmunit.yottopascals
# yottosecond = _mmunit.yottosecond
# yottoseconds = _mmunit.yottoseconds
zebi = _mmunit.zebi
# zepto = _mmunit.zepto
# zeptocalorie = _mmunit.zeptocalorie
# zeptocalories = _mmunit.zeptocalories
# zeptogram = _mmunit.zeptogram
# zeptograms = _mmunit.zeptograms
# zeptojoule = _mmunit.zeptojoule
# zeptojoules = _mmunit.zeptojoules
# zeptoliter = _mmunit.zeptoliter
# zeptoliters = _mmunit.zeptoliters
# zeptometer = _mmunit.zeptometer
# zeptometers = _mmunit.zeptometers
# zeptomolar = _mmunit.zeptomolar
# zeptomolars = _mmunit.zeptomolars
# zeptonewton = _mmunit.zeptonewton
# zeptonewtons = _mmunit.zeptonewtons
# zeptopascal = _mmunit.zeptopascal
# zeptopascals = _mmunit.zeptopascals
# zeptosecond = _mmunit.zeptosecond
# zeptoseconds = _mmunit.zeptoseconds
# zetta = _mmunit.zetta
# zettacalorie = _mmunit.zettacalorie
# zettacalories = _mmunit.zettacalories
# zettagram = _mmunit.zettagram
# zettagrams = _mmunit.zettagrams
# zettajoule = _mmunit.zettajoule
# zettajoules = _mmunit.zettajoules
# zettaliter = _mmunit.zettaliter
# zettaliters = _mmunit.zettaliters
# zettameter = _mmunit.zettameter
# zettameters = _mmunit.zettameters
# zettamolar = _mmunit.zettamolar
# zettamolars = _mmunit.zettamolars
# zettanewton = _mmunit.zettanewton
# zettanewtons = _mmunit.zettanewtons
# zettapascal = _mmunit.zettapascal
# zettapascals = _mmunit.zettapascals
# zettasecond = _mmunit.zettasecond
# zettaseconds = _mmunit.zettaseconds
