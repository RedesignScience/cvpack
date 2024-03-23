"""
.. class:: Utils
   :platform: Linux, MacOS, Windows
   :synopsis: Utility functions and classes for CVpack

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import inspect
import typing as t

import numpy as np
import openmm
from numpy import typing as npt
from openmm import XmlSerializer
from openmm import unit as mmunit, app as mmapp

from .serialization import Serializable, SerializableAtom, SerializableResidue
from .units import Quantity, Unit, value_in_md_units

# pylint: disable=protected-access,c-extension-no-member


class NonbondedForceSurrogate(
    Serializable
):  # pylint: disable=too-many-instance-attributes
    """A surrogate class for the NonbondedForce class in OpenMM."""

    def __init__(self, other: openmm.NonbondedForce) -> None:
        self._cutoff = other.getCutoffDistance()
        self._uses_pbc = other.usesPeriodicBoundaryConditions()
        self._num_particles = other.getNumParticles()
        self._particle_parameters = list(
            map(other.getParticleParameters, range(self._num_particles))
        )
        self._num_exceptions = other.getNumExceptions()
        self._exception_parameters = list(
            map(other.getExceptionParameters, range(self._num_exceptions))
        )
        self._use_switching_function = other.getUseSwitchingFunction()
        self._switching_distance = other.getSwitchingDistance()

    def __getstate__(self) -> t.Dict[str, str]:
        return {
            "cutoff": self.getCutoffDistance(),
            "uses_pbc": self.usesPeriodicBoundaryConditions(),
            "num_particles": self.getNumParticles(),
            "particle_parameters": [
                self.getParticleParameters(i) for i in range(self.getNumParticles())
            ],
            "num_exceptions": self.getNumExceptions(),
            "exception_parameters": [
                self.getExceptionParameters(i) for i in range(self.getNumExceptions())
            ],
            "use_switching_function": self.getUseSwitchingFunction(),
            "switching_distance": self.getSwitchingDistance(),
        }

    def __setstate__(self, state: t.Dict[str, str]) -> None:
        self._cutoff = state["cutoff"]
        self._uses_pbc = state["uses_pbc"]
        self._num_particles = state["num_particles"]
        self._particle_parameters = state["particle_parameters"]
        self._num_exceptions = state["num_exceptions"]
        self._exception_parameters = state["exception_parameters"]
        self._use_switching_function = state["use_switching_function"]
        self._switching_distance = state["switching_distance"]

    def getCutoffDistance(self) -> float:
        """Get the cutoff distance."""
        return value_in_md_units(self._cutoff)

    def usesPeriodicBoundaryConditions(self) -> bool:
        """Return whether periodic boundary conditions are used."""
        return self._uses_pbc

    def getNumParticles(self) -> int:
        """Get the number of particles."""
        return self._num_particles

    def getParticleParameters(self, index: int) -> t.Tuple[float, float, float]:
        """Get the parameters of a particle at the given index."""
        return tuple(map(value_in_md_units, self._particle_parameters[index]))

    def getNumExceptions(self):
        """Get the number of exceptions."""
        return self._num_exceptions

    def getExceptionParameters(
        self, index: int
    ) -> t.Tuple[int, int, float, float, float]:
        """Get the parameters of an exception at the given index."""
        i, j, *params = self._exception_parameters[index]
        return i, j, *map(value_in_md_units, params)

    def getUseSwitchingFunction(self) -> bool:
        """Return whether a switching function is used."""
        return self._use_switching_function

    def getSwitchingDistance(self) -> float:
        """Get the switching distance."""
        return value_in_md_units(self._switching_distance)


NonbondedForceSurrogate.registerTag("!cvpack.NonbondedForce")


def evaluate_in_context(
    forces: t.Union[openmm.Force, t.Iterable[openmm.Force]], context: openmm.Context
) -> t.Union[float, t.List[float]]:
    """Evaluate the potential energies of OpenMM Forces in a given context.

    Parameters
    ----------
        forces
            The forces to be evaluated.
        context
            The context in which the force will be evaluated.

    Returns
    -------
        float
            The potential energy of the force in the given context.
    """
    is_single = isinstance(forces, openmm.Force)
    if is_single:
        forces = [forces]
    system = openmm.System()
    for _ in range(context.getSystem().getNumParticles()):
        system.addParticle(1.0)
    for i, force in enumerate(forces):
        force_copy = XmlSerializer.deserialize(XmlSerializer.serialize(force))
        force_copy.setForceGroup(i)
        system.addForce(force_copy)
    state = context.getState(getPositions=True)
    context = openmm.Context(system, openmm.VerletIntegrator(1.0))
    context.setPositions(state.getPositions())
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    energies = []
    for i in range(len(forces)):
        state = context.getState(  # pylint: disable=unexpected-keyword-arg
            getEnergy=True, groups=1 << i
        )
        energies.append(value_in_md_units(state.getPotentialEnergy()))
    return energies[0] if is_single else tuple(energies)


def convert_to_matrix(array: npt.ArrayLike) -> t.Tuple[np.ndarray, int, int]:
    """Convert a 1D or 2D array-like object to a 2D numpy array.

    Parameters
    ----------
        array : array_like
            The array to be converted.

    Returns
    -------
        numpy.ndarray
            The 2D numpy array.
        int
            The number of rows in the array.
        int
            The number of columns in the array.
    """
    array = np.atleast_2d(array)
    numrows, numcols, *other_dimensions = array.shape
    if other_dimensions:
        raise ValueError("Array-like object cannot have more than two dimensions.")
    return array, numrows, numcols


def get_single_force_state(
    force: openmm.Force,
    context: openmm.Context,
    getEnergy: bool = False,
    getForces: bool = False,
) -> openmm.State:
    """
    Get an OpenMM State containing the potential energy and/or force values computed
    from a single force object.

    Parameters
    ----------
    force
        The force object from which the state should be extracted
    context
        The context from which the state should be extracted
    getEnergy
        If True, the potential energy will be computed
    getForces
        If True, the forces will be computed

    Returns
    -------
    openmm.State
        The state containing the requested values

    Raises
    ------
    ValueError
        If this force is not present in the given context
    """
    forces_and_groups = [
        (f, f.getForceGroup()) for f in context.getSystem().getForces()
    ]
    if not any(f.this == force.this for f, _ in forces_and_groups):
        raise RuntimeError("This force is not present in the given context.")
    self_group = force.getForceGroup()
    other_groups = {g for f, g in forces_and_groups if f.this != force.this}
    if self_group not in other_groups:
        return context.getState(
            getEnergy=getEnergy, getForces=getForces, groups=1 << self_group
        )
    new_group = force._setUnusedForceGroup(context.getSystem())
    context.reinitialize(preserveState=True)
    state = context.getState(
        getEnergy=getEnergy, getForces=getForces, groups=1 << new_group
    )
    force.setForceGroup(self_group)
    context.reinitialize(preserveState=True)
    return state


def compute_effective_mass(force: openmm.Force, context: openmm.Context) -> float:
    r"""
    Compute the effective mass of an :OpenMM:`Force` at a given :OpenMM:`Context`.

    Parameters
    ----------
    force
        The force object from which the effective mass should be computed
    context
        The context at which the force's effective mass should be evaluated

    Returns
    -------
    float
        The effective mass of the force at the given context
    """
    state = get_single_force_state(force, context, getForces=True)
    get_mass = functools.partial(
        openmm._openmm.System_getParticleMass, context.getSystem()
    )
    force_vectors = state.getForces(asNumpy=True)._value
    squared_forces = np.sum(np.square(force_vectors), axis=1)
    nonzeros = np.nonzero(squared_forces)[0]
    if nonzeros.size == 0:
        return mmunit.Quantity(np.inf, force._mass_unit)
    mass_values = np.fromiter(map(get_mass, nonzeros), dtype=np.float64)
    return 1.0 / np.sum(squared_forces[nonzeros] / mass_values)


def preprocess_args(func: t.Callable) -> t.Callable:
    """
    A decorator that converts instances of unserializable classes to their
    serializable counterparts.

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Example
    -------
    >>> from cvpack import units, utils
    >>> from openmm import unit as mmunit
    >>> @utils.preprocess_units
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
        if isinstance(data, mmapp.Atom):
            return SerializableAtom(data)
        if isinstance(data, mmapp.Residue):
            return SerializableResidue(data)
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
