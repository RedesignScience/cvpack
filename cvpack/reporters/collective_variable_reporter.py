"""
.. class:: CollectiveVariableReporter
   :platform: Linux, MacOS, Windows
   :synopsis: This module provides classes for reporting simulation data

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import io
import typing as t

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from ..collective_variable import CollectiveVariable


class CollectiveVariableReporter(mmapp.StateDataReporter):
    """
    Reports values and/or effective masses of collective variables during an OpenMM
    `Simulation`_.

    To use it, create a :class:`CollectiveVariableReporter` object and add it to the
    `Simulation`_'s list of reporters (see example below). The reporter writes data to
    a file or file-like object at regular intervals. The set of data to write is
    configurable using lists of :class:`CollectiveVariable` objects passed to the
    constructor. The data is written in comma-separated-value (CSV) format by default,
    but the user can specify a different separator.

    .. _Simulation: http://docs.openmm.org/latest/api-python/generated/
        openmm.app.simulation.Simulation.html

    Parameters
    ----------
    file
        The file to write to. This can be a file name or a file object.
    reportInterval
        The interval (in time steps) at which to report data.
    values
        The collective variables whose values will be reported. These objects must
        have distinct names and have been added to the simulation's :OpenMM:`System`.
    masses
        The collective variable whose effective masses will be reported. They can be
        the same objects passed to the `values` argument or distinct ones. The same
        restrictions apply.
    separator
        The separator to use between columns in the file.
    append
        If `True`, omit the header line and append the report to an existing file.

    Raises
    ------
    ValueError
        If `values` and `masses` are both empty.
    ValueError
        If `values` and `masses` contain non-`CollectiveVariable` objects.

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from math import pi
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> phi.addToSystem(model.system)
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    >>> psi.addToSystem(model.system)
    >>> reporter = cvpack.reporters.CollectiveVariableReporter(
    ...     stdout, 100, [phi, psi], [phi, psi], step=True,
    ... )
    >>> integrator = openmm.LangevinIntegrator(
    ...     300 * unit.kelvin,
    ...     1 / unit.picosecond,
    ...     2 * unit.femtosecond,
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> simulation = app.Simulation(model.topology, model.system, integrator)
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(1000)  # doctest: +SKIP
    #"Step","phi (rad)",...,"mass[psi] (nm**2 Da/(rad**2))"
    100,2.7102...,3.1221...,0.04970...,0.05386...
    200,2.1573...,2.9959...,0.04481...,0.05664...
    300,2.0859...,-3.001...,0.04035...,0.04506...
    400,2.8061...,3.0792...,0.05399...,0.04992...
    500,-2.654...,3.1139...,0.04784...,0.05592...
    600,3.1390...,-3.071...,0.05137...,0.05063...
    700,2.1133...,3.1047...,0.04145...,0.04724...
    800,1.7348...,-3.004...,0.04123...,0.05548...
    900,1.6273...,3.1277...,0.03007...,0.05271...
    1000,1.680...,2.9692...,0.03749...,0.04450...
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        file: t.Union[str, io.TextIOBase],
        reportInterval: int,
        values: t.Sequence[CollectiveVariable] = (),
        masses: t.Sequence[CollectiveVariable] = (),
        step: bool = False,
        time: bool = False,
        separator: str = ",",
        append: bool = False,
    ) -> None:
        super().__init__(
            file,
            reportInterval,
            step=step,
            time=time,
            separator=separator,
            append=append,
        )
        self._values = values
        self._masses = masses
        self._validate()

    def _validate(self) -> None:
        if not (self._values or self._masses):
            raise ValueError("Arguments 'values' and 'masses' cannot be both empty")
        for cv in self._values + self._masses:
            if not isinstance(cv, CollectiveVariable):
                raise TypeError(
                    "All items in values/masses must be CollectiveVariable instances"
                )
        if len({cv.getName() for cv in self._values}) < len(self._values):
            raise ValueError("All CVs in 'values' must have distinct names")
        if len({cv.getName() for cv in self._masses}) < len(self._masses):
            raise ValueError("All CVs in 'masses' must have distinct names")

    def _constructHeaders(self) -> t.List[str]:
        headers = []
        if self._step:
            headers.append("Step")
        if self._time:
            headers.append("Time (ps)")
        for cv in self._values:
            headers.append(f"{cv.getName()} ({cv.getUnit().get_symbol()})")
        for cv in self._masses:
            headers.append(f"mass[{cv.getName()}] ({cv.getMassUnit().get_symbol()})")
        return headers

    def _constructReportValues(  # pylint: disable=too-many-branches
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> t.List[float]:
        values = []
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(mmunit.picosecond))
        context = simulation.context
        for cv in self._values:
            values.append(cv.getValue(context) / cv.getUnit())
        for cv in self._masses:
            values.append(cv.getEffectiveMass(context) / cv.getMassUnit())
        return values
