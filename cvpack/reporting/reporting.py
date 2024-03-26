"""
.. module:: reporting
   :platform: Linux, MacOS, Windows
   :synopsis: This module provides classes for reporting simulation data

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import io
import typing as t
from collections import OrderedDict

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from ..collective_variable import CollectiveVariable
from ..meta_collective_variable import MetaCollectiveVariable as MetaCV


class CollectiveVariableReporter(mmapp.StateDataReporter):
    """
    Reports values and effective masses of collective variables during a simulation.

    To use it, create a :class:`CollectiveVariableReporter`, then add it to the
    simulation's list of reporters. The set of data to write is configurable using
    boolean flags passed to the constructor.

    By default the data is written in comma-separated-value (CSV) format, but you can
    specify a different separator to use.

    Parameters
    ----------
    file
        The file to write to, specified as a file name or file object.
    reportInterval
        The interval (in time steps) at which to write frames.
    variables
        The collective variables to report.
    step
        Whether to write the current step index to the file.
    time
        Whether to write the current time to the file.
    values
        Whether to write the current values of the collective variables to the file.
    effectiveMasses
        Whether to write the current effective masses of the collective variables to
        the file.
    excludeInnerVariables
        Whether to exclude the inner collective variables of meta-collective variables
        from the report.
    separator
        The separator to use between columns in the file.
    append
        If `True`, omit the header line and append the report to an existing file.

    Raises
    ------
    ValueError
        If `variables` is empty.
    ValueError
        If `values` and `effectiveMasses` are both `False`.

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from math import pi
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> umbrella = cvpack.MetaCollectiveVariable(
    ...     f"50*min(delta,2*pi-delta)^2"
    ...     "; delta=abs(phi-5*pi/6)"
    ...     f"; pi={pi}",
    ...     [phi],
    ...     unit.kilojoules_per_mole,
    ...     name="umbrella"
    ... )
    >>> reporter = cvpack.reporting.CollectiveVariableReporter(
    ...     stdout,
    ...     100,
    ...     [umbrella],
    ...     step=True,
    ...     values=True,
    ...     effectiveMasses=True
    ... )
    >>> integrator = openmm.LangevinIntegrator(
    ...     300 * unit.kelvin,
    ...     1 / unit.picosecond,
    ...     2 * unit.femtosecond,
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> umbrella.addToSystem(model.system)
    >>> simulation = app.Simulation(model.topology, model.system, integrator)
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(1000)  # doctest: +SKIP
    #"Step","umbrella (kJ/mol)",...,"phi (rad)","phi mass (nm**2 Da/(rad**2))"
    100,0.3834...,0.00062...,2.5304...,0.04792...
    200,0.2180...,0.00117...,2.6840...,0.05114...
    300,0.0144...,0.01772...,2.6009...,0.05122...
    400,0.6639...,0.00035...,2.5027...,0.04688...
    500,0.1658...,0.00138...,2.6755...,0.04585...
    600,1.5860...,0.00015...,2.7960...,0.04875...
    700,4.0932...,0.00005...,2.3318...,0.04457...
    800,3.8208...,0.00006...,2.8944...,0.04639...
    900,1.2252...,0.00017...,2.4614...,0.04213...
    1000,0.564...,0.00044...,2.5117...,0.05042...
    """

    def __init__(
        self,
        file: t.Union[str, io.TextIOBase],
        reportInterval: int,
        variables: t.Sequence[CollectiveVariable],
        step: bool = False,
        time: bool = False,
        values: bool = False,
        effectiveMasses: bool = False,
        excludeInnerVariables: bool = False,
        separator: str = ",",
        append: bool = False,
    ) -> None:
        if not variables:
            raise ValueError("Argument 'variables' cannot be empty")
        if not (values or effectiveMasses):
            raise ValueError(
                "Arguments 'values' and 'effectiveMasses' cannot be both False"
            )
        super().__init__(
            file,
            reportInterval,
            step=step,
            time=time,
            separator=separator,
            append=append,
        )
        self._variables = variables
        self._values = values
        self._effective_masses = effectiveMasses
        self._exclude_inner_cvs = excludeInnerVariables

    def _constructReportValues(  # pylint: disable=too-many-branches
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> t.List[float]:
        values = []
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(mmunit.picosecond))
        context = simulation.context
        cv_values = OrderedDict()
        if self._values:
            for cv in self._variables:
                cv_values[cv.getName()] = cv.getValue(context)
                if isinstance(cv, MetaCV) and not self._exclude_inner_cvs:
                    for name, value in cv.getInnerValues(context).items():
                        cv_values[name] = value
        cv_masses = OrderedDict()
        if self._effective_masses:
            for cv in self._variables:
                cv_masses[cv.getName()] = cv.getEffectiveMass(context)
                if isinstance(cv, MetaCV) and not self._exclude_inner_cvs:
                    for name, mass in cv.getInnerEffectiveMasses(context).items():
                        cv_masses[name] = mass
        for name in cv_values if self._values else cv_masses:
            if name in cv_values:
                values.append(cv_values[name] / cv_values[name].unit)
            if name in cv_masses:
                values.append(cv_masses[name] / cv_masses[name].unit)
        return values

    def _constructHeaders(self) -> t.List[str]:
        headers = []
        if self._step:
            headers.append("Step")
        if self._time:
            headers.append("Time (ps)")

        def add_headers(cv: CollectiveVariable) -> None:
            if self._values:
                headers.append(f"{cv.getName()} ({cv.getUnit().get_symbol()})")
            if self._effective_masses:
                headers.append(f"{cv.getName()} mass ({cv.getMassUnit().get_symbol()})")

        for variable in self._variables:
            add_headers(variable)
            if isinstance(variable, MetaCV) and not self._exclude_inner_cvs:
                for inner in variable.getInnerVariables():
                    add_headers(inner)
        return headers
