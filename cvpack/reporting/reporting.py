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
    :class:`~openmm.app.Simulation`'s list of reporters. The set of data to write is
    configurable using boolean flags passed to the constructor.

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
                headers.append(f"{cv.getName()} ({cv.getUnit()})")
            if self._effective_masses:
                headers.append(f"mass[{cv.getName}] ({cv.getMassUnit()})")

        for variable in self._variables:
            add_headers(variable)
            if isinstance(variable, MetaCV) and not self._exclude_inner_cvs:
                for inner in variable.getInnerCollectiveVariables():
                    add_headers(inner)
        return headers
