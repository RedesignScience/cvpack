"""
.. class:: MetaCVReporter
   :platform: Linux, MacOS, Windows
   :synopsis: This module provides classes for reporting simulation data

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import io
import typing as t

import openmm
from openmm import app as mmapp
from openmm import unit as mmunit

from ..meta_collective_variable import MetaCollectiveVariable


class MetaCVReporter(mmapp.StateDataReporter):
    """
    Reports values and/or effective masses of inner variables of a meta-collective
    variable during an OpenMM `Simulation`_. Also reports the values of parameters and
    derivatives of the meta-collective variable with respect to its parameters.

    To use it, create a :class:`MetaCVReporter` object and add it to the
    `Simulation`_'s list of reporters (see example below). The reporter writes data
    to a file or file-like object at regular intervals. The set of data to write is
    configurable using lists of :class:`~cvpack.CollectiveVariable` objects passed
    to the constructor. The data is written in comma-separated-value (CSV) format by
    default, but the user can specify a different separator.

    .. _Simulation: http://docs.openmm.org/latest/api-python/generated/
        openmm.app.simulation.Simulation.html

    Parameters
    ----------
    file
        The file to write to. This can be a file name or a file object.
    reportInterval
        The interval (in time steps) at which to report data.
    metaCV
        The meta-collective variable whose associated values will be reported.
    step
        Whether to report the current step index.
    time
        Whether to report the current simulation time.
    innerValues
        The names of the inner variables whose values will be reported.
    innerMasses
        The names of the inner variables whose effective masses will be reported.
    parameterValues
        The names of the parameters whose values will be reported.
    parameterDerivatives
        The names of the parameters with respect to which the derivatives of the
        meta-collective variable will be reported.
    separator
        The separator to use between columns in the file.
    append
        If `True`, omit the header line and append the report to an existing file.

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
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    >>> umbrella = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*(min(dphi,{2*pi}-dphi)^2+min(dpsi,{2*pi}-dpsi)^2)"
    ...     "; dphi=abs(phi-phi0); dpsi=abs(psi-psi0)",
    ...     [phi, psi],
    ...     unit.kilojoules_per_mole,
    ...     name="umbrella",
    ...     kappa=100 * unit.kilojoules_per_mole/unit.radian**2,
    ...     phi0=5*pi/6 * unit.radian,
    ...     psi0=-5*pi/6 * unit.radian,
    ... )
    >>> reporter = cvpack.reporters.MetaCVReporter(
    ...     stdout,
    ...     100,
    ...     umbrella,
    ...     step=True,
    ...     innerValues=["phi", "psi"],
    ...     innerMasses=["phi", "psi"],
    ...     parameterValues=["phi0", "psi0"],
    ...     parameterDerivatives=["phi0", "psi0"],
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
    #"Step","phi (rad)",...,"diff[umbrella,psi0] (kJ/(mol rad))"
    100,2.36849...,40.3718...
    200,2.88515...,27.9109...
    300,2.43112...,-12.743...
    400,2.96786...,3.97688...
    500,2.58383...,41.8782...
    600,2.72482...,25.2626...
    700,2.55836...,25.3424...
    800,2.71046...,11.3498...
    900,2.43913...,37.3804...
    1000,2.7584...,31.1599...
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        file: t.Union[str, io.TextIOBase],
        reportInterval: int,
        metaCV: MetaCollectiveVariable,
        innerValues: t.Sequence[str] = (),
        innerMasses: t.Sequence[str] = (),
        parameterValues: t.Sequence[str] = (),
        parameterDerivatives: t.Sequence[str] = (),
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
        inner_cvs = {cv.getName(): cv for cv in metaCV.getInnerVariables()}
        parameters = metaCV.getParameterDefaultValues()
        self._meta_cv = metaCV
        self._values = [inner_cvs[name] for name in innerValues]
        self._masses = [inner_cvs[name] for name in innerMasses]
        self._parameters = {name: parameters[name] for name in parameterValues}
        self._derivatives = {name: parameters[name] for name in parameterDerivatives}

    def _constructHeaders(self) -> t.List[str]:
        headers = []
        if self._step:
            headers.append("Step")
        if self._time:
            headers.append("Time (ps)")

        def add_header(name: str, unit: mmunit.Unit) -> None:
            headers.append(f"{name} ({unit.get_symbol()})")

        for cv in self._values:
            add_header(cv.getName(), cv.getUnit())
        for cv in self._masses:
            add_header(f"mass[{cv.getName()}]", cv.getMassUnit())
        for name, quantity in self._parameters.items():
            add_header(name, quantity.unit)
        for name, quantity in self._derivatives.items():
            add_header(
                f"diff[{self._meta_cv.getName()},{name}]",
                self._meta_cv.getUnit() / quantity.unit,
            )
        return headers

    def _constructReportValues(  # pylint: disable=too-many-branches
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> t.List[float]:
        values = []
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(mmunit.picosecond))
        if self._values:
            inner_values = self._meta_cv.getInnerValues(simulation.context)
            for cv in self._values:
                values.append(inner_values[cv.getName()] / cv.getUnit())
        if self._masses:
            inner_masses = self._meta_cv.getInnerEffectiveMasses(simulation.context)
            for cv in self._masses:
                values.append(inner_masses[cv.getName()] / cv.getMassUnit())
        if self._parameters:
            parameters = self._meta_cv.getParameterValues(simulation.context)
            for name, quantity in self._parameters.items():
                values.append(parameters[name] / quantity.unit)
        if self._derivatives:
            derivatives = self._meta_cv.getParameterDerivatives(simulation.context)
            for name, quantity in self._derivatives.items():
                values.append(
                    derivatives[name] / (self._meta_cv.getUnit() / quantity.unit)
                )
        return values
