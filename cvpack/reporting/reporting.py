"""
.. module:: reporting
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
from ..meta_collective_variable import MetaCollectiveVariable


class CollectiveVariableReporter(mmapp.StateDataReporter):
    """
    Reports values and/or effective masses of collective variables during an OpenMM
    `Simulation`_.

    Create a :class:`CollectiveVariableReporter` add it to the `Simulation`_'s list
    of reporters (see example below). The reporter writes data to a file or file-like
    object at regular intervals. The set of data to write is configurable using boolean
    flags passed to the constructor. The data is written in comma-separated-value (CSV)
    format by default, but the user can specify a different separator.

    .. _Simulation: http://docs.openmm.org/latest/api-python/generated/
        openmm.app.simulation.Simulation.html

    Parameters
    ----------
    file
        The file to write to. This can be a file name or a file object.
    reportInterval
        The interval (in time steps) at which to report data.
    variables
        The collective variable whose values and/or effective masses will be reported.
        These must be the same objects added to the simulation's :OpenMM:`System`.
    metaCVs
        The meta-collective variables whose inner variables will be reported. These must
        be the same objects added to the simulation's :OpenMM:`System`.
    step
        Whether to report the current step index.
    time
        Whether to report the current simulation time.
    values
        Whether to report the current values of the collective variables.
    masses
        Whether to report the current effective masses of the collective variables.
    separator
        The separator to use between columns in the file.
    append
        If `True`, omit the header line and append the report to an existing file.

    Raises
    ------
    ValueError
        If `variables` and `metaCVs` are both empty.
    ValueError
        If `values` and `masses` are both `False`.

    Examples
    --------
    >>> import cvpack
    >>> from cvpack import reporting
    >>> import openmm
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    >>> umbrella = cvpack.MetaCollectiveVariable(
    ...     "50*(min(dphi,2*pi-dphi)^2+min(dpsi,2*pi-dpsi)^2)"
    ...     "; dphi=abs(phi-5*pi/6); dpsi=abs(psi+5*pi/6)",
    ...     [phi, psi],
    ...     unit.kilojoules_per_mole,
    ...     name="umbrella",
    ...     pi=3.141592653589793,
    ... )
    >>> reporter = reporting.CollectiveVariableReporter(
    ...     stdout,
    ...     100,
    ...     variables=[umbrella],
    ...     metaCVs=[umbrella],
    ...     step=True,
    ...     values=True,
    ...     masses=True
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
    >>> simulation.step(1000)
    #"Step","umbrella (kJ/mol)",...,"psi mass (nm**2 Da/(rad**2))"
    100,11.2618...,1.8774...e-05,2.3684...,0.04388...,-3.0217...,0.05552...
    200,7.46382...,4.8803...e-05,2.8851...,0.04968...,-2.8971...,0.05230...
    300,2.55804...,8.9086...e-05,2.4311...,0.04637...,-2.4905...,0.03721...
    400,6.19945...,4.6824...e-05,2.9678...,0.05449...,-2.6577...,0.04840...
    500,8.82728...,3.0253...e-05,2.5838...,0.04584...,-3.0367...,0.05485...
    600,3.76160...,7.9099...e-05,2.7248...,0.04772...,-2.8706...,0.05212...
    700,3.38895...,6.3273...e-05,2.5583...,0.04999...,-2.8714...,0.04674...
    800,1.07166...,0.00028746...,2.7104...,0.05321...,-2.7314...,0.05418...
    900,8.58602...,2.4899...e-05,2.4391...,0.04096...,-2.9917...,0.04675...
    1000,5.8404...,5.1011...e-05,2.7584...,0.04951...,-2.9295...,0.05030...
    """

    def __init__(
        self,
        file: t.Union[str, io.TextIOBase],
        reportInterval: int,
        variables: t.Sequence[CollectiveVariable] = (),
        metaCVs: t.Sequence[MetaCollectiveVariable] = (),
        step: bool = False,
        time: bool = False,
        values: bool = False,
        masses: bool = False,
        separator: str = ",",
        append: bool = False,
    ) -> None:
        if not (variables or metaCVs):
            raise ValueError("Arguments 'variables' and 'metaCVs' cannot be both empty")
        if not (values or masses):
            raise ValueError("Arguments 'values' and 'masses' cannot be both False")
        super().__init__(
            file,
            reportInterval,
            step=step,
            time=time,
            separator=separator,
            append=append,
        )
        self._variables = variables
        self._meta_cvs = metaCVs
        self._values = values
        self._masses = masses

    def _constructReportValues(  # pylint: disable=too-many-branches
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> t.List[float]:
        values = []
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append(state.getTime().value_in_unit(mmunit.picosecond))
        context = simulation.context
        for cv in self._variables:
            if self._values:
                values.append(cv.getValue(context) / cv.getUnit())
            if self._masses:
                values.append(cv.getEffectiveMass(context) / cv.getMassUnit())
        for cv in self._meta_cvs:
            cv_values = cv.getInnerValues(context) if self._values else {}
            cv_masses = cv.getInnerEffectiveMasses(context) if self._masses else {}
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
            if self._masses:
                headers.append(f"{cv.getName()} mass ({cv.getMassUnit().get_symbol()})")

        for cv in self._variables:
            add_headers(cv)
        for cv in self._meta_cvs:
            for inner_cv in cv.getInnerVariables():
                add_headers(inner_cv)
        return headers
