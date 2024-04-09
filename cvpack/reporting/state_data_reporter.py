"""
.. class:: StateDataReporter
   :platform: Linux, MacOS, Windows
   :synopsis: An extension of the OpenMM `StateDataReporter`_ class to include writers

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import io
import typing as t

import openmm
from openmm import app as mmapp

from .custom_writer import CustomWriter


class StateDataReporter(mmapp.StateDataReporter):
    """
    An extended version of OpenMM's `openmm.app.StateDataReporter`_ class that includes
    custom writers for reporting additional simulation data.

    .. _openmm.app.StateDataReporter: http://docs.openmm.org/latest/api-python/
        generated/openmm.app.statedatareporter.StateDataReporter.html

    A custom writer is an object that includes the methods two particular methods.
    The first one is ``getHeaders``, which returns a list of strings containing the
    headers to be added to the report. It has the following signature:

    .. code-block::

        def getHeaders(self) -> List[str]:
            pass

    The second method is ``getReportValues``, which accepts an `openmm.app.Simulation`_
    and an :OpenMM:`State` as arguments and returns a list of floats containing the
    values to be added to the report. It has the following signature:

    .. _openmm.app.Simulation: http://docs.openmm.org/latest/api-python/generated/
        openmm.app.simulation.Simulation.html

    .. code-block::

        def getReportValues(
            self,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> List[float]:
            pass

    Parameters
    ----------
    file
        The file to write to. This can be a file name or a file object.
    reportInterval
        The interval (in time steps) at which to report data.
    writers
        A sequence of custom writers.
    **kwargs
        Additional keyword arguments to be passed to the `StateDataReporter`_
        constructor.

    Examples
    --------
    >>> import cvpack
    >>> import openmm
    >>> from cvpack.reporting import StateDataReporter, CVWriter
    >>> from math import pi
    >>> from openmm import app, unit
    >>> from sys import stdout
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> phi.addToSystem(model.system)
    >>> psi = cvpack.Torsion(8, 14, 16, 18, name="psi")
    >>> psi.addToSystem(model.system)
    >>> reporter = StateDataReporter(
    ...     stdout,
    ...     100,
    ...     step=True,
    ...     writers=[
    ...         CVWriter(phi, value=True, emass=True),
    ...         CVWriter(psi, value=True, emass=True),
    ...     ],
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
    #"Step","phi (rad)",...,"emass[psi] (nm**2 Da/(rad**2))"
    100,2.7102...,0.04970...,3.1221...,0.05386...
    200,2.1573...,0.04481...,2.9959...,0.05664...
    300,2.0859...,0.04035...,-3.001...,0.04506...
    400,2.8061...,0.05399...,3.0792...,0.04992...
    500,-2.654...,0.04784...,3.1139...,0.05592...
    600,3.1390...,0.05137...,-3.071...,0.05063...
    700,2.1133...,0.04145...,3.1047...,0.04724...
    800,1.7348...,0.04123...,-3.004...,0.05548...
    900,1.6273...,0.03007...,3.1277...,0.05271...
    1000,1.680...,0.03749...,2.9692...,0.04450...
    """

    def __init__(
        self,
        file: t.Union[str, io.TextIOBase],
        reportInterval: int,
        writers: t.Sequence[CustomWriter] = (),
        **kwargs,
    ) -> None:
        super().__init__(file, reportInterval, **kwargs)
        if not all(isinstance(w, CustomWriter) for w in writers):
            raise TypeError("All items in writers must satisfy the Writer protocol")
        self._writers = writers
        self._back_steps = sum([self._speed, self._elapsedTime, self._remainingTime])

    def _expand(self, sequence: list, addition: t.Iterable) -> list:
        pos = len(sequence) - self._back_steps
        return sum(addition, sequence[:pos]) + sequence[pos:]

    def _constructHeaders(self) -> t.List[str]:
        return self._expand(
            super()._constructHeaders(),
            (w.getHeaders() for w in self._writers),
        )

    def _constructReportValues(
        self, simulation: mmapp.Simulation, state: openmm.State
    ) -> t.List[float]:
        return self._expand(
            super()._constructReportValues(simulation, state),
            (w.getReportValues(simulation, state) for w in self._writers),
        )
