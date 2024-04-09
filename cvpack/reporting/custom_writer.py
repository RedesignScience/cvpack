"""
.. class:: CustomWriter
   :platform: Linux, MacOS, Windows
   :synopsis: An abstract class for StateDataReporter writers

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm
from openmm import app as mmapp


@t.runtime_checkable
class CustomWriter(t.Protocol):
    """
    An abstract class for StateDataReporter writers
    """

    def getHeaders(self) -> t.List[str]:
        """
        Gets a list of strigs containing the headers to be added to the report.
        """
        raise NotImplementedError("Method getHeaders not implemented")

    def getReportValues(
        self,
        simulation: mmapp.Simulation,
        state: openmm.State,
    ) -> t.List[float]:
        """
        Gets a list of floats containing the values to be added to the report.

        Parameters
        ----------
        simulation
            The simulation object.
        state
            The state object.
        """
        raise NotImplementedError("Method getReportValues not implemented")
