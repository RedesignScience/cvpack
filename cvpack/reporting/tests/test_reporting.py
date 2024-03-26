"""
Unit test for the reporting subpackage of CVPack
"""

import os
import tempfile
from math import pi

import openmm
from openmm import app, unit
from openmmtools import testsystems

import cvpack


def test_collective_variable_reporter():
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    umbrella = cvpack.MetaCollectiveVariable(
        f"50*min(delta,2*pi-delta)^2" "; delta=abs(phi-5*pi/6)" f"; pi={pi}",
        [phi],
        unit.kilojoules_per_mole,
        name="umbrella",
    )
    with tempfile.TemporaryDirectory() as dirpath:
        integrator = openmm.LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
        )
        integrator.setRandomNumberSeed(1234)
        umbrella.addToSystem(model.system)
        simulation = app.Simulation(model.topology, model.system, integrator)
        simulation.context.setPositions(model.positions)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5678)
        with open(os.path.join(dirpath, "report.csv"), "w") as file:
            reporter = cvpack.reporting.CollectiveVariableReporter(
                file,
                1,
                [umbrella],
                step=True,
                values=True,
                effectiveMasses=True,
            )
            simulation.reporters.append(reporter)
            simulation.step(10)
        with open(os.path.join(dirpath, "report.csv"), "r") as file:
            assert file.readline() == ",".join(
                [
                    '#"Step"',
                    '"umbrella (kJ/mol)"',
                    '"umbrella mass (nm**2 mol**2 Da/(kJ**2))"',
                    '"phi (rad)"',
                    '"phi mass (nm**2 Da/(rad**2))"\n',
                ]
            )
