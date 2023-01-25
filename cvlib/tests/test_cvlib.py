"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import os
import sys

import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import cvlib


def test_cvlib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cvlib" in sys.modules


def test_radius_of_gyration():
    model = testsystems.AlanineDipeptideVacuum()
    R = model.positions._value
    N = len(R)
    Rmean = sum(R, openmm.Vec3(0, 0, 0)) / N
    RgSqVal = 0.0
    for r in R:
        dr = r - Rmean
        RgSqVal += dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2
    RgSqVal /= N

    Rg = cvlib.RadiusOfGyration(range(model.topology._numAtoms))
    Rg.setForceGroup(1)
    model.system.addForce(Rg)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    Rg = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    assert Rg * Rg == pytest.approx(RgSqVal)
