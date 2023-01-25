"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import sys

import openmm
import pytest
from openmmtools import testsystems

import cvlib


def test_cvlib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cvlib" in sys.modules


def test_radius_of_gyration():
    model = testsystems.AlanineDipeptideVacuum()
    r = model.positions._value
    rmean = r.mean(axis=0)
    rgsq = ((r - rmean) ** 2).sum() / model.system.getNumParticles()

    cv = cvlib.RadiusOfGyration(range(model.topology._numAtoms))
    cv.setForceGroup(1)
    model.system.addForce(cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rg = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    assert rg * rg == pytest.approx(rgsq)
