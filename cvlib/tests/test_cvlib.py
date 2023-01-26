"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import sys

import openmm
import pytest
from openmm import unit
from openmmtools import testsystems

import cvlib


def test_cvlib_imported():
    """
    Sample test, will always pass so long as import statement worked.

    """
    assert "cvlib" in sys.modules


def test_effective_mass():
    """
    Test effective mass evaluation at a given context.

    """
    model = testsystems.AlanineDipeptideVacuum()
    rg_cv = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
    model.system.addForce(rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    effective_mass = rg_cv.effectiveMassInContext(context)
    assert effective_mass/effective_mass.unit == pytest.approx(30.946932)


def perform_common_tests(collectiveVariable: cvlib.AbstractCollectiveVariable) -> None:
    """
    Function to be called in every individual cv test.

    """
    assert collectiveVariable.getName() == collectiveVariable.__class__.__name__
    assert (1*collectiveVariable.getUnit()).value_in_unit_system(unit.md_unit_system) == 1


def test_radius_of_gyration():
    """
    Test whether the radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    rgsq = ((positions - centroid) ** 2).sum() / model.system.getNumParticles()
    rg_cv = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
    perform_common_tests(rg_cv)
    model.system.addForce(rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rgval = rg_cv.evaluateInContext(context).value_in_unit(unit.nanometers)
    assert rgval**2 == pytest.approx(rgsq)
