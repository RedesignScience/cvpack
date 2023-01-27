"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import sys

import numpy as np
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
    assert effective_mass / effective_mass.unit == pytest.approx(30.946932)


def perform_common_tests(collectiveVariable: cvlib.AbstractCollectiveVariable) -> None:
    """
    Function to be called in every individual cv test.

    """
    assert collectiveVariable.getName() == collectiveVariable.__class__.__name__
    unity = 1 * collectiveVariable.getUnit()
    assert unity.value_in_unit_system(unit.md_unit_system) == 1


def test_distance():
    """
    Test whether a distance is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atom1, atom2 = 0, 5
    distance = cvlib.Distance(atom1, atom2)
    perform_common_tests(distance)
    model.system.addForce(distance)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = distance.evaluateInContext(context).value_in_unit(distance.getUnit())
    value2 = np.sqrt(np.sum(((model.positions[atom1] - model.positions[atom2]) ** 2)))
    assert value1 == pytest.approx(value2)


def test_angle():
    """
    Test whether an angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10]
    angle = cvlib.Angle(*atoms)
    perform_common_tests(angle)
    model.system.addForce(angle)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = angle.evaluateInContext(context).value_in_unit(angle.getUnit())
    delta = [
        model.positions[atoms[i + 1]] - model.positions[atoms[i]] for i in range(2)
    ]
    numerator = -np.dot(delta[0], delta[1])
    denominator = np.linalg.norm(delta[0]) * np.linalg.norm(delta[1])
    value2 = np.arccos(numerator / denominator)
    assert value1 == pytest.approx(value2)


def test_torsion():
    """
    Test whether a torsion angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10, 15]
    torsion = cvlib.Torsion(*atoms)
    perform_common_tests(torsion)
    model.system.addForce(torsion)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = torsion.evaluateInContext(context).value_in_unit(torsion.getUnit())
    delta = [
        model.positions[atoms[i + 1]] - model.positions[atoms[i]] for i in range(3)
    ]
    numerator = np.dot(
        np.cross(np.cross(delta[0], delta[1]), np.cross(delta[1], delta[2])),
        delta[1] / np.linalg.norm(delta[1]),
    )
    denominator = np.dot(np.cross(delta[0], delta[1]), np.cross(delta[1], delta[2]))
    value2 = np.arctan2(numerator, denominator)
    assert value1 == pytest.approx(value2)


def test_radius_of_gyration():
    """
    Test whether a radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    rgsq = np.sum((positions - centroid) ** 2) / model.system.getNumParticles()
    rg_cv = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
    perform_common_tests(rg_cv)
    model.system.addForce(rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rgval = rg_cv.evaluateInContext(context).value_in_unit(unit.nanometers)
    assert rgval**2 == pytest.approx(rgsq)
