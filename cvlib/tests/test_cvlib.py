"""
Unit and regression test for the cvlib package.
"""

import io
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


def perform_common_tests(
    collectiveVariable: cvlib.AbstractCollectiveVariable, context: openmm.Context
) -> None:
    """
    Function to be called in every individual cv test.

    """
    # Default name must match the class name
    assert collectiveVariable.getName() == collectiveVariable.__class__.__name__

    # Unit must conform to the default OpenMM system
    unity = 1 * collectiveVariable.getUnit()
    assert unity.value_in_unit_system(unit.md_unit_system) == 1

    # Test serialization/deserialization
    pipe = io.StringIO()
    cvlib.serialization.serialize(collectiveVariable, pipe)
    pipe.seek(0)
    new_cv = cvlib.serialization.deserialize(pipe)
    context.getSystem().addForce(new_cv)
    context.reinitialize(preserveState=True)
    value1 = collectiveVariable.evaluateInContext(context)
    value2 = new_cv.evaluateInContext(context)
    assert value1 / value1.unit == value2 / value2.unit
    mass1 = collectiveVariable.effectiveMassInContext(context)
    mass2 = new_cv.effectiveMassInContext(context)
    assert mass1 / mass1.unit == mass2 / mass2.unit


def test_distance():
    """
    Test whether a distance is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atom1, atom2 = 0, 5
    distance = cvlib.Distance(atom1, atom2)
    model.system.addForce(distance)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = distance.evaluateInContext(context).value_in_unit(distance.getUnit())
    value2 = np.sqrt(np.sum(((model.positions[atom1] - model.positions[atom2]) ** 2)))
    assert value1 == pytest.approx(value2)
    perform_common_tests(distance, context)


def test_angle():
    """
    Test whether an angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10]
    angle = cvlib.Angle(*atoms)
    model.system.addForce(angle)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = angle.evaluateInContext(context).value_in_unit(angle.getUnit())
    r21 = model.positions[atoms[0]] - model.positions[atoms[1]]
    r23 = model.positions[atoms[2]] - model.positions[atoms[1]]
    value2 = np.arccos(np.dot(r21, r23) / (np.linalg.norm(r21) * np.linalg.norm(r23)))
    assert value1 == pytest.approx(value2)
    perform_common_tests(angle, context)


def test_torsion():
    """
    Test whether a torsion angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10, 15]
    torsion = cvlib.Torsion(*atoms)
    model.system.addForce(torsion)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = torsion.evaluateInContext(context).value_in_unit(torsion.getUnit())
    r21 = model.positions[atoms[0]] - model.positions[atoms[1]]
    u23 = model.positions[atoms[2]] - model.positions[atoms[1]]
    u23 /= np.linalg.norm(u23)
    r34 = model.positions[atoms[3]] - model.positions[atoms[2]]
    value2 = np.arctan2(
        np.cross(r21, r34).dot(u23), r21.dot(r34) - r21.dot(u23) * r34.dot(u23)
    )
    assert value1 == pytest.approx(value2)
    perform_common_tests(torsion, context)


def test_radius_of_gyration():
    """
    Test whether a radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    rgsq = np.sum((positions - centroid) ** 2) / model.system.getNumParticles()
    rg_cv = cvlib.RadiusOfGyration(range(model.system.getNumParticles()))
    model.system.addForce(rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rgval = rg_cv.evaluateInContext(context).value_in_unit(unit.nanometers)
    assert rgval**2 == pytest.approx(rgsq)
    perform_common_tests(rg_cv, context)
