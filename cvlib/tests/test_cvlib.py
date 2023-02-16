"""
Unit and regression test for the cvlib package.
"""

import inspect
import io
import itertools
import sys
from typing import List

import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems
from scipy.spatial.transform import Rotation

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


def test_argument_inspection():
    """
    Test argument inspection of a arbitrary AbstractCollectiveVariable subclass

    """
    # pylint: disable=missing-class-docstring, unused-argument
    class Test(cvlib.cvlib.AbstractCollectiveVariable):
        def __init__(self, first: int, second: float, third: str = "3"):
            super().__init__(self)

    # pylint: enable=missing-class-docstring, unused-argument

    args, defaults = Test.getArguments()
    assert args["first"] is int
    assert args["second"] is float
    assert args["third"] is str
    assert defaults["third"] == "3"


def perform_common_tests(
    collectiveVariable: cvlib.cvlib.AbstractCollectiveVariable, context: openmm.Context
) -> None:
    """
    Function to be called in every individual cv test.

    """
    # Default name must match the class name
    assert collectiveVariable.getName() == collectiveVariable.__class__.__name__

    # Unit must conform to the default OpenMM system
    unity = 1 * collectiveVariable.getUnit()
    assert unity.value_in_unit_system(unit.md_unit_system) == 1

    # Class must have full type annotation (except for argument `self`)
    args, _ = collectiveVariable.getArguments()
    for _, annotation in args.items():
        assert annotation is not inspect.Parameter.empty

    # Test serialization/deserialization
    pipe = io.StringIO()
    cvlib.serializer.serialize(collectiveVariable, pipe)
    pipe.seek(0)
    new_cv = cvlib.serializer.deserialize(pipe)
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
    value2 = np.arctan2(np.cross(r21, r34).dot(u23), r21.dot(r34) - r21.dot(u23) * r34.dot(u23))
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


def test_number_of_contacts():
    """
    Test whether a number of contacts is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    pos = model.positions
    group1 = [a.index for a in model.topology.atoms() if a.element == app.element.carbon]
    group2 = [a.index for a in model.topology.atoms() if a.element == app.element.oxygen]
    pairs = set()
    for i, j in itertools.product(group1, group2):
        if j != i and (j, i) not in pairs:
            pairs.add((i, j))
    threshold = 0.3
    contacts = [np.linalg.norm(pos[i] - pos[j]) <= threshold for i, j in pairs]
    num_atoms = model.topology.getNumAtoms()
    number_of_contacts = cvlib.NumberOfContacts(
        group1, group2, num_atoms, stepFunction="step(1-x)", thresholdDistance=threshold
    )
    model.system.addForce(number_of_contacts)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    nc_value = number_of_contacts.evaluateInContext(context)
    assert nc_value / nc_value.unit == pytest.approx(sum(contacts))
    perform_common_tests(number_of_contacts, context)


def run_rmsd_test(
    coordinates: np.ndarray,
    group: List[int],
    passGroupOnly: bool,
    passVec3: bool,
) -> None:
    """
    Performs a specific RMSD test

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.topology.getNumAtoms()
    reference = np.array(model.positions.value_in_unit(unit.nanometers))
    group_ref = reference[group, :] - reference[group, :].mean(axis=0)
    if passVec3:
        reference = [openmm.Vec3(*row) for row in reference]
    rmsd = cvlib.RootMeanSquareDeviation(
        group_ref if passGroupOnly else reference,
        group,
        num_atoms,
    )
    model.system.addForce(rmsd)
    integrator = openmm.VerletIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(coordinates)
    group_coords = coordinates[group, :] - coordinates[group, :].mean(axis=0)
    _, rssd = Rotation.align_vectors(group_coords, group_ref)
    rmsd_value = rmsd.evaluateInContext(context)
    assert rmsd_value / rmsd_value.unit == pytest.approx(rssd / np.sqrt(len(group)))


def test_root_mean_square_deviation():
    """
    Test whether an RMSD is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.topology.getNumAtoms()
    rmsd = cvlib.RootMeanSquareDeviation(model.positions, np.arange(num_atoms), num_atoms)
    model.system.addForce(rmsd)
    integrator = openmm.VerletIntegrator(2 * unit.femtosecond)
    integrator.setIntegrationForceGroups({0})
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    context.getIntegrator().step(10000)
    state = context.getState(getPositions=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    for pass_vec3 in [False, True]:
        for pass_group_only in [False, True]:
            group = np.arange(num_atoms)
            run_rmsd_test(coordinates, group, pass_group_only, pass_vec3)
            run_rmsd_test(coordinates, group[:num_atoms//2], pass_group_only, pass_vec3)
            np.random.shuffle(group)
            run_rmsd_test(coordinates, group[:num_atoms//2], pass_group_only, pass_vec3)
    perform_common_tests(rmsd, context)
