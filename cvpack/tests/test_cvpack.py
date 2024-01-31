"""
Unit and regression test for the cvpack package.
"""

import copy
import inspect
import io
import itertools as it
import sys
import typing as t

import mdtraj
import numpy as np
import openmm
import pytest
from openmm import app
from openmmtools import testsystems
from openmmtools.constants import ONE_4PI_EPS0
from scipy.spatial.transform import Rotation

import cvpack
from cvpack import serializer, unit


def test_cvpack_imported():
    """
    Sample test, will always pass so long as import statement worked.

    """
    assert "cvpack" in sys.modules


def test_effective_mass():
    """
    Test effective mass evaluation at a given context.

    """
    model = testsystems.AlanineDipeptideVacuum()
    rg_cv = cvpack.RadiusOfGyration(range(model.system.getNumParticles()))
    rg_cv.setUnusedForceGroup(0, model.system)
    model.system.addForce(rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    effective_mass = rg_cv.getEffectiveMass(context)
    assert effective_mass / effective_mass.unit == pytest.approx(30.946932)


def test_argument_inspection():
    """
    Test argument inspection of a arbitrary BaseCollectiveVariable subclass

    """

    # pylint: disable=missing-class-docstring, unused-argument
    class Test(cvpack.cvpack.BaseCollectiveVariable):
        def __init__(self, first: int, second: float, third: str = "3"):
            super().__init__(self)

    # pylint: enable=missing-class-docstring, unused-argument

    args, defaults = Test.getArguments()
    assert args["first"] is int
    assert args["second"] is float
    assert args["third"] is str
    assert defaults["third"] == "3"


def perform_common_tests(
    collectiveVariable: cvpack.cvpack.BaseCollectiveVariable,
    context: openmm.Context,
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
    serializer.serialize(collectiveVariable, pipe)
    pipe.seek(0)
    new_cv = serializer.deserialize(pipe)
    new_cv.setUnusedForceGroup(0, context.getSystem())
    context.getSystem().addForce(new_cv)
    context.reinitialize(preserveState=True)
    value1 = collectiveVariable.getValue(context)
    value2 = new_cv.getValue(context)
    assert value1 / value1.unit == value2 / value2.unit
    mass1 = collectiveVariable.getEffectiveMass(context)
    mass2 = new_cv.getEffectiveMass(context)
    assert mass1 / mass1.unit == mass2 / mass2.unit


def test_cv_is_in_context():
    """
    Test whether a collective variable is in a context.

    """
    model = testsystems.AlanineDipeptideVacuum()
    rg_cv = cvpack.RadiusOfGyration(range(model.system.getNumParticles()))
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    with pytest.raises(RuntimeError) as excinfo:
        rg_cv.getValue(context)
    assert str(excinfo.value) == "This force is not present in the given context."


def test_distance():
    """
    Test whether a distance is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atom1, atom2 = 0, 5
    distance = cvpack.Distance(atom1, atom2)
    distance.setUnusedForceGroup(0, model.system)
    model.system.addForce(distance)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = distance.getValue(context).value_in_unit(distance.getUnit())
    value2 = np.sqrt(np.sum(((model.positions[atom1] - model.positions[atom2]) ** 2)))
    assert value1 == pytest.approx(value2)
    perform_common_tests(distance, context)


def test_angle():
    """
    Test whether an angle is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    atoms = [0, 5, 10]
    angle = cvpack.Angle(*atoms)
    angle.setUnusedForceGroup(0, model.system)
    model.system.addForce(angle)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = angle.getValue(context).value_in_unit(angle.getUnit())
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
    torsion = cvpack.Torsion(*atoms)
    torsion.setUnusedForceGroup(0, model.system)
    model.system.addForce(torsion)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    value1 = torsion.getValue(context).value_in_unit(torsion.getUnit())
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
    masses = np.array(
        [unit.value_in_md_units(atom.element.mass) for atom in model.topology.atoms()]
    )
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    center_of_mass = np.einsum("i,ij->j", masses, positions) / np.sum(masses)
    num_atoms = model.system.getNumParticles()
    rgsq = np.sum((positions - centroid) ** 2) / num_atoms
    rg_cv = cvpack.RadiusOfGyration(range(num_atoms))
    weighted_rgsq = np.sum((positions - center_of_mass) ** 2) / num_atoms
    weighted_rg_cv = cvpack.RadiusOfGyration(range(num_atoms), weighByMass=True)
    rg_cv.setUnusedForceGroup(0, model.system)
    model.system.addForce(rg_cv)
    weighted_rg_cv.setUnusedForceGroup(0, model.system)
    model.system.addForce(weighted_rg_cv)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rgval = rg_cv.getValue(context).value_in_unit(unit.nanometers)
    assert rgval**2 == pytest.approx(rgsq)
    weighted_rgval = weighted_rg_cv.getValue(context).value_in_unit(unit.nanometers)
    assert weighted_rgval**2 == pytest.approx(weighted_rgsq)
    perform_common_tests(rg_cv, context)


def test_radius_of_gyration_squared():
    """
    Test whether a squared radius of gyration is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    masses = np.array(
        [unit.value_in_md_units(atom.element.mass) for atom in model.topology.atoms()]
    )
    positions = model.positions.value_in_unit(unit.nanometers)
    centroid = positions.mean(axis=0)
    center_of_mass = np.einsum("i,ij->j", masses, positions) / np.sum(masses)
    num_atoms = model.system.getNumParticles()
    rgsq = np.sum((positions - centroid) ** 2) / num_atoms
    rg_sq = cvpack.RadiusOfGyrationSq(range(num_atoms))
    weighted_rgsq = np.sum((positions - center_of_mass) ** 2) / num_atoms
    weighted_rg_sq = cvpack.RadiusOfGyrationSq(range(num_atoms), weighByMass=True)
    rg_sq.setUnusedForceGroup(0, model.system)
    model.system.addForce(rg_sq)
    weighted_rg_sq.setUnusedForceGroup(0, model.system)
    model.system.addForce(weighted_rg_sq)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    rg_sq_value = unit.value_in_md_units(rg_sq.getValue(context))
    assert rg_sq_value == pytest.approx(rgsq)
    weighted_rg_sq_value = unit.value_in_md_units(weighted_rg_sq.getValue(context))
    assert weighted_rg_sq_value == pytest.approx(weighted_rgsq)
    perform_common_tests(rg_sq, context)


def test_number_of_contacts():
    """
    Test whether a number of contacts is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    pos = model.positions
    group1 = [
        a.index for a in model.topology.atoms() if a.element == app.element.carbon
    ]
    group2 = [
        a.index for a in model.topology.atoms() if a.element == app.element.oxygen
    ]
    forces = {f.getName(): f for f in model.system.getForces()}
    nonbonded = forces["NonbondedForce"]
    exclusions = set()
    for index in range(nonbonded.getNumExceptions()):
        i, j, *_ = nonbonded.getExceptionParameters(index)
        exclusions.add((i, j) if i < j else (j, i))
    pairs = set()
    for i, j in it.product(group1, group2):
        if j != i:
            pair = (i, j) if i < j else (j, i)
            if pair not in exclusions:
                pairs.add(pair)
    distances = np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])
    contacts = np.where(distances <= 0.6, 1 / (1 + (distances / 0.3) ** 6), 0)
    number_of_contacts = cvpack.NumberOfContacts(
        group1,
        group2,
        forces["NonbondedForce"],
        switchFactor=None,
    )
    number_of_contacts.setUnusedForceGroup(0, model.system)
    model.system.addForce(number_of_contacts)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    nc_value = number_of_contacts.getValue(context)
    assert nc_value / nc_value.unit == pytest.approx(contacts.sum())
    perform_common_tests(number_of_contacts, context)


def run_rmsd_test(
    coordinates: np.ndarray,
    group: t.Sequence[int],
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
    rmsd = cvpack.RMSD(
        dict(zip(group, group_ref)) if passGroupOnly else reference,
        group,
        num_atoms,
    )
    rmsd.setUnusedForceGroup(0, model.system)
    model.system.addForce(rmsd)
    integrator = openmm.VerletIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(coordinates)
    group_coords = coordinates[group, :] - coordinates[group, :].mean(axis=0)
    _, rssd = Rotation.align_vectors(group_coords, group_ref)
    rmsd_value = rmsd.getValue(context)
    assert rmsd_value / rmsd_value.unit == pytest.approx(rssd / np.sqrt(len(group)))


def test_rmsd():
    """
    Test whether an RMSD is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.topology.getNumAtoms()
    rmsd = cvpack.RMSD(model.positions, np.arange(num_atoms), num_atoms)
    rmsd.setUnusedForceGroup(0, model.system)
    model.system.addForce(rmsd)
    integrator = openmm.VerletIntegrator(2 * unit.femtosecond)
    integrator.setIntegrationForceGroups({0})
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    context.getIntegrator().step(10000)
    state = context.getState(  # pylint: disable=unexpected-keyword-arg
        getPositions=True
    )
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    for pass_vec3 in [False, True]:
        for pass_group_only in [False, True]:
            group = np.arange(num_atoms)
            run_rmsd_test(coordinates, group, pass_group_only, pass_vec3)
            run_rmsd_test(
                coordinates, group[: num_atoms // 2], pass_group_only, pass_vec3
            )
            np.random.shuffle(group)
            run_rmsd_test(
                coordinates, group[: num_atoms // 2], pass_group_only, pass_vec3
            )
    perform_common_tests(rmsd, context)


def test_composite_rmsd():
    """
    Test whether a CompositeRMSD is computed correctly.

    """
    model = testsystems.HostGuestExplicit()
    host_atoms, guest_atoms = (
        [a.index for a in r.atoms()] for r in it.islice(model.topology.residues(), 0, 2)
    )
    try:
        crmsd1 = cvpack.CompositeRMSD(model.positions, [host_atoms, guest_atoms])
        atoms = host_atoms + guest_atoms
        crmsd2 = cvpack.CompositeRMSD(
            dict(zip(atoms, model.positions[atoms])),
            [host_atoms, guest_atoms],
            numAtoms=model.system.getNumParticles(),
        )
    except ImportError:
        pytest.skip("openmm-cpp-forces is not installed")
    crmsd1.setUnusedForceGroup(0, model.system)
    crmsd2.setUnusedForceGroup(1, model.system)
    model.system.addForce(crmsd1)
    model.system.addForce(crmsd2)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    perform_common_tests(crmsd1, context)
    perform_common_tests(crmsd2, context)


def test_helix_torsion_content():
    """
    Test whether a helix ramachandran content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    _, phi = mdtraj.compute_phi(traj)
    _, psi = mdtraj.compute_psi(traj)
    x = (np.rad2deg(phi.ravel()[:-1]) + 63.8) / 25
    y = (np.rad2deg(psi.ravel()[1:]) + 41.1) / 25
    computed_value = np.sum(1 / (1 + x**6) + 1 / (1 + y**6)) / 2

    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixTorsionContent(residues)
    assert str(excinfo.value) == "Could not find atom N in residue TMP163"
    helix_content = cvpack.HelixTorsionContent(residues[0:-1])
    helix_content.setUnusedForceGroup(0, model.system)
    model.system.addForce(helix_content)
    integrator = openmm.VerletIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    cv_value = helix_content.getValue(context)

    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_angle_content():
    """
    Test whether a helix angle content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    alpha_carbons = traj.top.select("name CA")
    angle_atoms = np.array(
        [alpha_carbons[:-2], alpha_carbons[1:-1], alpha_carbons[2:]]
    ).T
    angles = mdtraj.compute_angles(traj, angle_atoms)
    x = (np.rad2deg(angles.ravel()) - 88) / 15
    computed_value = np.sum(1 / (1 + x**6))

    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixAngleContent(residues)
    assert str(excinfo.value) == "Could not find atom CA in residue TMP163"
    helix_content = cvpack.HelixAngleContent(residues[0:-1])
    helix_content.setUnusedForceGroup(0, model.system)
    model.system.addForce(helix_content)
    integrator = openmm.VerletIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    cv_value = helix_content.getValue(context)
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_hbond_content():
    """
    Test whether a helix hydrogen-bond content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    hydrogens = traj.top.select("resSeq 59 to 79 and name H")
    oxygens = traj.top.select("resSeq 59 to 79 and name O")
    distances = mdtraj.compute_distances(
        traj, np.array([hydrogens[4:], oxygens[:-4]]).T
    )
    x = distances.ravel() / 0.33
    computed_value = np.sum(1 / (1 + x**6))

    residues = list(model.topology.residues())
    with pytest.raises(ValueError):
        helix_content = cvpack.HelixHBondContent(residues)
    helix_content = cvpack.HelixHBondContent(residues[58:79])
    helix_content.setUnusedForceGroup(0, model.system)
    model.system.addForce(helix_content)
    integrator = openmm.VerletIntegrator(0)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    cv_value = helix_content.getValue(context)
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_rmsd_content():
    """
    Test whether a helix rmsd content is computed correctly.

    """
    model = testsystems.LysozymeImplicit()

    num_atoms = model.topology.getNumAtoms()
    residues = list(model.topology.residues())
    with pytest.raises(ValueError) as excinfo:
        helix_content = cvpack.HelixRMSDContent(
            residues[len(residues) - 37 :], num_atoms
        )
    assert str(excinfo.value) == "Atom N not found in residue TMP163"

    start, end = 39, 51
    helix_content = cvpack.HelixRMSDContent(residues[start:end], num_atoms)
    helix_content.setUnusedForceGroup(0, model.system)
    model.system.addForce(helix_content)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    cv_value = helix_content.getValue(context)

    traj = mdtraj.Trajectory(
        model.positions, mdtraj.Topology.from_openmm(model.topology)
    )
    ref = copy.deepcopy(traj)
    positions = cvpack.helix_rmsd_content.ALPHA_POSITIONS
    # pylint: enable=protected-access

    def compute_cv_value(atoms):
        computed_value = 0
        for i in range(len(atoms) // 5 - 5):
            group = atoms[5 * i : 5 * i + 30]
            ref.xyz[:, group, :] = positions
            x4 = (mdtraj.rmsd(traj, ref, 0, group).item() / 0.08) ** 4
            computed_value += (1 + x4) / (1 + x4 + x4**2)
        return computed_value

    def select_atoms(stard, end):
        atoms = ()
        for residue in residues[stard:end]:
            residue_atoms = {atom.name: atom.index for atom in residue.atoms()}
            if residue.name == "GLY":
                residue_atoms["CB"] = residue_atoms["HA2"]
            atoms += tuple(residue_atoms[name] for name in ["N", "CA", "CB", "C", "O"])
        return atoms

    computed_value = compute_cv_value(select_atoms(start, end))
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(helix_content, context)


def test_helix_torsion_similarity():
    """
    Test whether a torsion similarity CV is computed correctly.

    """
    model = testsystems.LysozymeImplicit()
    positions = model.positions.value_in_unit(unit.nanometers)
    traj = mdtraj.Trajectory(positions, mdtraj.Topology.from_openmm(model.topology))
    phi_atoms, phi = mdtraj.compute_phi(traj)
    psi_atoms, psi = mdtraj.compute_psi(traj)
    torsion_similarity = cvpack.TorsionSimilarity(
        np.vstack([phi_atoms[1:], psi_atoms[1:]]),
        np.vstack([phi_atoms[:-1], psi_atoms[:-1]]),
    )
    torsion_similarity.setUnusedForceGroup(0, model.system)
    model.system.addForce(torsion_similarity)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    cv_value = torsion_similarity.getValue(context)
    phi = phi.ravel()
    psi = psi.ravel()
    deltas = np.hstack([phi[1:], psi[1:]]) - np.hstack([phi[:-1], psi[:-1]])
    deltas = np.array([min(delta, 2 * np.pi - delta) for delta in deltas])
    assert cv_value / cv_value.unit == pytest.approx(np.sum(0.5 * (1 + np.cos(deltas))))
    perform_common_tests(torsion_similarity, context)


def test_sheet_rmsd_content():
    """
    Test whether a sheet rmsd content is computed correctly.

    """
    model = testsystems.SrcImplicit()
    positions = np.vstack(
        [np.array(pos) for pos in unit.value_in_md_units(model.positions)]
    )
    topology = mdtraj.Topology.from_openmm(model.topology)
    residues = list(it.islice(model.topology.residues(), 68, 82))

    sheet_content = cvpack.SheetRMSDContent(residues, topology.n_atoms)
    sheet_content.setForceGroup(31)
    sheet_content.setUnusedForceGroup(0, model.system)
    model.system.addForce(sheet_content)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)
    cv_value = sheet_content.getValue(context)

    traj = mdtraj.Trajectory(positions, topology)
    ref = copy.deepcopy(traj)

    def compute_step_function(indices):
        group = sum((residue_atoms[i] for i in indices), ())
        ref.xyz[:, group, :] = cvpack.sheet_rmsd_content.ANTIBETA_POSITIONS
        x4 = (mdtraj.rmsd(traj, ref, 0, group).item() / 0.08) ** 4
        return (1 + x4) / (1 + x4 + x4**2)

    residue_atoms = []
    for residue in residues:
        atoms = {atom.name: atom.index for atom in residue.atoms()}
        if residue.name == "GLY":
            atoms["CB"] = atoms["HA2"]
        residue_atoms.append(tuple(atoms[name] for name in ["N", "CA", "CB", "C", "O"]))

    min_separation = 5
    computed_value = sum(
        compute_step_function([i, i + 1, i + 2, j, j + 1, j + 2])
        for i, j in it.combinations(range(len(residues) - 2), 2)
        if j - i >= min_separation
    )

    assert cv_value / cv_value.unit == pytest.approx(computed_value, abs=1e-5)
    perform_common_tests(sheet_content, context)


def test_atomic_function():
    """
    Test whether an atomic-function CV is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.system.getNumParticles()
    atoms = np.arange(num_atoms)
    np.random.shuffle(atoms)
    function = "+".join(f"distance(p{i+1}, p{i+2})" for i in range(num_atoms - 1))
    with pytest.raises(ValueError) as excinfo:
        colvar = cvpack.AtomicFunction(function, unit.angstrom, atoms)
    assert (
        str(excinfo.value) == "Unit angstrom is not compatible with the MD unit system."
    )
    colvar = cvpack.AtomicFunction(function, unit.nanometers, atoms)
    colvar.setUnusedForceGroup(0, model.system)
    model.system.addForce(colvar)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    cv_value = colvar.getValue(context)
    positions = model.positions.value_in_unit(unit.nanometers)
    computed_value = np.sum(
        [
            np.linalg.norm(positions[atoms[i + 1]] - positions[atoms[i]])
            for i in range(num_atoms - 1)
        ]
    )
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(colvar, context)


def test_centroid_function():
    """
    Test whether a centroid-function CV is computed correctly.

    """
    model = testsystems.AlanineDipeptideVacuum()
    num_atoms = model.system.getNumParticles()
    atoms = np.arange(num_atoms)
    np.random.shuffle(atoms)
    num_groups = num_atoms // 3
    groups = np.reshape(atoms[: 3 * num_groups], (num_groups, 3))
    function = "+".join(f"distance(g{i+1}, g{i+2})" for i in range(num_groups - 1))
    with pytest.raises(ValueError) as excinfo:
        colvar = cvpack.CentroidFunction(function, unit.angstrom, groups)
    assert (
        str(excinfo.value) == "Unit angstrom is not compatible with the MD unit system."
    )
    colvar = cvpack.CentroidFunction(
        function, unit.nanometers, groups, weighByMass=False
    )
    colvar.setUnusedForceGroup(0, model.system)
    model.system.addForce(colvar)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    cv_value = colvar.getValue(context)
    positions = model.positions.value_in_unit(unit.nanometers)
    computed_value = np.sum(
        [
            np.linalg.norm(
                np.mean(positions[groups[i + 1]], axis=0)
                - np.mean(positions[groups[i]], axis=0)
            )
            for i in range(num_groups - 1)
        ]
    )
    assert cv_value / cv_value.unit == pytest.approx(computed_value)
    perform_common_tests(colvar, context)


def test_attraction_strength():
    """
    Test whether an attraction strength CV is computed correctly.

    """
    num_particles = 20
    group1 = list(range(num_particles // 2))
    group2 = list(range(num_particles // 2, num_particles))
    charges = 1.0 - 2.0 * np.random.rand(num_particles)
    sigmas = np.random.rand(num_particles)
    epsilons = np.random.rand(num_particles)
    positions = np.random.rand(num_particles, 3)
    cutoff = 1.0
    strength = 0.0
    for i in group1:
        for j in group2:
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij <= cutoff:
                x = np.abs((2 * rij / (sigmas[i] + sigmas[j])) ** 6 - 2.0) + 2.0
                strength -= 4 * np.sqrt(epsilons[i] * epsilons[j]) * (1 / x**2 - 1 / x)
                if charges[i] * charges[j] < 0.0:
                    x = rij / cutoff
                    strength -= (
                        ONE_4PI_EPS0
                        * charges[i]
                        * charges[j]
                        * (1 / x + (x**2 - 3) / 2)
                    )
    system = openmm.System()
    for i in range(num_particles):
        system.addParticle(1.0)
    force = openmm.NonbondedForce()
    for params in zip(charges, sigmas, epsilons):
        force.addParticle(*params)
    force.setUseSwitchingFunction(False)
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(cutoff)
    colvar = cvpack.AttractionStrength(group1, group2, force)
    system.addForce(colvar)
    platform = openmm.Platform.getPlatformByName("Reference")
    integreator = openmm.VerletIntegrator(0)
    context = openmm.Context(system, integreator, platform)
    context.setPositions(positions)
    assert unit.value_in_md_units(colvar.getValue(context)) == pytest.approx(strength)
    perform_common_tests(colvar, context)


def test_residue_coordination():
    """
    Test whether a residue coordination CV is computed correctly.

    """
    model = testsystems.LysozymeImplicit()
    positions = model.positions.value_in_unit(unit.nanometers)
    groups = [
        list(it.islice(model.topology.residues(), start, end))
        for start, end in [(125, 142), (142, 156)]
    ]
    centroids = [
        np.vstack(
            [np.mean(positions[[a.index for a in r.atoms()]], axis=0) for r in group]
        )
        for group in groups
    ]
    distances = np.linalg.norm(centroids[0][:, None, :] - centroids[1], axis=2)
    computed_cv = np.sum(1 / (1 + distances**6))

    res_coord = cvpack.ResidueCoordination(*groups, pbc=False, weighByMass=False)
    res_coord.setUnusedForceGroup(0, model.system)
    model.system.addForce(res_coord)
    context = openmm.Context(
        model.system,
        openmm.VerletIntegrator(0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)
    cv_value = res_coord.getValue(context)

    assert cv_value / cv_value.unit == pytest.approx(computed_cv)
    perform_common_tests(res_coord, context)
