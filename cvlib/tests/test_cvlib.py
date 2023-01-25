"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import os
import sys

import openmm
import pytest

from openmm import app, unit

import cvlib


def test_cvlib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cvlib" in sys.modules


def alanineDipeptideModel(force_field='amber03', water=None, box_length=25*unit.angstroms,
                          constraints=openmm.app.HBonds, rigidWater=True):
    pdb = app.PDBFile(os.path.join(cvlib.__path__[0], 'data', 'alanine-dipeptide.pdb'))
    if water is None:
        force_field = app.ForceField(f'{force_field}.xml')
        topology = pdb.topology
        positions = pdb.positions
        L = box_length.value_in_unit(unit.nanometers)
        vectors = [openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)]
        topology.setPeriodicBoxVectors(vectors)
    else:
        force_field = app.ForceField(f'{force_field}.xml', f'{water}.xml')
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(force_field, model=water, boxSize=box_length*openmm.Vec3(1, 1, 1))
        topology = modeller.topology
        positions = modeller.positions
    system = force_field.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff if water is None else app.PME,
        constraints=constraints,
        rigidWater=rigidWater,
        removeCMMotion=False,
    )
    return system, topology, positions

def test_radius_of_gyration():
    system, topology, positions = alanineDipeptideModel()

    R = positions._value
    N = len(R)
    Rmean = sum(R, openmm.Vec3(0, 0, 0))/N
    RgSqVal = 0.0
    for r in R:
        dr = r - Rmean
        RgSqVal += dr.x**2 + dr.y**2 + dr.z**2
    RgSqVal /= N

    Rg = cvlib.RadiusOfGyration(range(topology._numAtoms))
    Rg.setForceGroup(1)
    system.addForce(Rg)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    Rg = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    assert Rg*Rg == pytest.approx(RgSqVal)
