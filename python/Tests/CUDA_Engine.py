# -*- coding: utf-8 -*-
"""
 CUDA engine regression test -- comparison with the CPU engine

 The test cases are SeanMollet's python/Tests/GPU_Engine.py from his openEMS
 GPU-engine pull request (thliebig/openEMS#235), (c) 2026 Sean Mollet, GPL;
 the driver below is rewritten for this fork's CUDA engine.

 Every case runs in its own process with
   cpu         the default (multithreaded) CPU engine          -- reference
   cuda        engine='cuda', one GPU, default code paths
   cuda-legacy engine='cuda', one GPU, OPENEMS_CUDA_LEGACY_KERNELS=1,
               OPENEMS_GPU_DFT=0 (flat kernels, separate UPML kernels, host DFT)
   cuda-mgpu   engine='cuda', OPENEMS_CUDA_VIRTUAL=1 OPENEMS_CUDA_GPUS=3
               (three slabs on one device; models the multi-GPU engine does
               not support run single-GPU, which is fine to compare too)

 Pass criteria (per case)
   cuda vs cpu:          every probe and HDF5 dump within 1e-4 of its peak
   cuda vs cuda-legacy:  bit-identical (the fast paths change no arithmetic)
   cuda vs cuda-mgpu:    bit-identical (dumps included: both engines gather them
                         on the device with the same stencils)
 A case whose model the CUDA engine refuses (setup error code 4: an extension
 without a CUDA implementation) must be refused by every CUDA run and is
 reported as such; cylindrical meshes always run on the CPU operator.

 usage: python3 CUDA_Engine.py [case ...]
"""

import os, re, sys, glob, json, tempfile, subprocess, shutil
import numpy as np
import h5py

from CSXCAD  import ContinuousStructure
from CSXCAD.CSProperties import CSPropLorentzMaterial, CSPropDebyeMaterial
from CSXCAD.CSProperties import ABCtype
from openEMS.physical_constants import C0
from openEMS import openEMS
from openEMS.ports import LumpedPort

unit = 1e-3   # drawing unit: mm


def channel_1d(FDTD, CSX, sinus=False):
    """ TEM channel (PEC walls normal to x, PMC walls normal to y, PML at the z-ends) """
    if sinus:
        FDTD.SetSinusExcite(3e9)
    else:
        FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PMC', 'PMC', 'PML_8', 'PML_8'])
    mesh = CSX.GetGrid()
    mesh.AddLine('x', [0, 0.5, 1])
    mesh.AddLine('y', [0, 0.5, 1])
    mesh.AddLine('z', np.arange(0, 100.5, 0.5))
    CSX.AddExcitation('plane', exc_type=0, exc_val=[1, 0, 0]).AddBox([0, 0, 20], [1, 1, 20])
    CSX.AddProbe('et', p_type=2).AddPoint([0.5, 0.5, 70])
    CSX.AddProbe('ht', p_type=3).AddPoint([0.5, 0.5, 70])


def case_dispersive_pml():
    FDTD = openEMS(NrTS=5000, EndCriteria=1e-5)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    CSX.GetGrid().SetDeltaUnit(unit)
    channel_1d(FDTD, CSX)
    drude = CSPropLorentzMaterial(CSX.GetParameterSet(), order=1)
    drude.SetName('drude')
    drude.SetDispersiveMaterialProperty(0, eps_plasma=5e9, eps_relax=1e-9)
    CSX.AddProperty(drude)
    drude.AddBox([0, 0, 40], [1, 1, 50], priority=10)
    return FDTD, CSX


def case_3d_mixed():
    FDTD = openEMS(NrTS=800, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-10, 10.5, 1))

    # lumped port and series RLC element connected by two wires
    LumpedPort(CSX, 1, 50, [-4, 0, 0], [-4, 0, 2], 'z', excite=1)
    rlc = CSX.AddLumpedElement('rlc', ny='z', caps=False, R=10, L=1e-9, C=1e-12, LEtype=1)
    rlc.AddBox([4, 0, 0], [4, 0, 2], priority=10)
    wire = CSX.AddMetal('wire')
    wire.AddCurve([[-4, 4], [0, 0], [0, 0]])
    wire.AddCurve([[-4, 4], [0, 0], [2, 2]])

    # conducting sheet patch
    sheet = CSX.AddConductingSheet('sheet', conductivity=5.8e7, thickness=1e-6)
    sheet.AddBox([-3, -3, 5], [3, 3, 5], priority=10)

    # TF/SF plane wave
    pw = CSX.AddExcitation('plane_wave', exc_type=10, exc_val=[0, 0, 1])
    pw.SetPropagationDir([1, 0, 0])
    pw.SetFrequency(5e9)
    pw.AddBox([-6, -6, -6], [6, 6, 6])

    CSX.AddProbe('et', p_type=2).AddPoint([0, 5, 0])
    CSX.AddProbe('ht', p_type=3).AddPoint([0, 5, 0])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([-10, -10, 0], [10, 10, 0])
    CSX.AddDump('Hf', dump_type=11, file_type=1, frequency=[5e9]).AddBox([-10, 0, -10], [10, 0, 10])
    return FDTD, CSX


def case_dumps():
    """ E and H dumps in all interpolation modes, time and frequency domain, on a non-uniform mesh """
    FDTD = openEMS(NrTS=300, EndCriteria=0)
    FDTD.SetGaussExcite(5e9, 4e9)
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PEC', 'PMC', 'MUR', 'PML_8'])
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', np.concatenate([np.linspace(-30, -5, 12), np.linspace(-4, 4, 17), np.linspace(5, 30, 9)]))
    mesh.AddLine('y', np.linspace(-25, 25, 31)**3/625.0 + np.linspace(-25, 25, 31)*0.5)
    mesh.AddLine('z', np.linspace(-20, 20, 27))
    CSX.AddMaterial('diel', epsilon=4, kappa=0.01).AddBox([-10, -8, -5], [6, 9, 7])
    CSX.AddMetal('pec').AddBox([8, -5, -3], [9, 5, 3])
    CSX.AddExcitation('d', exc_type=0, exc_val=[0, 0, 1]).AddBox([0, 0, -2], [0, 0, 2])
    CSX.AddProbe('et', p_type=2).AddPoint([5, 5, 5])
    for mode in (0, 1, 2):
        for t in (0, 1):
            CSX.AddDump(f'td_{t}_{mode}', dump_type=t, dump_mode=mode, file_type=1).AddBox([-30, -25, -20], [30, 25, 20])
            CSX.AddDump(f'fd_{t}_{mode}', dump_type=10+t, dump_mode=mode, file_type=1, frequency=[3e9, 6e9]).AddBox([-12, -25, -20], [12, 25, 20])
    return FDTD, CSX


def case_dumps_snapshot():
    """ NF2FF box (TD dumps of E and H) and FD dumps on a grid large enough for field snapshots (>= 512k nodes) """
    FDTD = openEMS(NrTS=400, EndCriteria=0)
    FDTD.SetGaussExcite(5e9, 4e9)
    FDTD.SetBoundaryCond(['PML_8']*6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', np.linspace(-50, 50, 100))
    mesh.AddLine('y', np.linspace(-40, 40, 80))
    mesh.AddLine('z', np.linspace(-36, 36, 72))
    CSX.AddMaterial('diel', epsilon=4, kappa=0.01).AddBox([-10, -8, -5], [6, 9, 7])
    CSX.AddExcitation('d', exc_type=0, exc_val=[0, 0, 1]).AddBox([0, 0, -2], [0, 0, 2])
    CSX.AddProbe('et', p_type=2).AddPoint([5, 5, 5])
    FDTD.CreateNF2FFBox()
    CSX.AddDump('td_cell', dump_type=0, dump_mode=2, file_type=1).AddBox([-20, -20, -10], [20, 20, 10])
    for t in (0, 1):
        CSX.AddDump(f'fd_{t}', dump_type=10+t, dump_mode=1, file_type=1, frequency=[3e9, 6e9]).AddBox([-30, -30, -5], [30, 30, 5])
    return FDTD, CSX


def case_excitation():
    """ PEC cavity: only the excitation extension, with overlapping sources """
    FDTD = openEMS(NrTS=600, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['PEC'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-10, 10.5, 1))
    CSX.AddExcitation('e_soft', exc_type=0, exc_val=[1, 0, 0]).AddBox([0, -2, 2], [4, 2, 2])
    CSX.AddExcitation('e_soft2', exc_type=0, exc_val=[0.5, 0, 0], delay=0.1e-9).AddBox([2, 0, 2], [6, 0, 2])  # shares edges
    CSX.AddExcitation('h_soft', exc_type=2, exc_val=[0, 0, 1]).AddBox([2, 2, -4], [4, 4, -2])
    CSX.AddProbe('et', p_type=2).AddPoint([5, 5, 5])
    CSX.AddProbe('ht', p_type=3).AddPoint([-5, 5, -5])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([-10, -10, 0], [10, 10, 0])
    return FDTD, CSX


def case_pml():
    """ free space with PML on all sides: excitation and UPML extensions """
    FDTD = openEMS(NrTS=700, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['PML_8'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-15, 15.5, 1))
    CSX.AddExcitation('dipole', exc_type=0, exc_val=[0, 0, 1]).AddBox([0, 0, -1], [0, 0, 1])
    CSX.AddProbe('et', p_type=2).AddPoint([4, 3, 2])
    CSX.AddProbe('ht', p_type=3).AddPoint([-3, 5, 0])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([-15, -15, 0], [15, 15, 0])
    return FDTD, CSX


def case_mur():
    """ free space with Mur ABC on all sides and a source on a boundary plane: excitation and Mur extensions """
    FDTD = openEMS(NrTS=700, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-10, 10.5, 1))
    CSX.AddExcitation('dipole', exc_type=0, exc_val=[0, 0, 1]).AddBox([0, 0, -1], [0, 0, 1])
    # a source on the x-max plane delays that Mur ABC until the excitation is done
    CSX.AddExcitation('wall', exc_type=0, exc_val=[0, 1, 0]).AddBox([10, -2, 0], [10, 2, 0])
    CSX.AddProbe('et', p_type=2).AddPoint([4, 3, 2])
    CSX.AddProbe('ht', p_type=3).AddPoint([-3, 5, 0])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([-10, -10, 0], [10, 10, 0])
    return FDTD, CSX


def case_materials():
    """ Drude, Lorentz and Debye materials, a magnetic Drude material and a conducting sheet in a PML channel """
    FDTD = openEMS(NrTS=3000, EndCriteria=0)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    CSX.GetGrid().SetDeltaUnit(unit)
    channel_1d(FDTD, CSX)
    def lorentz(name, z0, z1, **kw):
        m = CSPropLorentzMaterial(CSX.GetParameterSet(), order=1)
        m.SetName(name)
        m.SetDispersiveMaterialProperty(0, **kw)
        CSX.AddProperty(m)
        m.AddBox([0, 0, z0], [1, 1, z1], priority=10)
    lorentz('drude', 25, 30, eps_plasma=5e9, eps_relax=1e-9)
    lorentz('lorentz', 32, 37, eps_plasma=4e9, eps_pole_freq=3e9, eps_relax=1e-9)
    lorentz('double_drude', 39, 44, eps_plasma=5e9, eps_relax=1e-8, mue_plasma=5e9, mue_relax=1e-8)
    debye = CSPropDebyeMaterial(CSX.GetParameterSet(), order=2, epsilon=4)
    debye.SetName('debye')
    debye.SetDispersiveMaterialProperty(0, eps_delta=1, eps_relax=4e-11)
    debye.SetDispersiveMaterialProperty(1, eps_delta=0.5, eps_relax=1e-10)
    CSX.AddProperty(debye)
    debye.AddBox([0, 0, 46], [1, 1, 51], priority=10)
    CSX.AddConductingSheet('sheet', conductivity=1e5, thickness=10e-6).AddBox([0, 0, 55], [1, 1, 55], priority=10)
    return FDTD, CSX


def case_lumped(bc='MUR', size=10):
    """ lumped port with series and parallel RLC elements in a Mur box: excitation, lumped RLC and Mur extensions """
    FDTD = openEMS(NrTS=1500, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond([bc] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-size, size + 0.5, 1))
    LumpedPort(CSX, 1, 50, [-4, 0, 0], [-4, 0, 2], 'z', excite=1)
    ser = CSX.AddLumpedElement('ser_rlc', ny='z', caps=False, R=10, L=1e-9, C=1e-12, LEtype=1)
    ser.AddBox([0, 0, 0], [0, 0, 2], priority=10)
    par = CSX.AddLumpedElement('par_rlc', ny='z', caps=False, R=200, L=2e-9, C=0.5e-12, LEtype=0)
    par.AddBox([4, 0, 0], [4, 0, 2], priority=10)
    wire = CSX.AddMetal('wire')
    wire.AddCurve([[-4, 4], [0, 0], [0, 0]])
    wire.AddCurve([[-4, 4], [0, 0], [2, 2]])
    CSX.AddProbe('et', p_type=2).AddPoint([0, 5, 0])
    return FDTD, CSX


def case_lumped_pml():
    """ the lumped RLC elements with PML: the voltages they change in the fused device step """
    return case_lumped('PML_8', 20)


def case_tfsf():
    """ oblique plane wave on a PEC sphere in a PML box: excitation, TF/SF and UPML extensions """
    FDTD = openEMS(NrTS=700, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['PML_8'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    for ax in 'xyz':
        mesh.AddLine(ax, np.arange(-15, 15.5, 1))
    k_dir = np.array([1, 2, 3]) / np.sqrt(14)
    pw = CSX.AddExcitation('plane_wave', exc_type=10, exc_val=[2, -1, 0])
    pw.SetPropagationDir(k_dir)
    pw.SetFrequency(5e9)
    pw.AddBox([-6, -6, -6], [6, 6, 6])
    CSX.AddMetal('sphere').AddSphere(priority=10, center=[0, 0, 0], radius=3)
    CSX.AddProbe('et_in', p_type=2).AddPoint([4, -3, 2])
    CSX.AddProbe('et_out', p_type=2).AddPoint([-10, 1, 3])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([-15, -15, 0], [15, 15, 0])
    return FDTD, CSX


def case_absorbers():
    """ PEC-terminated channel with local absorbing sheets (Mur and Mur with super-absorption) """
    FDTD = openEMS(NrTS=3000, EndCriteria=0)
    FDTD.SetGaussExcite(5.5e9, 4.5e9)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PMC', 'PMC', 'PEC', 'PEC'])
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [0, 0.5, 1])
    mesh.AddLine('y', [0, 0.5, 1])
    mesh.AddLine('z', np.arange(0, 100.5, 0.5))
    CSX.AddExcitation('plane', exc_type=0, exc_val=[1, 0, 0]).AddBox([0, 0, 40], [1, 1, 40])
    CSX.AddAbsorbingBC('abs_low', NormalSignPositive=False, AbsorbingBoundaryType=ABCtype.MUR_1ST,
                       PhaseVelocity=C0).AddBox([0, 0, 5], [1, 1, 5], priority=6)
    CSX.AddAbsorbingBC('abs_high', NormalSignPositive=True, AbsorbingBoundaryType=ABCtype.MUR_1ST_SA,
                       PhaseVelocity=C0).AddBox([0, 0, 95], [1, 1, 95], priority=6)
    CSX.AddProbe('et', p_type=2).AddPoint([0.5, 0.5, 70])
    CSX.AddProbe('ht', p_type=3).AddPoint([0.5, 0.5, 20])
    return FDTD, CSX


def cylinder_mesh(FDTD, alpha, r0, r1, z_pad=0):
    """ cylindrical mesh, the CPU reference is the cylindrical engine (engine='basic' has no effect);
        z_pad extends z beyond 0..30 on both sides, e.g. to make room for a PML """
    CSX = ContinuousStructure(CoordSystem=1)
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('r', np.arange(r0, r1 + 1, 2))
    mesh.AddLine('a', alpha)
    mesh.AddLine('z', np.arange(-z_pad, 30 + z_pad + 0.5, 2))
    return CSX


def case_cylinder_closed():
    """ closed cylindrical mesh including r=0: excitation and cylinder extensions """
    FDTD = openEMS(CoordSystem=1, NrTS=1500, EndCriteria=0)
    FDTD.SetGaussExcite(3e9, 2e9)
    FDTD.SetBoundaryCond(['PEC'] * 6)
    CSX = cylinder_mesh(FDTD, (np.arange(25) - 12) * 2*np.pi/24, 0, 40)
    CSX.AddExcitation('line', exc_type=0, exc_val=[0, 0, 1]).AddBox([14, 0, 0], [14, 0, 30])
    CSX.AddExcitation('radial', exc_type=0, exc_val=[1, 0, 0]).AddBox([6, np.pi/2, 10], [10, np.pi/2, 10])
    CSX.AddProbe('et_axis', p_type=2).AddPoint([0, 0, 16])
    CSX.AddProbe('et', p_type=2).AddPoint([20, np.pi/4, 14])
    CSX.AddProbe('ht', p_type=3).AddPoint([10, -np.pi/3, 8])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([0, -np.pi, 14], [40, np.pi, 14])
    return FDTD, CSX


def case_cylinder_wedge():
    """ open alpha wedge with r>0 and PML in z: excitation, UPML and (inactive) cylinder extensions """
    FDTD = openEMS(CoordSystem=1, NrTS=1000, EndCriteria=0)
    FDTD.SetGaussExcite(3e9, 2e9)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PML_8', 'PML_8'])
    CSX = cylinder_mesh(FDTD, np.linspace(-np.pi/4, np.pi/4, 13), 10, 40, z_pad=16)
    CSX.AddExcitation('coax', exc_type=0, exc_val=[1, 0, 0]).AddBox([10, -np.pi/4, 12], [40, np.pi/4, 12])
    CSX.AddProbe('et', p_type=2).AddPoint([20, 0, 20])
    CSX.AddProbe('ht', p_type=3).AddPoint([30, np.pi/8, 6])
    return FDTD, CSX


def multigrid(radii, alpha, r1=40, z_bc='PEC'):
    """ cylindrical multi-grid with sources and probes inside and outside the sub-grids """
    FDTD = openEMS(CoordSystem=1, NrTS=1200, EndCriteria=0, MultiGrid=radii)
    FDTD.SetGaussExcite(3e9, 2e9)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', z_bc, z_bc])
    CSX = cylinder_mesh(FDTD, alpha, 0, r1, z_pad=16 if z_bc.startswith('PML') else 0)
    CSX.AddExcitation('inner', exc_type=0, exc_val=[0, 0, 1]).AddBox([4, 0, 0], [4, 0, 30])
    CSX.AddExcitation('outer', exc_type=0, exc_val=[1, 0, 0]).AddBox([26, alpha[3], 12], [32, alpha[3], 12])
    CSX.AddProbe('et_axis', p_type=2).AddPoint([0, 0, 16])
    CSX.AddProbe('et_inner', p_type=2).AddPoint([6, alpha[len(alpha)//3], 14])
    CSX.AddProbe('ht_inner', p_type=3).AddPoint([8, alpha[len(alpha)//4], 8])
    CSX.AddProbe('et_outer', p_type=2).AddPoint([30, alpha[2*len(alpha)//3], 14])
    CSX.AddDump('Et', dump_type=0, file_type=1).AddBox([0, alpha[0], 14], [r1, alpha[-1], 14])
    return FDTD, CSX


def case_multigrid():
    """ closed cylindrical mesh with one multi-grid level """
    return multigrid([14], (np.arange(49) - 24) * 2*np.pi/48)


def case_multigrid2():
    """ closed cylindrical mesh with two nested multi-grid levels """
    return multigrid([10, 20], (np.arange(49) - 24) * 2*np.pi/48)


def case_multigrid_wedge():
    """ open alpha wedge with one multi-grid level and PML in z """
    return multigrid([14], np.linspace(-np.pi/2, np.pi/2, 25), z_bc='PML_8')


def case_steady_state():
    FDTD = openEMS(NrTS=100000, EndCriteria=1e-6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    CSX.GetGrid().SetDeltaUnit(unit)
    channel_1d(FDTD, CSX, sinus=True)
    return FDTD, CSX


def case_coax_resonator():
    """ re-entrant coaxial cavity: a quarter-wave line shorted at one end and loaded
        by a lumped capacitor at the other, in a closed PEC can. Nothing damps it, so
        it rings for the whole run and a small difference between the engines grows
        out of the noise instead of staying in it. """
    post_r, can_r, can_t, post_len, gap = 3.0, 7.0, 1.0, 30.0, 1.0
    lid = post_len + gap
    FDTD = openEMS(NrTS=100000, EndCriteria=0)
    FDTD.SetGaussExcite(1.8e9, 1.2e9)
    FDTD.SetBoundaryCond(['PEC'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    metal = CSX.AddMetal('can')
    metal.AddCylindricalShell([0, 0, 0], [0, 0, lid], can_r + can_t/2, can_t)
    metal.AddCylinder([0, 0, -can_t], [0, 0, 0], can_r + can_t)
    metal.AddCylinder([0, 0, lid], [0, 0, lid + can_t], can_r + can_t)
    metal.AddCylinder([0, 0, 0], [0, 0, post_len], post_r)

    capa = CSX.AddLumpedElement('tuning_C', ny='z', caps=True, C=2e-12)
    capa.AddBox([-post_r, -post_r, post_len], [post_r, post_r, lid], priority=5)
    LumpedPort(CSX, 1, 1000, [0, 0, post_len], [0, 0, lid], 'z', excite=1, priority=10)

    for ny in ('x', 'y'):
        mesh.AddLine(ny, [-can_r - can_t, -can_r, -post_r, 0, post_r, can_r, can_r + can_t])
        mesh.SmoothMeshLines(ny, 0.5, 1.4)
    mesh.AddLine('z', [-can_t, 0, post_len, lid, lid + can_t])
    mesh.AddLine('z', np.linspace(post_len, lid, 3))
    mesh.SmoothMeshLines('z', 1.0, 1.4)

    CSX.AddProbe('et', p_type=0).AddBox([0, 0, post_len], [0, 0, lid])
    return FDTD, CSX



cases = [('excitation',      case_excitation),
         ('pml',             case_pml),
         ('mur',             case_mur),
         ('materials',       case_materials),
         ('lumped',          case_lumped),
         ('lumped_pml',      case_lumped_pml),
         ('tfsf',            case_tfsf),
         ('absorbers',       case_absorbers),
         ('cylinder_closed', case_cylinder_closed),
         ('cylinder_wedge',  case_cylinder_wedge),
         ('multigrid',       case_multigrid),
         ('multigrid2',      case_multigrid2),
         ('multigrid_wedge', case_multigrid_wedge),
         ('dispersive_pml',  case_dispersive_pml),
         ('3d_mixed',        case_3d_mixed),
         ('steady_state',    case_steady_state),
         ('dumps',           case_dumps),
         ('dumps_snapshot',  case_dumps_snapshot),
         ('coax_resonator',  case_coax_resonator)]

RUNS = {
    'cpu':         ({}, {}),
    'cuda':        ({'engine': 'cuda'}, {'OPENEMS_CUDA_GPUS': '1'}),
    'cuda-legacy': ({'engine': 'cuda'}, {'OPENEMS_CUDA_GPUS': '1', 'OPENEMS_CUDA_LEGACY_KERNELS': '1', 'OPENEMS_GPU_DFT': '0'}),
    'cuda-mgpu':   ({'engine': 'cuda'}, {'OPENEMS_CUDA_VIRTUAL': '1', 'OPENEMS_CUDA_GPUS': '3'}),
}
DEVICE_RTOL = 1e-4


def run_one(name, run, path):
    """ child process: build the case and run it, write the setup return code """
    kw, _ = RUNS[run]
    FDTD, CSX = dict(cases)[name]()
    rc = FDTD.Run(path, cleanup=True, verbose=0, **kw)
    with open(path + '.rc', 'w') as f:
        f.write(str(rc if rc is not None else 0))


def run_case(name, run, path):
    env = dict(os.environ)
    for k in ('OPENEMS_CUDA_GPUS', 'OPENEMS_CUDA_VIRTUAL', 'OPENEMS_CUDA_LEGACY_KERNELS', 'OPENEMS_GPU_DFT'):
        env.pop(k, None)
    env.update(RUNS[run][1])
    env['OPENEMS_PROF'] = '1'
    with open(path + '.log', 'w') as log:
        subprocess.run([sys.executable, os.path.abspath(__file__), '--child', name, run, path],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    try:
        rc = int(open(path + '.rc').read())
    except (OSError, ValueError):
        rc = -1
    return rc, open(path + '.log').read()


def deviation(a, b, peak=None):
    """ max. deviation of b from a, relative to peak (default: the peak of a), 0 if identical """
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    if np.array_equal(a, b):
        return 0.0
    if peak is None:
        peak = np.max(np.abs(a))
    return np.max(np.abs(a - b)) / (peak if peak > 0 else 1.0)


def compare_outputs(path_a, path_b, rtol=0):
    """ compare all probe files and HDF5 dumps, return a list of (name, deviation) above rtol """
    diff = []
    files = [f for f in os.listdir(path_a) if os.path.isfile(os.path.join(path_a, f))]
    probes = [f for f in files if '.' not in f]   # probe and port files have no extension
    dumps  = [f for f in files if f.endswith('.h5')]
    assert probes, f'FAIL: no probe files in {path_a}'
    for f in probes:
        a = np.loadtxt(os.path.join(path_a, f), comments='%')
        b = np.loadtxt(os.path.join(path_b, f), comments='%')
        diff.append((f, deviation(a, b)))
    for f in dumps:
        with h5py.File(os.path.join(path_a, f), 'r') as a, h5py.File(os.path.join(path_b, f), 'r') as b:
            names = []
            a.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
            # field data relative to the peak of the whole dump, not of each (possibly decayed) timestep
            fields = [n for n in names if n.startswith('FieldData')]
            peak = max([np.max(np.abs(a[n][()])) for n in fields] or [0])
            for n in names:
                if n not in b:
                    diff.append((f'{f}:{n}', np.inf))
                else:
                    diff.append((f'{f}:{n}', deviation(a[n][()], b[n][()], peak if n in fields else None)))
    worst = max(d for _, d in diff)
    return [(n, d) for n, d in diff if d > rtol], len(probes), len(dumps), worst




def compare_outputs(path_a, path_b, rtol=0):
    """ compare all probe files and HDF5 dumps, return (list of (name, deviation) above rtol, n_probes, n_dumps, worst) """
    diff = []
    files = [f for f in os.listdir(path_a) if os.path.isfile(os.path.join(path_a, f))]
    probes = [f for f in files if '.' not in f]   # probe and port files have no extension
    dumps  = [f for f in files if f.endswith('.h5')]
    assert probes, f'FAIL: no probe files in {path_a}'
    for f in probes:
        a = np.loadtxt(os.path.join(path_a, f), comments='%')
        b = np.loadtxt(os.path.join(path_b, f), comments='%')
        diff.append((f, deviation(a, b)))
    for f in dumps:
        with h5py.File(os.path.join(path_a, f), 'r') as a, h5py.File(os.path.join(path_b, f), 'r') as b:
            names = []
            a.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
            fields = [n for n in names if n.startswith('FieldData')]
            peak = max([np.max(np.abs(a[n][()])) for n in fields] or [0])
            for n in names:
                if n not in b:
                    diff.append((f'{f}:{n}', np.inf))
                else:
                    diff.append((f'{f}:{n}', deviation(a[n][()], b[n][()], peak if n in fields else None)))
    worst = max(d for _, d in diff)
    return [(n, d) for n, d in diff if d > rtol], len(probes), len(dumps), worst


def main(selected):
    base = os.path.join(tempfile.gettempdir(), 'CUDA_Engine')
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    failed = []
    for name, _ in cases:
        if selected and name not in selected:
            continue
        print(f'Testing case: {name}', flush=True)
        paths = {r: os.path.join(base, f'{name}_{r}') for r in RUNS}
        res = {r: run_case(name, r, paths[r]) for r in RUNS}
        ok = True
        if res['cpu'][0] != 0:
            print(f'  FAIL: CPU run returned {res["cpu"][0]}'); failed.append(name); continue
        cuda_rcs = {r: res[r][0] for r in RUNS if r != 'cpu'}
        if all(rc == 4 for rc in cuda_rcs.values()):
            print('  CUDA engine refused the model (extension without CUDA implementation) -- as designed')
            print(f'PASS [{name}]'); continue
        if any(rc != 0 for rc in cuda_rcs.values()):
            print(f'  FAIL: CUDA runs returned {cuda_rcs}'); failed.append(name); continue
        cyl = 'Create FDTD engine (CUDA' not in res['cuda'][1] and 'Create CUDA' not in res['cuda'][1]
        diff, n_p, n_d, worst = compare_outputs(paths['cpu'], paths['cuda'], rtol=DEVICE_RTOL)
        print(f'  cuda vs cpu: {n_p} probe files, {n_d} dumps, max. deviation {worst:.1e} of the peak'
              + (' (ran on the CPU operator)' if cyl else ''))
        if diff:
            ok = False; print('  FAIL: ' + ', '.join(f'{n} ({d:.1e})' for n, d in diff))
        for other in ('cuda-legacy', 'cuda-mgpu'):
            diff, _, _, worst = compare_outputs(paths['cuda'], paths[other], rtol=0)
            print(f'  cuda vs {other}: ' + ('bit-identical' if not diff else 'DIFFERS ' + ', '.join(f'{n} ({d:.1e})' for n, d in diff[:4])))
            if diff:
                ok = False
        fused = 'PML folded into the update kernels' in res['cuda'][1]
        mg = re.search(r'(\d+) slab\(s\)', res['cuda-mgpu'][1])
        print(f'  folded PML: {"yes" if fused else "no"}, multi-GPU slabs: {mg.group(1) if mg else "1 (not eligible)"}')
        print(('PASS' if ok else 'FAIL') + f' [{name}]', flush=True)
        if not ok:
            failed.append(name)
    print('FAILED: ' + ', '.join(failed) if failed else 'PASS')
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(main(sys.argv[1:]))
