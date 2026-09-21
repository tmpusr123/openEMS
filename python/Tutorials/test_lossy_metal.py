"""
Test script for LossyMetal (SIBC) implementation.

Simulates a simple half-wave dipole at 1 GHz with:
  1. PEC (perfect metal)
  2. LossyMetal (copper, sigma=5.8e7 S/m)

Compares input impedance - LossyMetal should show slightly higher
real part (ohmic loss) compared to PEC.
"""

import os
import tempfile
import shutil
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0


def run_dipole_sim(sim_path, use_lossy_metal=False, conductivity=5.8e7):
    """Run a half-wave dipole simulation using a lumped port."""

    f0 = 1e9        # center frequency 1 GHz
    fc = 0.5e9      # bandwidth
    f_max = f0 + fc

    # Dipole dimensions
    dipole_length = 0.5 * C0 / f0   # half-wavelength at 1 GHz ~ 150 mm
    dipole_width = 2e-3              # 2 mm width
    gap = 1e-3                       # 1 mm feed gap

    # Simulation domain
    lambda0 = C0 / f0
    sim_size = lambda0 * 0.6

    # Create FDTD
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PML_8'] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # drawing unit = mm

    # Material: PEC or LossyMetal
    metal_name = 'copper_lossy' if use_lossy_metal else 'copper_pec'
    if use_lossy_metal:
        metal = CSX.AddLossyMetal(metal_name, conductivity=conductivity)
    else:
        metal = CSX.AddMetal(metal_name)

    # Dipole arms (two halves with gap in z)
    half_len = dipole_length / 2 * 1e3  # mm
    w = dipole_width / 2 * 1e3          # mm
    g = gap / 2 * 1e3                   # mm

    # Upper arm: z from +g to +half_len
    metal.AddBox([-w, -w, g], [w, w, half_len])
    # Lower arm: z from -half_len to -g
    metal.AddBox([-w, -w, -half_len], [w, w, -g])

    # Lumped port in the gap (z-directed)
    port = FDTD.AddLumpedPort(
        port_nr=1,
        R=50,  # 50 Ohm reference impedance
        start=[-w, -w, -g],
        stop=[w, w, g],
        p_dir='z',
        excite=1.0
    )

    # Mesh
    sim_mm = sim_size * 1e3
    mesh_res = C0 / f_max / 20 * 1e3  # lambda/20 in mm

    mesh.AddLine('x', np.array([-sim_mm, -w, 0, w, sim_mm]))
    mesh.SmoothMeshLines('x', mesh_res, 1.3)

    mesh.AddLine('y', np.array([-sim_mm, -w, 0, w, sim_mm]))
    mesh.SmoothMeshLines('y', mesh_res, 1.3)

    mesh.AddLine('z', np.array([-sim_mm, -half_len, -g, 0, g, half_len, sim_mm]))
    mesh.SmoothMeshLines('z', mesh_res, 1.3)

    # Run
    os.makedirs(sim_path, exist_ok=True)
    CSX.Write2XML(os.path.join(sim_path, 'geometry.xml'))
    FDTD.Run(sim_path, verbose=1)

    # Post-process
    freq = np.linspace(f0 - fc, f0 + fc, 501)
    port.CalcPort(sim_path, freq)

    Z_in = port.uf_tot / port.if_tot
    S11 = port.uf_ref / port.uf_inc

    return freq, Z_in, S11


def main():
    base_dir = tempfile.mkdtemp(prefix='lossy_metal_test_')
    print(f"Test directory: {base_dir}")

    try:
        # Run PEC simulation
        print("\n=== Running PEC dipole simulation ===")
        pec_path = os.path.join(base_dir, 'pec_dipole')
        freq_pec, Z_pec, S11_pec = run_dipole_sim(pec_path, use_lossy_metal=False)

        # Run LossyMetal simulation
        print("\n=== Running LossyMetal (copper) dipole simulation ===")
        lm_path = os.path.join(base_dir, 'lossy_dipole')
        freq_lm, Z_lm, S11_lm = run_dipole_sim(lm_path, use_lossy_metal=True, conductivity=5.8e7)

        # Compare at center frequency
        f0_idx = np.argmin(np.abs(freq_pec - 1e9))

        print("\n" + "="*60)
        print("Results at 1 GHz:")
        print("="*60)
        print(f"PEC:        Z_in = {np.real(Z_pec[f0_idx]):.2f} + j{np.imag(Z_pec[f0_idx]):.2f} Ohm")
        print(f"LossyMetal: Z_in = {np.real(Z_lm[f0_idx]):.2f} + j{np.imag(Z_lm[f0_idx]):.2f} Ohm")
        print(f"|S11| PEC:        {20*np.log10(np.abs(S11_pec[f0_idx])):.2f} dB")
        print(f"|S11| LossyMetal: {20*np.log10(np.abs(S11_lm[f0_idx])):.2f} dB")

        dR = np.real(Z_lm[f0_idx]) - np.real(Z_pec[f0_idx])
        dX = np.imag(Z_lm[f0_idx]) - np.imag(Z_pec[f0_idx])
        print(f"\nDifference: dR = {dR:.4f} Ohm, dX = {dX:.4f} Ohm")

        # Find resonance (minimum |S11|)
        res_idx_pec = np.argmin(np.abs(S11_pec))
        res_idx_lm = np.argmin(np.abs(S11_lm))
        print(f"\nResonance PEC:        f = {freq_pec[res_idx_pec]/1e9:.4f} GHz, Z = {np.real(Z_pec[res_idx_pec]):.2f} + j{np.imag(Z_pec[res_idx_pec]):.2f} Ohm")
        print(f"Resonance LossyMetal: f = {freq_lm[res_idx_lm]/1e9:.4f} GHz, Z = {np.real(Z_lm[res_idx_lm]):.2f} + j{np.imag(Z_lm[res_idx_lm]):.2f} Ohm")

        dR_res = np.real(Z_lm[res_idx_lm]) - np.real(Z_pec[res_idx_pec])
        print(f"\nResistance increase at resonance: {dR_res:.4f} Ohm")

        if dR_res > 0:
            print("SUCCESS: LossyMetal correctly adds ohmic resistance to the dipole.")
        elif dR_res > -0.1:
            print("NOTE: Resistance difference is very small (< 0.1 Ohm) - may be within numerical noise for copper.")
        else:
            print("WARNING: Unexpected resistance decrease - check SIBC implementation.")

        # Expected: R_s for copper at 1 GHz ~ 0.026 Ohm/sq
        # For a dipole, ohmic loss should add a small fraction of an Ohm
        import math
        R_s_expected = math.sqrt(2*math.pi*1e9 * 4*math.pi*1e-7 / (2 * 5.8e7))
        print(f"\nExpected surface resistance at 1 GHz: R_s = {R_s_expected*1000:.2f} mOhm/sq")

    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
