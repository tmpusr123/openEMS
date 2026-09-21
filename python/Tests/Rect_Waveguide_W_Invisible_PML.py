"""
 Rectangular waveguide terminated by the invisible PML

 The invisible PML (ABCtype.PML_8/16/32) is an N-cell UPML that lives only in
 the engine extension, extruded behind a sheet that lies on a PEC face. This
 test runs the same physical waveguide [0, length] three ways:

   IPML_MODE=IPML   sheets on the PEC domain boundaries z=0 and z=length
   IPML_MODE=BLOCK  sheets on the faces of PEC blocks inside the domain
                    (the domain runs on 2 cells behind each block face)
   IPML_MODE=REAL   no sheets; the mesh runs on N cells past each end and the
                    z boundaries are openEMS' own PML_N

 The invisible PML uses the equations of the real one, so IPML and BLOCK must
 reproduce REAL to floating-point round-off. Run all three and compare the
 data lines of the port_* files in the Sim_Path folders; they are identical.

 Knobs (IPML_PLOT=0 skips the S-parameter figure, IPML_ZMEAN=0 the zero-mean excitation):
   IPML_DUMP        full E field dump (0)
   IPML_PORT_CELLS  cells between each port's excitation and measurement planes (2)
   IPML_MODE  IPML | BLOCK | REAL  (IPML)
   IPML_N     8 | 16 | 32          (8)
   IPML_GAP   in BLOCK mode, the number of air cells between the PEC block face
              and the sheet (0). The sheet owns every field that couples its
              two sides, so a gap must change nothing.

 (c) 2026 Gadi Lahav <gadi@rfwithcare.com>
"""

import os, tempfile, shutil
import numpy as np

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, Z0

from CSXCAD.CSProperties import ABCtype

MODE = os.environ.get('IPML_MODE', 'IPML').upper()
N = int(os.environ.get('IPML_N', '8'))
PORT_CELLS = int(os.environ.get('IPML_PORT_CELLS', '2'))  # cells between a port's excitation
assert PORT_CELLS >= 1  # plane and its measurement plane
GAP = int(os.environ.get('IPML_GAP', '0'))
assert MODE in ('IPML', 'BLOCK', 'REAL')
assert N in (8, 16, 32)
PML_TYPE = {8: ABCtype.PML_8, 16: ABCtype.PML_16, 32: ABCtype.PML_32}[N]

Sim_Path = os.path.join(tempfile.gettempdir(), 'Rect_WG_IPML_%s_%d%s' % (MODE, N, '_gap%d' % GAP if GAP else ''))
shutil.rmtree(Sim_Path, ignore_errors=True)
os.makedirs(Sim_Path)

unit = 1e-3  # drawing unit in mm

# WR430: 4.300 x 2.150 in. TE10 cut-off c/2a = 1.372 GHz; recommended band
# 1.70-2.60 GHz. Below cut-off TE10 is evanescent: the port impedance is
# imaginary there (inductive), and the S-parameters stay defined.
a = 109.22
b = 54.61
length = 500.0

f0 = 1.55e9  # excitation centre
fc_exc = 1.45e9  # excitation 20 dB half-width: f0 - fc_exc .. f0 + fc_exc
f_start = f0 - fc_exc
f_stop = f0 + fc_exc
f_cutoff = C0 / (2 * a * unit)

TE_mode = 'TE10'

# lambda/50 at the top of the band, in x, y and z; the z mesh is uniform, so
# every variant shares the physical region cell by cell
mesh_res = C0 / f_stop / unit / 50
dz = mesh_res
nz = int(round(length / dz))
dz = length / nz

# energy near cut-off travels (and rings) slowly: allow a long run
FDTD = openEMS(NrTS=200000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc_exc)
# zero time-integral excitation, so the pulse leaves no static charge at the
# port (IPML_ZMEAN=0 turns it off)
FDTD.SetExciteZeroMean(bool(int(os.environ.get('IPML_ZMEAN', '1'))))

zbc = 'PML_%d' % N if MODE == 'REAL' else 'PEC'
FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', zbc, zbc])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

mesh.AddLine('x', [0, a])
mesh.AddLine('y', [0, b])
mesh.SmoothMeshLines('x', mesh_res, ratio=1.4)
mesh.SmoothMeshLines('y', mesh_res, ratio=1.4)

ext = {'IPML': 0, 'BLOCK': 2 + GAP, 'REAL': N}[MODE]
mesh.AddLine('z', dz * np.arange(-ext, nz + ext + 1))

ports = []
ports.append(FDTD.AddRectWaveGuidePort(0, [0, 0, 3 * dz], [a, b, (3 + PORT_CELLS) * dz], 'z', a * unit, b * unit, TE_mode, 1))
ports.append(FDTD.AddRectWaveGuidePort(1, [0, 0, length - 3 * dz], [a, b, length - (3 + PORT_CELLS) * dz], 'z', a * unit, b * unit, TE_mode))

if MODE == 'BLOCK':
    pec = CSX.AddMetal('PEC_blocks')
    pec.AddBox(priority=5, start=[0, 0, -ext * dz], stop=[a, b, -GAP * dz])
    pec.AddBox(priority=5, start=[0, 0, length + GAP * dz], stop=[a, b, length + ext * dz])

if MODE in ('IPML', 'BLOCK'):
    abs1 = CSX.AddAbsorbingBC('abs1', NormalSignPositive=True, AbsorbingBoundaryType=PML_TYPE)
    abs1.AddBox([0, 0, 0], [a, b, 0], priority=6)
    abs2 = CSX.AddAbsorbingBC('abs2', NormalSignPositive=False, AbsorbingBoundaryType=PML_TYPE)
    abs2.AddBox([0, 0, length], [a, b, length], priority=6)

if int(os.environ.get('IPML_DUMP', '0')):  # full E field dump
    Et = CSX.AddDump('Et', file_type=0, dump_type=0, dump_mode=1)
    Et.AddBox([0, 0, 0], [a, b, length])

print('WR430: a = %.2f mm, b = %.2f mm, TE10 cut-off %.4f GHz; mesh %.3f mm, %d z cells'
      % (a, b, f_cutoff / 1e9, mesh_res, nz))
print('mode = %s, N = %d, Sim_Path = %s' % (MODE, N, Sim_Path))
FDTD.Run(Sim_Path, verbose=3, cleanup=False)

freq = np.linspace(f_start, f_stop, 481)
with np.errstate(invalid='ignore', divide='ignore'):
    for port in ports:
        port.CalcPort(Sim_Path, freq)
    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_ref / ports[0].uf_inc
# below cut-off the port impedance is imaginary (TE: inductive), the waves stay defined
evan = freq <= f_cutoff
s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))


def td_waves(port):
    """Time-domain incident / reflected voltage and current of a waveguide port.

    ports.py only splits in the time domain for a scalar Z_ref. The TE/TM port
    impedance depends on frequency (k*Z0/beta for TE), so the split is done per
    frequency and transformed back: FFT of the total u(t), i(t), zero-padded;
    u_inc = (U + Z*I)/2, i_inc = (I + U/Z)/2; inverse FFT. Below cut-off the
    decaying branch beta = -j*alpha is taken, so Z is imaginary (inductive for
    TE) there and the split still holds. The DC bin is dropped: the evanescent
    TE admittance is infinite at f = 0, and the zero-mean excitation has no DC.
    """
    t = np.asarray(port.u_data.ui_time[0])
    u = np.asarray(port.ut_tot, float)
    i = np.asarray(port.it_tot, float)
    dt = t[1] - t[0]
    n = 1 << int(np.ceil(np.log2(len(t)))) + 2
    f = np.fft.rfftfreq(n, dt)
    U = np.fft.rfft(u, n)
    I = np.fft.rfft(i, n)
    k0 = 2 * np.pi * f / C0
    k = k0 * port.ref_index
    beta = np.conj(np.sqrt(k ** 2 - port.kc ** 2 + 0j))  # propagating: > 0; evanescent: -j*alpha
    beta[beta == 0] = 1e-12
    # the same wave impedance as WaveguidePort.CalcPort
    Z = beta * Z0 / (k0 * port.ref_index ** 2) if port.mode_type == 'TM' else k0 * Z0 / beta
    U_inc = np.zeros_like(U); I_inc = np.zeros_like(I)
    U_inc[1:] = 0.5 * (U[1:] + Z[1:] * I[1:])
    I_inc[1:] = 0.5 * (I[1:] + U[1:] / Z[1:])
    u_inc = np.fft.irfft(U_inc, n)[:len(t)]
    i_inc = np.fft.irfft(I_inc, n)[:len(t)]
    return t, u_inc, u - u_inc, i_inc, i_inc - i


td = [td_waves(p) for p in ports]

print('\n  f/GHz   |S11| dB   |S21| dB')
for f in (0.5e9, 1.0e9, 1.5e9, 1.7e9, 2.0e9, 2.25e9, 2.5e9):
    k = np.argmin(abs(freq - f))
    print('  %5.2f   %8.2f   %8.3f%s' % (freq[k] / 1e9, s11_dB[k], s21_dB[k],
                                         '   (evanescent, below the %.3f GHz cut-off)' % (f_cutoff / 1e9) if evan[k] else ''))
band = freq >= 1.7e9
print('  worst |S11| over the WR430 band 1.70-%.2f GHz: %.2f dB' % (f_stop / 1e9, np.nanmax(s11_dB[band])))

# ## Plot the S-parameters (IPML_PLOT=0 skips the figure, for batch runs)
if int(os.environ.get('IPML_PLOT', '1')):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(freq / 1e9, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
    plt.plot(freq / 1e9, s21_dB, 'b-', linewidth=2, label='$S_{21}$')
    plt.axvline(f_cutoff / 1e9, color='r', linestyle='--', label='TE$_{10}$ cut-off')
    plt.xlim(f_start / 1e9, f_stop / 1e9)
    plt.grid()
    plt.legend()
    plt.ylabel('S-Parameter (dB)')
    plt.xlabel('Frequency (GHz)')
    plt.title('WR430, %s, N = %d' % (MODE, N))
    plt.show()
    
    # time-domain incident / reflected voltage and current, one subplot per port
    fig, axs = plt.subplots(len(ports), 1, sharex=True, figsize=(9, 3.2 * len(ports)))
    for n, (ax, (t, u_inc, u_ref, i_inc, i_ref)) in enumerate(zip(np.atleast_1d(axs), td)):
        ax.plot(t * 1e9, u_inc, 'k-', linewidth=1.5, label='$V^+$')
        ax.plot(t * 1e9, u_ref, 'b-', linewidth=1.5, label='$V^-$')
        ax.set_ylabel('voltage (V)')
        ax.grid()
        axi = ax.twinx()
        axi.plot(t * 1e9, i_inc, 'r--', linewidth=1.2, label='$I^+$')
        axi.plot(t * 1e9, i_ref, 'm--', linewidth=1.2, label='$I^-$')
        axi.set_ylabel('current (A)')
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axi.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right')
        ax.set_title('port %d' % (n + 1))
    np.atleast_1d(axs)[-1].set_xlabel('time (ns)')
    fig.suptitle('WR430, %s, N = %d: incident / reflected waves (TE$_{10}$ split, evanescent below cut-off)' % (MODE, N))
    fig.tight_layout()
    plt.show()
