"""
 Microstrip line terminated by the invisible PML

 A 120 mm z-directed microstrip with a mode-file waveguide port near each end.
 Same three variants as Rect_Waveguide_W_Invisible_PML.py:

   IPML_MODE=IPML   sheets on the PEC domain boundaries z=0 and z=length
   IPML_MODE=BLOCK  sheets on the faces of PEC blocks inside the domain
   IPML_MODE=REAL   no sheets; mesh, substrate, strip and ground run on N cells
                    past each end into openEMS' own PML_N z boundaries

 Here the PML is not empty: the substrate, the PEC strip and the ground plane
 all run into it. The invisible PML extrudes the cross-section in front of the
 sheet, so with a closed (PEC) box IPML and BLOCK must reproduce REAL to
 round-off.

 Knobs (IPML_PLOT=0 skips the S-parameter figure, IPML_ZMEAN=0 the zero-mean excitation):
   IPML_DUMP        full E field dump (0)
   IPML_VIEW        open the structure in AppCSXCAD before running (0)
   IPML_CEX         excess capacitance of the excited port: '' off, pF, or 'auto' (fit)
   IPML_PORT_CELLS  cells between each port's excitation and measurement planes (2)
   IPML_MODE  IPML | BLOCK | REAL        (IPML)
   IPML_N     8 | 16 | 32                (8)
   IPML_YMAX  PEC | MUR | PML_8  y-max boundary  (MUR). With MUR the two differ at the
              top of the PML: openEMS' Mur runs over the whole y-max face, into
              the real PML, while the virtual PML is closed by PEC there.

 The cross-section is that of MSL_E.csv / MSL_H.csv (see MSL_mode_params.json).

 (c) 2026 Gadi Lahav <gadi@rfwithcare.com>
"""

import os, tempfile, shutil, json
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

from CSXCAD.CSProperties import ABCtype

MODE = os.environ.get('IPML_MODE', 'IPML').upper()
N = int(os.environ.get('IPML_N', '8'))
PORT_CELLS = int(os.environ.get('IPML_PORT_CELLS', '2'))  # cells between a port's excitation
assert PORT_CELLS >= 1  # plane and its measurement plane
YMAX = os.environ.get('IPML_YMAX', 'MUR').upper()
assert MODE in ('IPML', 'BLOCK', 'REAL')
assert N in (8, 16, 32)
assert YMAX in ('PEC', 'MUR', 'PML_8')
PML_TYPE = {8: ABCtype.PML_8, 16: ABCtype.PML_16, 32: ABCtype.PML_32}[N]

Sim_Path = os.path.join(tempfile.gettempdir(), 'MSL_IPML_%s_%d_%s' % (MODE, N, YMAX))
shutil.rmtree(Sim_Path, ignore_errors=True)
os.makedirs(Sim_Path)

# ## Cross-section -- identical to the one the mode files were solved for
microstrip_W = 1.875
substrate_epsR = 4.5
substrate_width = 15.0
substrate_thickness = 1.0
substrate_cells = 8
trace_cells = 7
cu_thick = 0.1
port_w_fact = 6.0
port_h_fact = 5.5
Airbox_Add = 12.5
unit = 1e-3

substrate_length = 120.0

f0, fc_exc = 1.55e9, 1.45e9

FDTD = openEMS(NrTS=30000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc_exc)
# zero time-integral excitation, so the pulse leaves no static charge at the
# port (IPML_ZMEAN=0 turns it off)
FDTD.SetExciteZeroMean(bool(int(os.environ.get('IPML_ZMEAN', '1'))))

zbc = 'PML_%d' % N if MODE == 'REAL' else 'PEC'
FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', YMAX, zbc, zbc])

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)
mesh_res = ((C0 / (f0 + fc_exc)) / unit) / 75

# uniform z mesh; the variants differ only past the ends of the line
nz = int(round(substrate_length / (mesh_res / 2.0)))
dz = substrate_length / nz
ext = {'IPML': 0, 'BLOCK': 2, 'REAL': N}[MODE]
z0, z1 = -ext * dz, substrate_length + ext * dz
mesh.AddLine('z', dz * np.arange(-ext, nz + ext + 1))

SimBox = [-substrate_width * 0.5, substrate_width * 0.5,
          -cu_thick, substrate_thickness * (1.0 + port_h_fact) + Airbox_Add,
          z0, z1]

mesh.AddLine('x', SimBox[0:2])
mesh.AddLine('y', SimBox[2:4])

line = CSX.AddMetal('cu_top')
line.AddBox(priority=20, start=[-microstrip_W / 2, substrate_thickness, z0],
            stop=[microstrip_W / 2, substrate_thickness + cu_thick, z1])
mesh.AddLine('x', [-microstrip_W / 2, microstrip_W / 2])
mesh.AddLine('y', [substrate_thickness, substrate_thickness + cu_thick])
mesh.AddLine('x', np.linspace(-0.5 * microstrip_W, 0.5 * microstrip_W, trace_cells))

p_x = microstrip_W * (1.0 + port_w_fact) * 0.5
p_y1 = substrate_thickness * (1.0 + port_h_fact)
mesh.AddLine('x', [-p_x, p_x])
mesh.AddLine('y', [0.0, p_y1])

sub = CSX.AddMaterial('FR4', epsilon=substrate_epsR)
sub.AddBox(priority=2, start=[-substrate_width / 2, 0.0, z0],
           stop=[substrate_width / 2, substrate_thickness, z1])
mesh.AddLine('y', np.linspace(0, substrate_thickness, substrate_cells + 1))

gnd = CSX.AddMetal('cu_bot')
gnd.AddBox(priority=10, start=[-substrate_width / 2, -cu_thick, z0],
           stop=[substrate_width / 2, 0.0, z1])
mesh.AddLine('y', [-cu_thick, 0.0])

mesh.SmoothMeshLines('x', mesh_res, 1.4)
mesh.SmoothMeshLines('y', mesh_res, 1.4)

with open('MSL_mode_params.json') as fh:
    _mp = json.load(fh)
assert abs(_mp['microstrip_W'] - microstrip_W) < 1e-9
shutil.copy('MSL_E.csv', os.path.join(Sim_Path, 'MSL_E.csv'))
shutil.copy('MSL_H.csv', os.path.join(Sim_Path, 'MSL_H.csv'))

box_lo = [-p_x, 0.0]
box_hi = [p_x, p_y1]
zP1, zP2 = 4 * dz, substrate_length - 4 * dz
port1 = FDTD.AddWaveGuidePort(1, box_lo + [zP1], box_hi + [zP1 + PORT_CELLS * dz], 'z',
                              E_file="MSL_E.csv", H_file="MSL_H.csv", kc=0.0, excite=1, excite_type=0)
port2 = FDTD.AddWaveGuidePort(2, box_lo + [zP2], box_hi + [zP2 - PORT_CELLS * dz], 'z',
                              E_file="MSL_E.csv", H_file="MSL_H.csv", kc=0.0, excite=0, excite_type=0)

xy0 = [SimBox[0], -cu_thick] 
xy1 = [SimBox[1], substrate_thickness * (1.0 + port_h_fact) * 1.25]

if MODE == 'BLOCK':
    pec = CSX.AddMetal('PEC_blocks')
    pec.AddBox(priority=50, start=xy0 + [z0], stop=xy1 + [0.0])
    pec.AddBox(priority=50, start=xy0 + [substrate_length], stop=xy1 + [z1])

if MODE in ('IPML', 'BLOCK'):
    abs1 = CSX.AddAbsorbingBC('abs1', NormalSignPositive=True, AbsorbingBoundaryType=PML_TYPE)
    abs1.AddBox(xy0 + [0.0], xy1 + [0.0], priority=60)
    abs2 = CSX.AddAbsorbingBC('abs2', NormalSignPositive=False, AbsorbingBoundaryType=PML_TYPE)
    abs2.AddBox(xy0 + [substrate_length], xy1 + [substrate_length], priority=60)

if int(os.environ.get('IPML_DUMP', '0')):  # full E field dump
    Et = CSX.AddDump('Et', file_type=0, dump_type=0, dump_mode=1)
    Et.AddBox([float(SimBox[0]), float(SimBox[2]), float(SimBox[4])],
              [float(SimBox[1]), float(SimBox[3]), float(SimBox[5])])

if int(os.environ.get('IPML_VIEW', '0')):  # show the structure before running
    CSX_file = os.path.join(Sim_Path, 'MSL_IPML.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)

    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

print('mode = %s, N = %d, y-max = %s, dz = %.5f mm, Sim_Path = %s' % (MODE, N, YMAX, dz, Sim_Path))
FDTD.Run(Sim_Path, verbose=3, cleanup=False)

freq = np.linspace(f0 - fc_exc, f0 + fc_exc, 291)
for p in (port1, port2):
    p.CalcPort(Sim_Path, freq)
# Z_ref, measured by the passive port. The mode-match probes normalise their
# templates, so a forward wave gives U/I = the port's own ratio, which is the
# DISCRETE line's impedance -- a continuum mode solver does not give it (the
# staircased line has its own L and C). Port 2 is not excited and the invisible
# PML behind it absorbs everything, so it only ever sees the outgoing wave:
# -U2/I2 (its current probe points into the line) IS Z_ref, per frequency.
# Both ports sit on the same line, so the one value serves both.
Z_ref = -port2.uf_tot / port2.if_tot
for p in (port1, port2):
    p.CalcPort(Sim_Path, freq, ref_impedance=Z_ref)
print('Z_ref from the passive port: Re %.2f .. %.2f Ohm, |Im| <= %.2f Ohm'
      % (Z_ref.real.min(), Z_ref.real.max(), abs(Z_ref.imag).max()))

# Excess capacitance of the EXCITED port (see Port.CalcPort, C_excess): the
# injected field that is not the mode stays near the source and adds
# I/(j*w*C_ex) to port 1's voltage. IPML_CEX: '' off, a value in pF, or 'auto'
# -- fitted here from port 1 over f <= 1 GHz, which is valid because this test
# is a matched line (port 1 then sees Z_ref plus the excess reactance only).
CEX = os.environ.get('IPML_CEX', '').strip().lower()
C_ex = None
if CEX == 'auto':
    lf = freq <= 1e9
    dZ = port1.uf_tot / port1.if_tot - Z_ref
    C_ex = -1.0 / np.mean(2 * np.pi * freq[lf] * dZ.imag[lf])
elif CEX:
    C_ex = float(CEX) * 1e-12
if C_ex:
    port1.CalcPort(Sim_Path, freq, ref_impedance=Z_ref, C_excess=C_ex)
    print('C_excess of the excited port: %.3f pF (%s)' % (C_ex * 1e12, 'fitted' if CEX == 'auto' else 'given'))
s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc
print('\n  f/GHz   |S11| dB   |S21| dB')
for f in (0.1e9, 0.5e9, 1e9, 2e9, 3e9):
    k = np.argmin(abs(freq - f))
    print('  %5.2f   %8.2f   %8.3f' % (freq[k] / 1e9, 20 * np.log10(abs(s11[k])), 20 * np.log10(abs(s21[k]))))
print('  worst |S11| over the band: %.2f dB' % (20 * np.log10(abs(s11).max())))

# ## Plot the S-parameters (IPML_PLOT=0 skips the figure, for batch runs)
if int(os.environ.get('IPML_PLOT', '1')):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(freq / 1e9, 20 * np.log10(np.abs(s11)), 'k-', linewidth=2, label='$S_{11}$')
    plt.plot(freq / 1e9, 20 * np.log10(np.abs(s21)), 'b-', linewidth=2, label='$S_{21}$')
    plt.grid()
    plt.legend()
    plt.ylabel('S-Parameter (dB)')
    plt.xlabel('Frequency (GHz)')
    plt.title('MSL, %s, N = %d, y-max %s' % (MODE, N, YMAX))
    plt.show()
