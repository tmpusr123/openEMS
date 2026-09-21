"""
 Coplanar waveguide terminated by the invisible PML

 The cross-section and the post-processing are those of CPW_With_WG_Ports.py;
 the mode files (CPW_IPML_E/H.csv) come from gen_cpw_modes_ipml.py, which
 solves THIS script's cross-section and window. What changes is the
 termination: the line runs to the y ends of the domain, and a full
 cross-section invisible PML sheet sits on each end plane. The y mesh is
 uniform, so the virtual PML cells have the size of the line's cells.

   IPML_MODE=IPML   sheets on the PEC y domain boundaries
   IPML_MODE=BLOCK  sheets on the faces of PEC blocks inside the domain
   IPML_MODE=REAL   no sheets; mesh, substrate, line and grounds run on N cells
                    past each end into openEMS' own PML_N y boundaries

 Knobs (IPML_PLOT=0 skips the S-parameter figure, IPML_ZMEAN=0 the zero-mean excitation):
   IPML_DUMP        full E field dump (0)
   IPML_VIEW        open the structure in AppCSXCAD before running (0)
   IPML_CEX         excess capacitance of the excited port: '' off, pF, or 'auto' (fit)
   IPML_PORT_CELLS  cells between each port's excitation and measurement planes (1)
   IPML_MODE   IPML | BLOCK | REAL          (IPML)
   IPML_N      8 | 16 | 32                  (8)
   IPML_SIDES  MUR | PEC  x and z boundaries (PEC). With PEC, IPML and BLOCK
               reproduce REAL to round-off. With MUR they differ slightly at
               the rim of the PML: openEMS' Mur runs over the whole x/z faces,
               into the real PML, while the virtual PML is closed by PEC there.

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
PORT_CELLS = int(os.environ.get('IPML_PORT_CELLS', '1'))  # cells between a port's excitation
assert PORT_CELLS >= 1                                   # plane and its measurement plane
SIDES = os.environ.get('IPML_SIDES', 'PEC').upper()

assert MODE in ('IPML', 'BLOCK', 'REAL')
assert N in (8, 16, 32)
assert SIDES in ('MUR', 'PEC')
PML_TYPE = {8: ABCtype.PML_8, 16: ABCtype.PML_16, 32: ABCtype.PML_32}[N]

Sim_Path = os.path.join(tempfile.gettempdir(), 'CPW_IPML_%s_%d_%s' % (MODE, N, SIDES))
shutil.rmtree(Sim_Path, ignore_errors=True)
os.makedirs(Sim_Path)

# ## Geometry -- as in CPW_With_WG_Ports.py
Line_W = 1.15  # must match the mode file
CPW_gap = 0.3

substrate_epsR = 4.3
substrate_width = 11
substrate_length = 80
substrate_thickness = 1

gap_cells = 3
trace_cells = 7
substrate_cells = 4

port_w_fact = 7.5
port_h_fact = 11.0  # port window: z = -(port_h_fact-2)/2 .. port_h_fact/2, i.e. 10 mm
                    # centred on the substrate. Changing it (or anything in the
                    # cross-section) needs new mode files: gen_cpw_modes_ipml.py

cu_thick = 0.1

Airbox_Add = 12.5
unit_res = 1e-3

f0 = 1.55e9
fc = 1.45e9

FDTD = openEMS(NrTS=40000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
# zero time-integral excitation, so the pulse leaves no static charge at the
# port (IPML_ZMEAN=0 turns it off)
FDTD.SetExciteZeroMean(bool(int(os.environ.get('IPML_ZMEAN', '1'))))
ybc = 'PML_%d' % N if MODE == 'REAL' else 'PEC'
FDTD.SetBoundaryCond([SIDES, SIDES, ybc, ybc, SIDES, SIDES])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit_res)
mesh_res = ((C0 / (f0 + fc)) / unit_res) / 50

# uniform y mesh; the variants differ only past the ends of the line
dy = mesh_res / 2
ny = int(round(substrate_length / dy))
dy = substrate_length / ny
ext = {'IPML': 0, 'BLOCK': 2, 'REAL': N}[MODE]
y0, y1 = -ext * dy, substrate_length + ext * dy
mesh.AddLine('y', dy * np.arange(-ext, ny + ext + 1))

SimBox = np.array([-substrate_width * 0.5 - Airbox_Add,
                   substrate_width * 0.5 + Airbox_Add,
                   y0,
                   y1,
                   -substrate_thickness * (port_h_fact - 1.0) - Airbox_Add,
                   substrate_thickness * (1.0 + port_h_fact) + Airbox_Add])
mesh.AddLine('x', SimBox[0:2])
mesh.AddLine('z', SimBox[4:6])

line = CSX.AddMetal('cu_top')
line.AddBox(priority=20, start=[-Line_W / 2, y0, substrate_thickness],
            stop=[Line_W / 2, y1, substrate_thickness + cu_thick])
mesh.AddLine('x', [-Line_W / 2, Line_W / 2])
mesh.AddLine('z', [substrate_thickness, substrate_thickness + cu_thick])
mesh.AddLine('x', np.linspace(-0.5 * Line_W, 0.5 * Line_W, trace_cells))

p_x = Line_W * (1.0 + port_w_fact) * 0.5
p_z0 = -substrate_thickness * (port_h_fact - 2.0) * 0.5
p_z1 = substrate_thickness * port_h_fact * 0.5
mesh.AddLine('x', [-p_x, p_x])
mesh.AddLine('z', [p_z0, p_z1])

sub = CSX.AddMaterial('FR4', epsilon=substrate_epsR)
sub.AddBox(priority=2, start=[-substrate_width / 2, y0, 0.0],
           stop=[substrate_width / 2, y1, substrate_thickness])
mesh.AddLine('x', [-substrate_width / 2, substrate_width / 2])
mesh.AddLine('z', np.linspace(0, substrate_thickness, substrate_cells + 1))

gnd = CSX.AddMetal('cu_gnd')
g_in = 0.5 * (Line_W + CPW_gap * 2)
gnd.AddBox(priority=10, start=[-0.5 * substrate_width, y0, substrate_thickness],
           stop=[-g_in, y1, substrate_thickness + cu_thick])
gnd.AddBox(priority=10, start=[g_in, y0, substrate_thickness],
           stop=[0.5 * substrate_width, y1, substrate_thickness + cu_thick])
mesh.AddLine('x', [-g_in, g_in])
mesh.AddLine('x', np.linspace(-g_in, -Line_W * 0.5, gap_cells))
mesh.AddLine('x', np.linspace(Line_W * 0.5, g_in, gap_cells))

mesh.SmoothMeshLines('x', mesh_res, 1.4)
mesh.SmoothMeshLines('z', mesh_res, 1.4)

# ---- MESH DONE (gen_cpw_modes_ipml.py runs this script up to here)

# ## Validate the mode file against THIS geometry, before anything is run.
# The mode file defines the geometry: its sample grid must be this mesh's lines
# inside the port window, solved for this cross-section.
with open('CPW_IPML_mode_params.json') as fh:
    _mp = json.load(fh)
_regen = ' -- regenerate: python3 gen_cpw_modes_ipml.py (blit_port_mode_solver)'
for _k, _v in (('Line_W', Line_W), ('CPW_gap', CPW_gap), ('substrate_epsR', substrate_epsR),
               ('substrate_thickness', substrate_thickness), ('substrate_width', substrate_width),
               ('cu_thick', cu_thick), ('p_x', p_x), ('p_z0', p_z0), ('p_z1', p_z1),
               ('sim_zmin', SimBox[4]), ('sim_zmax', SimBox[5]),
               ('sim_xmin', SimBox[0]), ('sim_xmax', SimBox[1])):
    assert _k in _mp and abs(_mp[_k] - _v) < 1e-9, 'mode file: %s is %s, script has %s%s' % (_k, _mp.get(_k), _v, _regen)
_xs = np.unique(np.asarray(mesh.GetLines('x')))
_zs = np.unique(np.asarray(mesh.GetLines('z')))
_wx = _xs[(_xs >= -p_x - 1e-9) & (_xs <= p_x + 1e-9)]
_wz = _zs[(_zs >= p_z0 - 1e-9) & (_zs <= p_z1 + 1e-9)]
assert len(_wx) == len(_mp['x_lines']) and np.allclose(_wx, _mp['x_lines'], atol=1e-9), 'mode file: x lines differ from the mesh' + _regen
assert len(_wz) == len(_mp['z_lines']) and np.allclose(_wz, _mp['z_lines'], atol=1e-9), 'mode file: z lines differ from the mesh' + _regen
for _f in ('CPW_IPML_E.csv', 'CPW_IPML_H.csv'):
    _d = np.loadtxt(_f, delimiter=',')  # col0 = z - p_z0, col1 = x + p_x
    assert np.allclose(np.unique(_d[:, 0]) + p_z0, _wz, atol=1e-9) and \
           np.allclose(np.unique(_d[:, 1]) - p_x, _wx, atol=1e-9), '%s: grid differs from the mesh%s' % (_f, _regen)
if SIDES != _mp['outer_wall']:
    print('NOTE: the mode was solved with a %s wall at the SimBox; this run has %s sides' % (_mp['outer_wall'], SIDES))
print('mode file OK: %d x %d window lines, n_eff %.4f, Zw %.2f, Zl %.2f'
      % (len(_wx), len(_wz), _mp['n_eff'], _mp['Zw'], _mp['Zl']))
shutil.copy("CPW_IPML_E.csv", Sim_Path)
shutil.copy("CPW_IPML_H.csv", Sim_Path)

# ports 4 cells in from each end, as far from the sheets as in the MSL test
yP1, yP2 = 4 * dy, substrate_length - 4 * dy
port1 = FDTD.AddWaveGuidePort(1, [-p_x, yP1, p_z0], [p_x, yP1 + PORT_CELLS * dy, p_z1], 'y',
                              E_file="CPW_IPML_E.csv", H_file="CPW_IPML_H.csv", kc=0.0, excite=1, excite_type=0)
port2 = FDTD.AddWaveGuidePort(2, [-p_x, yP2, p_z0], [p_x, yP2 - PORT_CELLS * dy, p_z1], 'y',
                              E_file="CPW_IPML_E.csv", H_file="CPW_IPML_H.csv", kc=0.0, excite=0, excite_type=0)

xz0 = [-p_x * 1.5, p_z0 - (p_z1 - p_z0) * 0.25]
xz1 = [ p_x * 1.5, p_z1 + (p_z1 - p_z0) * 0.25]
# xz0 = [SimBox[0], SimBox[4]]  # (x, z) corners of the full cross-section
# xz1 = [SimBox[1], SimBox[5]]


def box(x_z, y):
    return [x_z[0], y, x_z[1]]


if MODE == 'BLOCK':
    pec = CSX.AddMetal('PEC_blocks')
    pec.AddBox(priority=50, start=box(xz0, y0), stop=box(xz1, 0.0))
    pec.AddBox(priority=50, start=box(xz0, substrate_length), stop=box(xz1, y1))

if MODE in ('IPML', 'BLOCK'):
    abs1 = CSX.AddAbsorbingBC('abs1', NormalSignPositive=True, AbsorbingBoundaryType=PML_TYPE)
    abs1.AddBox(box(xz0, 0.0), box(xz1, 0.0), priority=60)
    abs2 = CSX.AddAbsorbingBC('abs2', NormalSignPositive=False, AbsorbingBoundaryType=PML_TYPE)
    abs2.AddBox(box(xz0, substrate_length), box(xz1, substrate_length), priority=60)

if int(os.environ.get('IPML_DUMP', '0')):  # full E field dump
    Et = CSX.AddDump('Et', file_type=0, dump_type=0, dump_mode=1)
    Et.AddBox([float(SimBox[0]), float(SimBox[2]), float(SimBox[4])],
              [float(SimBox[1]), float(SimBox[3]), float(SimBox[5])])

if int(os.environ.get('IPML_VIEW', '0')):  # show the structure before running
    CSX_file = os.path.join(Sim_Path, 'CPW_IPML.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)

    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))
    
# Run the simulation
print('mode = %s, N = %d, sides = %s, dy = %.4f mm, Sim_Path = %s' % (MODE, N, SIDES, dy, Sim_Path))
FDTD.Run(Sim_Path, verbose=3, cleanup=False)

# ## Post-processing -- as in CPW_With_WG_Ports.py
Zl = _mp['Zl']  # from the mode solve, not hand-copied
Zw = _mp['Zw']
freq = np.linspace(f0 - fc, f0 + fc, 401)
port1.CalcPort(Sim_Path, freq, ref_impedance=Zw, ZL=Zl)
port2.CalcPort(Sim_Path, freq, ref_impedance=Zw, ZL=Zl)
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
for f in (1e9, 1.5e9, 2e9, 2.5e9, 3e9):
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
    plt.title('CPW, %s, N = %d, sides %s' % (MODE, N, SIDES))
    plt.show()
