"""
Coaxial waveguide with modal absorber ports (function-expression based)

Tests the waveguide modal absorber extension on a coaxial TEM line.
Based on Coax_W_WG_Ports.py but uses function expressions instead of
mode CSV files, and uses absorb_layers for boundary absorption.

(c) 2025 Gadi Lahav <gadi@rfwithcare.com>

"""

# ## Import Libraries
import os, tempfile
from pylab import *

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

# ## Configuration
# Set excitation bandwidth:
#   narrowband: f0=2.5e9, fc=1e9  (baseline comparison with original script)
#   wideband:   f0=0,     fc=5e9  (DC content — the problematic case)
f0 = 2.5e9   # center frequency
fc = 1e9     # 20 dB corner frequency

Sim_Path = os.path.join(tempfile.gettempdir(), 'Test_Coax_Modal_Absorbers')

post_proc_only = False

# ## General parameter setup
# Coax geometry
coax_D = 2               # outer conductor inner diameter [mm]
coax_shield_thick = 0.15  # shield thickness [mm]
coax_wire_D = 0.5         # inner conductor diameter [mm]
coax_L = 25               # length [mm]

teflon_epsR = 2.5

mesh_res = 0.5
Airbox_Add = 0
unit_res = 1e-3  # mm

# Known coax parameters (from original script)
kz = np.array([82.84554871])
Zw = np.array([238.26517157])
Zl = np.array([52.43928218])

# TEM mode functions (parser vars: x,y,z,rho,a,r,t)
# Masked to be nonzero only in the dielectric gap between inner and outer
# conductors.  Coordinates are in mesh units (mm).
ri = coax_wire_D / 2   # inner conductor radius [mm]
ro = coax_D / 2        # outer conductor inner radius [mm]
# fparser comparison operators: (rho>val) returns 1 or 0
# Note: fparser if() treats any nonzero as true, not just positive.
mask = '(rho>{ri})*(rho<{ro})/rho'.format(ri=ri, ro=ro)
E_func = ['cos(a)*({m})'.format(m=mask), 'sin(a)*({m})'.format(m=mask), '0']
H_func = ['-sin(a)*({m})'.format(m=mask), 'cos(a)*({m})'.format(m=mask), '0']

# size of the simulation box
SimBox = np.array([
            -(coax_D * 0.5 + coax_shield_thick + Airbox_Add),
            (coax_D * 0.5 + coax_shield_thick + Airbox_Add),
            -(coax_D * 0.5 + coax_shield_thick + Airbox_Add),
            (coax_D * 0.5 + coax_shield_thick + Airbox_Add),
            -Airbox_Add,
            coax_L + Airbox_Add])

# ## FDTD setup
FDTD = openEMS(NrTS=300000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
# PEC everywhere — modal absorbers handle z-termination,
# x/y faces are enclosed by the PEC shield anyway
FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit_res)
mesh_res = ((C0 / (f0 + fc)) / unit_res) / 100

# ## Generate properties, primitives and mesh-grid
# initialize the mesh with the "air-box" dimensions
mesh.AddLine('x', SimBox[0:2])
mesh.AddLine('y', SimBox[2:4])
mesh.AddLine('z', SimBox[4:6])

# create center wire
line = CSX.AddMetal('Wire_Inner')
start = [0.0, 0.0, 0.0]
stop = [0.0, 0.0, coax_L]
line.AddCylinder(priority=10, start=start, stop=stop, radius=coax_wire_D * 0.5)
mesh.AddLine('x', np.linspace(start[0] - coax_wire_D * 0.5, stop[0] + coax_wire_D * 0.5, 6).tolist())
mesh.AddLine('y', np.linspace(start[1] - coax_wire_D * 0.5, stop[1] + coax_wire_D * 0.5, 6).tolist())
mesh.AddLine('z', [start[2], stop[2]])

# create Outer Shield
shield = CSX.AddMetal('Shield_Outer')
start = [0.0, 0.0, 0.0]
stop = [0.0, 0.0, coax_L]
shield.AddCylindricalShell(priority=10, start=start, stop=stop, radius=(coax_D + coax_shield_thick) * 0.5, shell_width=coax_shield_thick)
rad = (coax_D + coax_shield_thick) * 0.5
hthick = coax_shield_thick * 0.5
mesh.AddLine('x',
                np.linspace(start[0] - (rad + hthick), start[0] - (rad - hthick), 4).tolist() +
                np.linspace(stop[0] + (rad - hthick), stop[0] + (rad + hthick), 4).tolist())
mesh.AddLine('y',
                np.linspace(start[1] - (rad + hthick), start[1] - (rad - hthick), 4).tolist() +
                np.linspace(stop[1] + (rad - hthick), stop[1] + (rad + hthick), 4).tolist())
mesh.AddLine('z', [start[2], stop[2]])

# Create teflon fill
teflon = CSX.AddMaterial('PTFE', epsilon=teflon_epsR)
start = [0.0, 0.0, 0.0]
stop = [0.0, 0.0, coax_L]
teflon.AddCylindricalShell(priority=8, start=start, stop=stop, radius=(coax_wire_D + coax_D) * 0.25, shell_width=(coax_D - coax_wire_D) * 0.5)
rad = (coax_wire_D + coax_D) * 0.25
hthick = (coax_D - coax_wire_D) * 0.25
mesh.AddLine('x',
                np.linspace(start[0] - (rad + hthick), start[0] - (rad - hthick), 12).tolist() +
                np.linspace(stop[0] + (rad - hthick), stop[0] + (rad + hthick), 12).tolist())
mesh.AddLine('y',
                np.linspace(start[1] - (rad + hthick), start[1] - (rad - hthick), 12).tolist() +
                np.linspace(stop[1] + (rad - hthick), stop[1] + (rad + hthick), 12).tolist())
mesh.AddLine('z', [start[2], stop[2]])

# Add dense mesh lines close to ports
mesh.AddLine('z', np.array([0.25, 0.5, 0.8, 1]) * mesh_res)
mesh.AddLine('z', coax_L - np.array([0.25, 0.5, 0.8, 1]) * mesh_res)

mesh.SmoothMeshLines('all', mesh_res, 1.25)

# Find start/stop mesh lines for port placement
Zz = mesh.GetLines('z')
idxPort1 = (np.where(Zz == 0.0)[0] + 1).item(0)
idxPort2 = (np.where(Zz == coax_L)[0] - 1).item(0)

# ## Port setup — function-expression TEM mode with modal absorbers
port_extent = coax_D * 0.5 + coax_shield_thick

# Port placement: same as original Coax_W_WG_Ports.py.
# Source at idxPort1, measurement at idxPort1+1 (= same as absorber E-plane).
# The absorber modifies the field at the measurement plane, removing the
# backward component — the probe records the post-absorption field.
n_abs = 1  # absorb_layers

start = [-port_extent, -port_extent, Zz.item(idxPort1 + 0)]
stop  = [ port_extent,  port_extent, Zz.item(idxPort1 + 1)]
port1 = FDTD.AddWaveGuidePort(1, start, stop, 'z',
    E_func=E_func, H_func=H_func, kc=0.0,
    excite=1, excite_type=0, absorb_layers=n_abs,
    ref_index=np.sqrt(teflon_epsR), wave_impedance=Zw[0])

start = [-port_extent, -port_extent, Zz.item(idxPort2 - 0)]
stop  = [ port_extent,  port_extent, Zz.item(idxPort2 - 1)]
port2 = FDTD.AddWaveGuidePort(2, start, stop, 'z',
    E_func=E_func, H_func=H_func, kc=0.0,
    excite=0, excite_type=0, absorb_layers=n_abs,
    ref_index=np.sqrt(teflon_epsR), wave_impedance=Zw[0])

# ## Run the simulation
if 1:  # write XML for debugging
    CSX_file = os.path.join(Sim_Path, 'coax_modal_absorbers.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)

if not post_proc_only:
    FDTD.Run(Sim_Path, verbose=0, cleanup=False)

# ## Post-processing and plotting
f = np.linspace(max(1e9, f0 - fc), f0 + fc, 401)
port1.CalcPort(Sim_Path, f, ref_impedance=Zw, ZL=50)
port2.CalcPort(Sim_Path, f, ref_impedance=Zw, ZL=50)
s11 = port1.uf_ref / port1.uf_inc
s21 = port2.uf_ref / port1.uf_inc

s11_dB = 20.0 * np.log10(np.abs(s11))
s21_dB = 20.0 * np.log10(np.abs(s21))

figure()
plot(f / 1e9, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
plot(f / 1e9, s21_dB, 'b-', linewidth=2, label='$S_{21}$')
grid()
legend()
ylabel('S-Parameter (dB)')
xlabel('Frequency (GHz)')
title('Coax Modal Absorber: f0={:.1f}GHz, fc={:.1f}GHz'.format(f0/1e9, fc/1e9))

# Save figure for remote execution
fig_path = os.path.join(Sim_Path, 'Coax_Modal_Absorbers_Sparam.png')
savefig(fig_path, dpi=150)
print('Figure saved to: {}'.format(fig_path))

show()
