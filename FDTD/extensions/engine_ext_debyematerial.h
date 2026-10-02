/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_DEBYEMATERIAL_H
#define ENGINE_EXT_DEBYEMATERIAL_H

#include "engine_ext_dispersive.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include <vector>

class Operator_Ext_DebyeMaterial;

//! Engine extension for an n-term Debye dispersive material
/*!
  The trapezoidal update of the pole capacitor voltage needs both the new and the
  old cell voltage, V^(n+1) + V^n. The main update overwrites the voltage in
  place, so V^n exists only before it and V^(n+1) only after it, and the update
  is split over the two hooks that straddle it:

    DoPreVoltageUpdates    (V is V^n, the only place it exists)
        h_k    = c1_k W_k + d_k V^n
        g     += h_k - W_k
        W_k    = h_k
    main voltage update    (V becomes V_raw)
    DoPostVoltageUpdates   (V_raw visible, V^n gone)
        V^(n+1) = inv (V_raw - g);  volt_ADE[0] = V_raw - V^(n+1)
        W_k   += d_k V^(n+1)
    Apply2Voltages          (inherited from Engine_Ext_Dispersive)
        V     -= volt_ADE[0]

  v_relax_ADE and v_drive_ADE play the roles v_int_ADE and v_ext_ADE play in
  Engine_Ext_LorentzMaterial: one multiplies the extension's own state, the other
  the cell voltage. This solves
  V^(n+1) = V_raw - sum_k (W_k^(n+1) - W_k^n) for V^(n+1) exactly,
  the implicit term being scalar per cell. W_k = r_k V_Ck is the voltage
  decrement pole k contributes, so the cell's total decrement is the plain sum
  of the W increments and no per pole r_k is needed at run time.
  */
class Engine_Ext_DebyeMaterial : public Engine_Ext_Dispersive
{
public:
	Engine_Ext_DebyeMaterial(Operator_Ext_DebyeMaterial* op_ext_debye);
	virtual ~Engine_Ext_DebyeMaterial();

	virtual void DoPreVoltageUpdates();
	virtual void DoPostVoltageUpdates();

	// Overridden to intercept the CUDA engine; the base Engine_Ext_Dispersive
	// implementation is used unchanged for CPU engines.
	virtual void Apply2Voltages();

#if WITH_CUDA
	// On-device port of the three voltage hooks above, captured into the
	// per-timestep CUDA graph like the Lorentz port. One entry per active cell,
	// carrying its three implicit-solve factors; the per-pole coefficients and
	// states live in flat device arrays indexed [pole][cell*3 + component].
	// Every pole of a cell is handled by the thread that owns the cell, so no
	// cross-thread writes and no atomics -- the same arithmetic, in the same
	// order, as the host loops.
	struct deb_cell {
		int x, y, z;
		FDTD_FLOAT solve[3];
	};
	virtual void SetEngine(Engine* eng);
	virtual bool IsCUDACapable() const {return true;}
#endif

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng);

	Operator_Ext_DebyeMaterial* m_Op_Ext_Deb;

#if WITH_CUDA
	void DoPreVoltageUpdatesCuda(Engine_cuda* eng);
	void DoPostVoltageUpdatesCuda(Engine_cuda* eng);
	void Apply2VoltagesCuda(Engine_cuda* eng);
	void FreeCuda();
	int PackCells(int x_lo, int x_hi, int x_shift, std::vector<deb_cell>& cells,
		std::vector<FDTD_FLOAT>& relax, std::vector<FDTD_FLOAT>& drive);

	int m_cuda_P = 0;                // poles
	// single GPU
	int m_cuda_N = 0;                // active cells
	deb_cell*   d_cells = NULL;
	FDTD_FLOAT* d_relax = NULL;      // [P][N*3]  (2 tau - dT)/(2 tau + dT)
	FDTD_FLOAT* d_drive = NULL;      // [P][N*3]  C vi/(2 tau + dT)
	FDTD_FLOAT* d_W     = NULL;      // [P][N*3]  pole state W_k
	FDTD_FLOAT* d_pre   = NULL;      // [N*3]     part of the correction V^n fixes
	FDTD_FLOAT* d_ade   = NULL;      // [N*3]     what Apply2Voltages subtracts

	// multi-GPU: cells partitioned by owning slab, x made slab-local incl. the
	// ghost offset -- the update is cell-local, as for the Lorentz port.
	void SetEngineMg(class Engine_cuda_mgpu* mg);
	std::vector<int> mg_dev, mg_N;
	std::vector<deb_cell*>   mg_cells;
	std::vector<FDTD_FLOAT*> mg_relax, mg_drive, mg_W, mg_pre, mg_ade;
#endif

	//! The one state per pole: W_k = r_k V_Ck, the voltage pole k currently takes
	//! off the cell. Array setup: volt_pole_ADE[N_poles][direction][index]
	FDTD_FLOAT ***volt_pole_ADE;

	//! The part of this step's correction that V^n already fixes, summed over the
	//! poles in DoPreVoltageUpdates() and consumed in DoPostVoltageUpdates(); it
	//! exists only because the main update destroys V^n.
	//! Array setup: volt_pre_ADE[direction][index]
	FDTD_FLOAT **volt_pre_ADE;
};

#endif // ENGINE_EXT_DEBYEMATERIAL_H
