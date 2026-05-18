/*
*	Copyright (C) 2025
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef OPERATOR_EXT_MODALFDTD_H
#define OPERATOR_EXT_MODALFDTD_H

#include "FDTD/operator.h"
#include "operator_extension.h"

class CSPropModeAbsorb;
class CSPrimitives;

// ---------------------------------------------------------------------------
// Per-mode 1-D Yee absorber (Luo & Chen 2007), Klein-Gordon transparent
// boundary coupling.
//
// For a waveguide modal amplitude a(z, t), the wave equation along the
// propagation axis is:
//      d2a/dt2 = c2 d2a/dz2 - omega_c2 a
// where omega_c = 2 pi f_cutoff. Above cutoff (omega > omega_c) this gives
// propagating modal waves with the correct dispersion beta = sqrt(omega2/c2
// - kc2); below cutoff (omega < omega_c) it gives evanescent decay with
// alpha = sqrt(kc2 - omega2/c2). ONE scheme covers both regimes.
//
// The 1-D scheme runs on a grid with the SAME dt and dz as the 3-D Yee
// scheme along the propagation axis. The 1-D dispersion thus EXACTLY
// matches the 3-D modal dispersion at every frequency, so waves crossing
// the port plane into the 1-D extension propagate with no impedance
// mismatch.
//
// Cell mapping (for direction = +1, V/I forward = +z; sheet at E plane
// m_srcLine; structure/source on +z side of sheet, fictional 1-D
// extension on -z side; Yee staggering H[i] between E[i] and E[i+1]):
//
//   1-D index    3-D H position           role
//   ---------    -----------------        --------------------------------
//   a[0]         H[m_srcLine]             INSIDE WG (between sheet and
//                                         source; sampled each step)
//   a[1]         H[m_srcLine - 1]         OUTSIDE WG, first extension
//                                         cell (override target)
//   a[2..N-1]    (1-D internal cells, no 3-D mapping)
//
// COUPLING (transparent boundary, NOT subtract-delta correction):
//   Each timestep in Apply2Current (after 3-D H update):
//
//   1. Project 3-D H at H-position INSIDE the WG (H[m_srcLine - 1] for dir=+1)
//      onto h_mode -> Hmm_in.  This is the wave amplitude entering the
//      1-D extension.
//
//   2. Drive 1-D's a[0] = Hmm_in (Dirichlet boundary; a[0] is not updated
//      by the leapfrog since that would need a[-1] which doesn't exist).
//
//   3. Update 1-D leapfrog interior:
//      a_new[i] = c_centre * a[i] + c_neigh * (a[i+1] + a[i-1]) - a_prev[i]
//      for i = 1..N-2.
//
//   4. Far-end Mur 1st-order ABC at a_new[N-1].
//
//   5. Override 3-D H at H-position OUTSIDE the WG (H[m_srcLine] for dir=+1)
//      with a_new[1] times the h_mode pattern.  The next 3-D E update at
//      port plane reads this overridden H via curl(H), making the WG
//      appear infinite to the 3-D simulation.
//
//   6. Shift 1-D state: a_prev <- a, a <- a_new.
//
// Why this works: the 3-D simulation never "sees" a boundary -- it just
// reads H values one cell beyond port plane and computes curl(H) as
// usual.  The 1-D scheme provides those H values such that the modal
// wave continues with the correct dispersion outward (no reflection).
//
// Why my previous "subtract delta from 3-D" attempt failed: that approach
// MODIFIES the 3-D E field after the fact, with a "predicted" value that's
// 0 at startup (1-D state empty) -> kills the source.  The transparent
// boundary doesn't modify 3-D fields directly; it just provides H values.
// ---------------------------------------------------------------------------

class Operator_Ext_ModalFDTD : public Operator_Extension
{
	friend class Engine_Ext_ModalFDTD;
public:
	Operator_Ext_ModalFDTD(Operator* op);
	virtual ~Operator_Ext_ModalFDTD();

	bool SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop);

	virtual Operator_Extension* Clone(Operator* op);
	virtual bool BuildExtension();
	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const { return std::string("Modal FDTD Extension"); }
	virtual void ShowStat(std::ostream& ostr) const;

protected:
	Operator_Ext_ModalFDTD(Operator* op, Operator_Ext_ModalFDTD* op_ext);
	void Initialize();

	CSPropModeAbsorb* m_prop;
	CSPrimitives*     m_prim;

	int m_ny, m_nyP, m_nyPP;     // propagation and transverse axes
	int m_dir;                    // +1 (absorbs +z waves) or -1

	// Sheet plane index in the 3-D mesh along m_ny (port plane)
	unsigned int m_srcLine;

	// Cross-section bounds in the 3-D mesh
	unsigned int m_start[3], m_stop[3];

	// Cross-section grid sizes (E primary mesh, H dual mesh).
	unsigned int m_numLines_E[2];
	unsigned int m_numLines_H[2];

	// 3-D H plane indices along m_ny:
	//   m_H_in_line:   inside WG (drive 1-D from here).  For dir=+1: m_srcLine-1.
	//   m_H_override_line: outside WG (write 1-D output here).  For dir=+1: m_srcLine.
	unsigned int m_H_in_line;
	unsigned int m_H_override_line;

	// Pre-computed mode-projection weights at the port plane.
	// Index 0 -> nyP component, index 1 -> nyPP component.
	double** m_E_OverlapW[2];     // for projecting transverse E onto e_mode (kept for diagnostics)
	double** m_E_SubtractW[2];    // not used by the transparent-boundary kernel

	// H-mode projection / reconstruction weights at the H planes.
	double** m_H_OverlapW[2];     // for projecting 3-D H onto h_mode at m_H_in_line
	double** m_H_SubtractW[2];    // for reconstructing 3-D H from a[1] at m_H_override_line

	// 1-D extension parameters
	unsigned int m_N1D;           // number of 1-D cells (configurable, default 1000)
	double       m_dz;            // 1-D cell spacing (= 3-D delta along m_ny at port)
	double       m_dt;            // simulation timestep
	double       m_kc;            // modal cutoff wavenumber [rad/m]
	double       m_omega_c;       // = m_kc * c

	// Pre-computed leapfrog coefficients
	//   c_centre = 2 (1 - r2 - 0.5 * (omega_c * dt)2)
	//   c_neigh  = r2                                       r = c * dt / dz
	double       m_c_centre;
	double       m_c_neigh;

	// 1-D Mur ABC coefficient at far end:  (rate - 1) / (rate + 1),
	// rate = c * dt / dz.
	double       m_mur_coef;
};

#endif // OPERATOR_EXT_MODALFDTD_H
