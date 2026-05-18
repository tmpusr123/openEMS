/*
*	Copyright (C) 2025
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_EXT_MODALFDTD_H
#define ENGINE_EXT_MODALFDTD_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_ModalFDTD;

// Per-mode 1-D Yee absorber (Luo-Chen 2007) with TRANSPARENT BOUNDARY
// coupling.  Runs a Klein-Gordon scheme alongside the 3-D simulation;
// each timestep, projects 3-D H onto h_mode at the WG side of the
// port plane to drive the 1-D, then OVERRIDES 3-D H one cell beyond
// the port plane with the 1-D's reconstruction.  The next 3-D E
// update reads the overridden H via curl(H), making the WG appear
// infinite.

class Engine_Ext_ModalFDTD : public Engine_Extension
{
public:
	Engine_Ext_ModalFDTD(Operator_Ext_ModalFDTD* op_ext);
	virtual ~Engine_Ext_ModalFDTD();

	// Hook AFTER 3-D current (H) update — H is just-updated; we override
	// the H values at m_H_override_line that the NEXT E update will read.
	virtual void Apply2Current() { Engine_Ext_ModalFDTD::Apply2Current(0); }
	virtual void Apply2Current(int threadID);

protected:
	Operator_Ext_ModalFDTD* m_Op_MF;

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, int threadID);

	// Cached operator state
	int m_ny, m_nyP, m_nyPP;
	int m_dir;
	unsigned int m_posStart[3];
	unsigned int m_srcLine;
	unsigned int m_H_in_line;
	unsigned int m_H_override_line;
	unsigned int m_numLines_E[2];   // kept for completeness (unused by this kernel)
	unsigned int m_numLines_H[2];

	double** m_E_OverlapW[2];   // unused (kept for ABI compatibility)
	double** m_E_SubtractW[2];  // unused
	double** m_H_OverlapW[2];   // project 3-D H at m_H_in_line -> Hmm
	double** m_H_SubtractW[2];  // reconstruct H at m_H_override_line from a[1]

	// 1-D scheme state
	unsigned int m_N1D;
	double m_c_centre;          // 2 * (1 - r^2 - 0.5*(omega_c*dt)^2)
	double m_c_neigh;            // r^2,  r = c * dt / dz
	double m_mur_coef;           // (rate - 1) / (rate + 1)
	double* m_a;                 // current step  a^n[i], i = 0..N-1
	double* m_a_prev;            // previous step a^{n-1}[i]
	double* m_a_new;             // scratch       a^{n+1}[i]
};

#endif // ENGINE_EXT_MODALFDTD_H
