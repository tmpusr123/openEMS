/*
*	Copyright (C) 2025
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

// Per-mode 1-D Yee absorber (Luo-Chen 2007) with TRANSPARENT BOUNDARY.
// See operator header for the full algorithm.  Summary:
//
// Per timestep in Apply2Current (after 3-D H update):
//   1. Project 3-D H at m_H_in_line (inside-WG side of absorber sheet)
//      onto h_mode -> Hmm_in.  This is the wave amplitude entering the
//      1-D extension at the current 3-D time t_k.
//   2. Run 1-D leapfrog interior for i = 1..N-2.  The stencil is
//      centered at m_a (= a^{k-1}, last iteration's produced step,
//      now post-swap), advancing one step to m_a_new (= a^k).  m_a[0]
//      already holds the previous-step Hmm from the prior iteration's
//      step 4 — DO NOT overwrite it here or the stencil times mix.
//   3. Far-end Mur 1st-order ABC at i = N-1.
//   4. Set m_a_new[0] = Hmm_in (Dirichlet drive at the produced step).
//      After the swap this becomes m_a[0] for the next iteration's
//      centered stencil.
//   5. Override 3-D H at m_H_override_line with m_a_new[1] * h_mode
//      pattern.  m_a_new[1] is at time t_k, matching the just-updated
//      3-D H, so the next 3-D E update reads a transparent boundary.
//   6. Shift state: m_a_prev <- m_a, m_a <- m_a_new.

#include "engine_ext_modalfdtd.h"
#include "operator_ext_modalfdtd.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"

#include <cstring>
#include <algorithm>

Engine_Ext_ModalFDTD::Engine_Ext_ModalFDTD(Operator_Ext_ModalFDTD* op_ext) :
	Engine_Extension(op_ext)
{
	m_Op_MF = op_ext;

	m_ny   = m_Op_MF->m_ny;
	m_nyP  = m_Op_MF->m_nyP;
	m_nyPP = m_Op_MF->m_nyPP;
	m_dir  = m_Op_MF->m_dir;

	for (int i = 0; i < 3; ++i)
		m_posStart[i] = m_Op_MF->m_start[i];
	m_srcLine         = m_Op_MF->m_srcLine;
	m_H_in_line       = m_Op_MF->m_H_in_line;
	m_H_override_line = m_Op_MF->m_H_override_line;

	m_numLines_E[0] = m_Op_MF->m_numLines_E[0];
	m_numLines_E[1] = m_Op_MF->m_numLines_E[1];
	m_numLines_H[0] = m_Op_MF->m_numLines_H[0];
	m_numLines_H[1] = m_Op_MF->m_numLines_H[1];

	m_E_OverlapW[0]  = m_Op_MF->m_E_OverlapW[0];
	m_E_OverlapW[1]  = m_Op_MF->m_E_OverlapW[1];
	m_E_SubtractW[0] = m_Op_MF->m_E_SubtractW[0];
	m_E_SubtractW[1] = m_Op_MF->m_E_SubtractW[1];
	m_H_OverlapW[0]  = m_Op_MF->m_H_OverlapW[0];
	m_H_OverlapW[1]  = m_Op_MF->m_H_OverlapW[1];
	m_H_SubtractW[0] = m_Op_MF->m_H_SubtractW[0];
	m_H_SubtractW[1] = m_Op_MF->m_H_SubtractW[1];

	m_N1D       = m_Op_MF->m_N1D;
	m_c_centre  = m_Op_MF->m_c_centre;
	m_c_neigh   = m_Op_MF->m_c_neigh;
	m_mur_coef  = m_Op_MF->m_mur_coef;

	m_a      = new double[m_N1D];
	m_a_prev = new double[m_N1D];
	m_a_new  = new double[m_N1D];
	std::memset(m_a,      0, sizeof(double) * m_N1D);
	std::memset(m_a_prev, 0, sizeof(double) * m_N1D);
	std::memset(m_a_new,  0, sizeof(double) * m_N1D);

	// Run AFTER excitation sources are applied, so we override H values
	// AFTER any source contribution to H has been folded in.
	SetPriority(ENG_EXT_PRIO_EXCITATION - 100);
	SetNumberOfThreads(1);
}

Engine_Ext_ModalFDTD::~Engine_Ext_ModalFDTD()
{
	delete[] m_a;
	delete[] m_a_prev;
	delete[] m_a_new;
}

template <typename EngType>
void Engine_Ext_ModalFDTD::Apply2CurrentImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	// 1. Project 3-D H at m_H_in_line onto h_mode -> Hmm_in.
	unsigned int pos[3] = {0, 0, 0};
	pos[m_ny] = m_H_in_line;
	double Hmm_in = 0.0;
	for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			Hmm_in += eng->EngType::GetCurr(m_nyP,  pos) * m_H_OverlapW[0][posP][posPP];
			Hmm_in += eng->EngType::GetCurr(m_nyPP, pos) * m_H_OverlapW[1][posP][posPP];
		}
	}

	// 2. 1-D leapfrog interior for i = 1..N-2.
	//    Klein-Gordon centered finite difference, stencil centered at the
	//    "current center" time which is m_a (= a^{k-1} at iteration k —
	//    last iteration's m_a_new, post-swap).  Do NOT touch m_a[0] before
	//    the leapfrog: it already carries the previous step's projected
	//    Hmm (set as m_a_new[0] at the END of last iteration), which is
	//    exactly the value the centered stencil expects.
	for (unsigned int i = 1; i + 1 < m_N1D; ++i)
	{
		m_a_new[i] = m_c_centre * m_a[i]
		           + m_c_neigh  * (m_a[i+1] + m_a[i-1])
		           - m_a_prev[i];
	}

	// 3. Far-end Mur 1st-order ABC at i = N-1:
	//    a^{n+1}[N-1] = a^n[N-2] + mur_coef * (a^{n+1}[N-2] - a^n[N-1])
	m_a_new[m_N1D - 1] = m_a[m_N1D - 2]
	                   + m_mur_coef * (m_a_new[m_N1D - 2] - m_a[m_N1D - 1]);

	// 4. Drive Dirichlet at i=0: m_a_new[0] is the produced "current step"
	//    boundary value at time t_k.  After the swap below it becomes
	//    m_a[0] for the NEXT iteration's centered stencil.
	m_a_new[0] = Hmm_in;

	// 5. Override 3-D H across the empty mesh past the absorber sheet
	//    using a_new[i] * h_mode pattern.  The 1-D's i=1..K maps to 3-D
	//    H planes deeper into the extension (decreasing line index for
	//    dir=+1, increasing for dir=-1).  Overriding multiple cells
	//    avoids the sharp seam at the override boundary that reflects
	//    waves back into the structure when only one cell is replaced.
	//    The seam still exists at i=K but is K cells away from the
	//    absorber sheet, so the wave passes through K cells of consistent
	//    1-D-driven evolution before encountering the natural-Yee region.
	int dir_sign = (m_dir > 0) ? -1 : +1;  // toward extension
	unsigned int N_override = (m_N1D > 21) ? 20 : (m_N1D - 1);
	for (unsigned int i = 1; i <= N_override; ++i)
	{
		int line = (int)m_H_override_line + dir_sign * (int)(i - 1);
		if (line < 0) break;
		pos[m_ny] = (unsigned int)line;
		double a_i = m_a_new[i];
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				eng->EngType::SetCurr(m_nyP,  pos,
					a_i * m_H_SubtractW[0][posP][posPP]);
				eng->EngType::SetCurr(m_nyPP, pos,
					a_i * m_H_SubtractW[1][posP][posPP]);
			}
		}
	}

	// 6. Shift state: a_prev <- a, a <- a_new (pointer swap).
	std::swap(m_a_prev, m_a);
	std::swap(m_a,      m_a_new);
}

void Engine_Ext_ModalFDTD::Apply2Current(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, threadID);
}
