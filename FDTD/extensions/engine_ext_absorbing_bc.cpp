/*
*	Copyright (C) 2023-2025 Gadi Lahav (gadi@rfwithcare.com), Thorsten Liebig (Thorsten.Liebig@gmx.de)
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


/* This version of absorbing boundary conditions is based on this article:
 *
 * Betz, Vaughn Timothy, and R. Mittra. "Absorbing boundary conditions for the finite-difference time-domain analysis of guided-wave structures." Coordinated Science Laboratory Report no. UILU-ENG-93-2243 (1993).
 *
 * After some trial and error, it was discovered that the simplest and most efficient implementations
 * are:
 * 1. Mur first order boundary conditions
 * 2. First order Mur with "super-absorption".
 * Later I discovered that the latter is equivalent to the so-called "Surface impedance boundary
 * conditions" (SIBC).
 */


#include "engine_ext_absorbing_bc.h"
#include "operator_ext_absorbing_bc.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/array_ops.h"
#include "tools/useful.h"
#include "operator_ext_excitation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
	// Runtime A/B test knobs — read once on first call to keep hot path tight.
	// Export CPML_DISABLE_PSI=1 to zero the ψ feedback (κ-only PML). Export
	// CPML_DISABLE_KAP=1 to drop the (1/κ−1) stretch term (ψ-only).  These
	// help isolate which half of the CFS-CPML correction is misbehaving.
	struct CpmlDiagFlags
	{
		bool disable_psi;
		bool disable_kap;
		bool warn_nonfinite;
		CpmlDiagFlags()
		{
			const char* p = std::getenv("CPML_DISABLE_PSI");
			disable_psi = (p != nullptr && p[0] == '1');
			const char* k = std::getenv("CPML_DISABLE_KAP");
			disable_kap = (k != nullptr && k[0] == '1');
			const char* w = std::getenv("CPML_WARN_NONFINITE");
			warn_nonfinite = (w == nullptr || w[0] != '0'); // default ON
			std::cerr << "[CPML-FLAGS] disable_psi=" << (disable_psi ? 1 : 0)
			          << " disable_kap=" << (disable_kap ? 1 : 0)
			          << " warn_nonfinite=" << (warn_nonfinite ? 1 : 0)
			          << std::endl;
		}
	};
	const CpmlDiagFlags& cpml_flags()
	{
		static CpmlDiagFlags f;
		return f;
	}
}

Engine_Ext_Absorbing_BC::Engine_Ext_Absorbing_BC(Operator_Ext_Absorbing_BC* op_ext) :
	Engine_Extension(op_ext),
	m_K1_nyP(op_ext->m_K1_nyP),
	m_K1_nyPP(op_ext->m_K1_nyPP),
	m_K2_nyP(op_ext->m_K2_nyP),
	m_K2_nyPP(op_ext->m_K2_nyPP)
{

	m_Op_ABC = op_ext;
	m_ABCtype = int(m_Op_ABC->m_ABCtype);

	for (unsigned int dimIdx = 0 ; dimIdx < 3 ; dimIdx++)
	{
		m_posStart[dimIdx] = m_Op_ABC->m_sheetX0[dimIdx];
		m_posStop[dimIdx] = m_Op_ABC->m_sheetX1[dimIdx];
	}

	m_ny	= m_Op_ABC->m_ny;
	m_nyP 	= m_Op_ABC->m_nyP;
	m_nyPP 	= m_Op_ABC->m_nyPP;

	m_numLines[0] = m_Op_ABC->m_numLines[0];
	m_numLines[1] = m_Op_ABC->m_numLines[1];

	bool normalSignPositive = m_Op_ABC->m_normalSignPositive;

	m_start_TS = 0;

	// Initialize shifted position for V
	m_pos_ny0_shift_V = m_posStart[m_ny] + (normalSignPositive  ? 1 : -1);

	// Initialize shifted position for I. Different for super-absorption
	m_pos_ny0_I = m_posStart[m_ny] + (normalSignPositive ? 0 : -1);
	m_pos_ny0_shift_I = m_posStart[m_ny] + (normalSignPositive ? 1 : -2);

	m_V_nyP.Init("volt_nyP",m_numLines);
	m_V_nyPP.Init("volt_nyPP",m_numLines);
	m_I_nyP.Init("curr_nyP",m_numLines);
	m_I_nyPP.Init("curr_nyPP",m_numLines);

	m_pmlDepth    = m_Op_ABC->m_pmlDepth;
	m_pmlStepSign = m_Op_ABC->m_pmlStepSign;

	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
	{
		// 3D state, indexed (i_nyP, j_nyPP, k_depth).
		unsigned int psi_extent[3] = {m_numLines[0], m_numLines[1], m_pmlDepth};
		m_psi_V_nyP .Init("psi_V_nyP" , psi_extent);
		m_psi_V_nyPP.Init("psi_V_nyPP", psi_extent);
		m_psi_I_nyP .Init("psi_I_nyP" , psi_extent);
		m_psi_I_nyPP.Init("psi_I_nyPP", psi_extent);
		// ArrayIJK::Init does not zero memory; explicit zero so the recursive
		// convolution accumulator starts cleanly.
		for (unsigned int i = 0; i < m_numLines[0]; ++i)
		for (unsigned int j = 0; j < m_numLines[1]; ++j)
		for (unsigned int k = 0; k < m_pmlDepth;   ++k)
		{
			m_psi_V_nyP (i, j, k) = 0;
			m_psi_V_nyPP(i, j, k) = 0;
			m_psi_I_nyP (i, j, k) = 0;
			m_psi_I_nyPP(i, j, k) = 0;
		}
	}

	// One thread per boundary
	SetNumberOfThreads(1);
}

Engine_Ext_Absorbing_BC::~Engine_Ext_Absorbing_BC()
{
}

void Engine_Ext_Absorbing_BC::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	// This command assigns the number of jobs (primitives) handled by each thread
	m_linesPerThread = AssignJobs2Threads(m_numLines[0],m_NrThreads,false);

	// Basically cumulative sum. Starting point of each thread.
	m_threadStartLine.resize(m_NrThreads,0);
	m_threadStartLine.at(0) = 0;
	for (size_t n = 1; n < m_linesPerThread.size(); ++n)
		m_threadStartLine.at(n) = m_threadStartLine.at(n - 1) + m_linesPerThread.at(n - 1);
}

// The first order Mur boundary condition is based on the following step:
// The field at time step n, location i, is u
// E(i,n + 1) = E(i + s,n) + K1*[E(i,n + 1) - E(i,n)]
// where s is the +-1 shift, depending on the direction, and
// K1 = (vp*Dt - Dx)/(vp*Dt + Dx)

template <typename EngType>
void Engine_Ext_Absorbing_BC::DoPreVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	// CPML does its work entirely in DoPostVoltageUpdates.
	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
		return;

	unsigned int pos[] = {0,0,0};
	unsigned int pos_shift[] = {0,0,0};

	pos[m_ny] = m_posStart[m_ny];
	pos_shift[m_ny] = m_pos_ny0_shift_V;
	for (unsigned int i = m_threadStartLine.at(threadID) ; i < (m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID)) ; i++)
	{
		// Store shifted location in this container
		pos_shift[m_nyP] = pos[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos_shift[m_nyPP] = pos[m_nyPP] = m_posStart[m_nyPP] + j;

			// E(i + s,n) - K1*E(i,n)
			m_V_nyP (i,j) = eng->EngType::GetVolt(m_nyP ,pos_shift) - m_Op_ABC->m_K1_nyP (i,j) * eng->EngType::GetVolt(m_nyP ,pos);
			m_V_nyPP(i,j) = eng->EngType::GetVolt(m_nyPP,pos_shift) - m_Op_ABC->m_K1_nyPP(i,j) * eng->EngType::GetVolt(m_nyPP,pos);
		}

	}

}

void Engine_Ext_Absorbing_BC::DoPreVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Absorbing_BC::DoPostVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
	{
		ApplyCPMLVoltageUpdateImpl<EngType>(eng, threadID);
		return;
	}

	unsigned int pos_shift[] = {0,0,0};

	pos_shift[m_ny] = m_pos_ny0_shift_V;
	for (unsigned int i = m_threadStartLine.at(threadID) ; i < (m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID)) ; i++)
	{
		// Store shifted location in this container
		pos_shift[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos_shift[m_nyPP] = m_posStart[m_nyPP] + j;

			// E(i + s,n) - K1*E(i,n) + K1*E(i + s,n) =
			// E(i + s,n) + [K1*E(i + s,n) - K1*E(i,n)]
			m_V_nyP (i,j) += m_K1_nyP (i,j) * eng->EngType::GetVolt(m_nyP ,pos_shift);
			m_V_nyPP(i,j) += m_K1_nyPP(i,j) * eng->EngType::GetVolt(m_nyPP,pos_shift);
		}

	}

}

void Engine_Ext_Absorbing_BC::DoPostVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Absorbing_BC::Apply2VoltagesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	// CPML applied its V correction in DoPostVoltageUpdates already.
	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
		return;

	unsigned int pos[] = {0,0,0};

	pos[m_ny] = m_posStart[m_ny];
	for (unsigned int i = m_threadStartLine.at(threadID) ; i < (m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID)) ; i++)
	{
		// Store shifted location in this container
		pos[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + j;

			// E(i,n + 1) = E(i + s,n) + [K1*E(i + s,n) - K1*E(i,n)]
			eng->EngType::SetVolt(m_nyP ,pos, m_V_nyP (i,j));
			eng->EngType::SetVolt(m_nyPP,pos, m_V_nyPP(i,j));
		}

	}
}

void Engine_Ext_Absorbing_BC::Apply2Voltages(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Absorbing_BC::DoPreCurrentUpdatesImpl(EngType* eng, int threadID)
{

	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	// CPML does its work in DoPostCurrentUpdates.
	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
		return;

	unsigned int 	pos[] = {0,0,0},
					pos_shift[] = {0,0,0};


	// If this isn't the appropriate boundary type, move on to the next primitive
	if ((Operator_Ext_Absorbing_BC::ABCtype)(m_ABCtype) != Operator_Ext_Absorbing_BC::MUR_1ST_SA)
		return;

	// For magnetic field, -1, due to dual grid
	unsigned int numLines_1 = std::min(
		m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID),
		m_numLines[0] - 1
	);
	unsigned int numLine_0 = m_threadStartLine.at(threadID);

	pos[m_ny] = m_pos_ny0_I;
	pos_shift[m_ny] = m_pos_ny0_shift_I;
	for (unsigned int i = numLine_0 ; i < numLines_1 ; i++)
	{
		// Store shifted location in this container
		pos_shift[m_nyP] = pos[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < (m_numLines[1] - 1); j++)
		{
			pos_shift[m_nyPP] = pos[m_nyPP] = m_posStart[m_nyPP] + j;

			// H(i + s,n) - K1*H(i,n)
			m_I_nyP (i,j) = eng->EngType::GetCurr(m_nyP ,pos_shift) - m_K1_nyP (i,j)*eng->EngType::GetCurr(m_nyP ,pos);
			m_I_nyPP(i,j) = eng->EngType::GetCurr(m_nyPP,pos_shift) - m_K1_nyPP(i,j)*eng->EngType::GetCurr(m_nyPP,pos);
		}
	}

}

void Engine_Ext_Absorbing_BC::DoPreCurrentUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, threadID);
}

// Super-absorption:
//
// 1. Re-iterate the Mur B.C.
// Hsa(i,n + 1) = H(i + s,n) + K1*[H(i,n + 1) - H(i,n)]
//
// 2. Update the H(i,n + 1) as such
// H(i,n + 1) = (K2*Hsa(i,n + 1) + Hc(i,n + 1))/(K2 + 1)
// Where Hsa(i,n + 1) is the field calculated by the boundary condition, and
// Hc(i,n + 1) is the field calculated by the FDTD step.
// and K2 = vp*Dt/Dx

template <typename EngType>
void Engine_Ext_Absorbing_BC::DoPostCurrentUpdatesImpl(EngType* eng, int threadID)
{

	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
	{
		ApplyCPMLCurrentUpdateImpl<EngType>(eng, threadID);
		return;
	}

	unsigned int pos_shift[] = {0,0,0};

	// If this isn't the appropriate boundary type, move on to the next primitive
	if ((Operator_Ext_Absorbing_BC::ABCtype)(m_ABCtype) != Operator_Ext_Absorbing_BC::MUR_1ST_SA)
		return;

	// For magnetic field, -1, due to dual grid
	unsigned int numLine_1 = std::min<unsigned int>(
		m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID),
		m_numLines[0] - 1
	);
	unsigned int numLine_0 = m_threadStartLine.at(threadID);

	pos_shift[m_ny] = m_pos_ny0_shift_I;
	for (unsigned int i = numLine_0 ; i < numLine_1 ; i++)
	{
		// Store shifted location in this container
		pos_shift[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < (m_numLines[1] - 1); j++)
		{
			pos_shift[m_nyPP] = m_posStart[m_nyPP] + j;

			// H(i + s,n) + K1*[H(i,n + 1) - H(i,n)]
			m_I_nyP (i,j) += m_K1_nyP (i,j)*eng->EngType::GetCurr(m_nyP ,pos_shift);
			m_I_nyPP(i,j) += m_K1_nyPP(i,j)*eng->EngType::GetCurr(m_nyPP,pos_shift);

		}
	}
}

void Engine_Ext_Absorbing_BC::DoPostCurrentUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPostCurrentUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Absorbing_BC::Apply2CurrentImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;

	if (m_Eng==NULL) return;

	if (threadID >= m_NrThreads)
		return;

	// CPML applied its I correction in DoPostCurrentUpdates already.
	if (m_ABCtype == int(Operator_Ext_Absorbing_BC::CPML))
		return;

	unsigned int pos[] = {0,0,0};

	// If this isn't the appropriate boundary type, move on to the next primitive
	if ((Operator_Ext_Absorbing_BC::ABCtype)(m_ABCtype) != Operator_Ext_Absorbing_BC::MUR_1ST_SA)
		return;

	// For magnetic field, -1, due to dual grid
	unsigned int numLine_1 = std::min<unsigned int>(
		m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID),
		m_numLines[0] - 1
	);
	unsigned int numLine_0 = m_threadStartLine.at(threadID);

	pos[m_ny] = m_pos_ny0_I;
	for (unsigned int i = numLine_0 ; i < numLine_1 ; i++)
	{
		// Store shifted location in this container
		pos[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < (m_numLines[1] - 1); j++)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + j;

			// H(i + s,n) = (Hsa*K2 + Hc)/(1 + K2)
			eng->EngType::SetCurr(m_nyP ,pos, (m_I_nyP (i,j)*m_K2_nyP (i,j) + eng->EngType::GetCurr(m_nyP ,pos))/(m_K2_nyP (i,j) + 1.0));
			eng->EngType::SetCurr(m_nyPP,pos, (m_I_nyPP(i,j)*m_K2_nyPP(i,j) + eng->EngType::GetCurr(m_nyPP,pos))/(m_K2_nyPP(i,j) + 1.0));


		}
	}

}

void Engine_Ext_Absorbing_BC::Apply2Current(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, threadID);
}

// ---------------------------------------------------------------------------
// CFS-CPML strip update (kappa = 1 simplification).
//
// At this point in the time step:
//   * Volt has just been updated to V[n+1/2] using the standard FIT formula
//   * Curr is still at I[n] (the curl source the engine just used)
// We compute the m_ny-axis backward difference of I exactly as the engine
// did, push it through the recursive memory variable psi_V (one per cell per
// transverse component), and add psi_V to the freshly-updated V. The
// per-cell coefficients m_pml_cv_* already include sigma/(sigma+alpha)*(b-1)
// and the local vi with the correct sign baked in.
// ---------------------------------------------------------------------------

template <typename EngType>
void Engine_Ext_Absorbing_BC::ApplyCPMLVoltageUpdateImpl(EngType* eng, int threadID)
{
	const unsigned int i_start = m_threadStartLine.at(threadID);
	const unsigned int i_stop  = i_start + m_linesPerThread.at(threadID);

	// [CPML-DIAG] gating: print every PRINT_EVERY steps for the first
	// PRINT_EARLY_FIRST steps (early transient), then sparsely afterwards.
	// Diagnostic is per-thread; only thread 0 prints to avoid flooding.
	const unsigned int ts = m_Eng ? m_Eng->GetNumberOfTimesteps() : 0;
	const bool print_step = (threadID == 0) && (
	    (ts < 30) ||                       // first 30 steps verbatim
	    (ts < 300 && (ts % 10) == 0) ||    // every 10 up to 300
	    ((ts % 200) == 0));                // then every 200
	const unsigned int i_mid = m_numLines[0] / 2;
	const unsigned int j_mid = m_numLines[1] / 2;
	const CpmlDiagFlags& flags = cpml_flags();

	// Accumulators for this voltage half-step (PML region only).
	double sum_psi_V_sq = 0.0;
	double sum_V_sq     = 0.0;
	double sum_diff_sq  = 0.0;
	double max_psi_V    = 0.0;
	double max_V        = 0.0;

	unsigned int pos[3];
	unsigned int pos_back[3];

	for (unsigned int i = i_start; i < i_stop; ++i)
	{
		pos[m_nyP] = pos_back[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j < m_numLines[1]; ++j)
		{
			pos[m_nyPP] = pos_back[m_nyPP] = m_posStart[m_nyPP] + j;
			for (unsigned int k = 0; k < m_pmlDepth; ++k)
			{
				const int p_ny = (int)m_posStart[m_ny] + m_pmlStepSign * (int)k;
				pos[m_ny] = (unsigned int)p_ny;

				// Engine uses pos[m_ny]-1 as the backward neighbour, regardless
				// of which face the strip is on.  CPML must read the SAME diff
				// the engine used so the κ correction (1/κ-1)·diff coherently
				// modifies the engine's curl.
				if (p_ny <= 0)
					pos_back[m_ny] = (unsigned int)p_ny;
				else
					pos_back[m_ny] = (unsigned int)(p_ny - 1);

				const bool is_corner = (i == i_start && j == 0);
				const bool is_mid    = (i == i_mid && j == j_mid);
				const bool is_spot   = (is_corner || is_mid)
				                       && (k == 0 || k == m_pmlDepth - 1);

				// V[m_nyP] update: m_ny-curl term involves curr(m_nyPP, ...).
				const FDTD_FLOAT diff_P =
					eng->EngType::GetCurr(m_nyPP, pos)
					- eng->EngType::GetCurr(m_nyPP, pos_back);

				const FDTD_FLOAT V_P_before  = eng->EngType::GetVolt(m_nyP, pos);
				const FDTD_FLOAT psi_P_old   = m_psi_V_nyP(i, j, k);
				const FDTD_FLOAT psi_P_decay = (FDTD_FLOAT)m_Op_ABC->m_pml_b_z[k] * psi_P_old;
				const FDTD_FLOAT psi_P_drive = m_Op_ABC->m_pml_cv_nyP(i, j, k) * diff_P;
				const FDTD_FLOAT psi_P_new   = psi_P_decay + psi_P_drive;
				m_psi_V_nyP(i, j, k) = psi_P_new;

				const FDTD_FLOAT kap_term_P_raw = m_Op_ABC->m_pml_kapV_nyP(i, j, k) * diff_P;
				const FDTD_FLOAT kap_term_P  = flags.disable_kap ? (FDTD_FLOAT)0 : kap_term_P_raw;
				const FDTD_FLOAT psi_term_P  = flags.disable_psi ? (FDTD_FLOAT)0 : psi_P_new;
				const FDTD_FLOAT V_P_after   = V_P_before + kap_term_P + psi_term_P;
				eng->EngType::SetVolt(m_nyP, pos, V_P_after);

				// V[m_nyPP] update: m_ny-curl term involves curr(m_nyP, ...).
				const FDTD_FLOAT diff_PP =
					eng->EngType::GetCurr(m_nyP, pos)
					- eng->EngType::GetCurr(m_nyP, pos_back);

				const FDTD_FLOAT V_PP_before  = eng->EngType::GetVolt(m_nyPP, pos);
				const FDTD_FLOAT psi_PP_old   = m_psi_V_nyPP(i, j, k);
				const FDTD_FLOAT psi_PP_decay = (FDTD_FLOAT)m_Op_ABC->m_pml_b_z[k] * psi_PP_old;
				const FDTD_FLOAT psi_PP_drive = m_Op_ABC->m_pml_cv_nyPP(i, j, k) * diff_PP;
				const FDTD_FLOAT psi_PP_new   = psi_PP_decay + psi_PP_drive;
				m_psi_V_nyPP(i, j, k) = psi_PP_new;

				const FDTD_FLOAT kap_term_PP_raw = m_Op_ABC->m_pml_kapV_nyPP(i, j, k) * diff_PP;
				const FDTD_FLOAT kap_term_PP = flags.disable_kap ? (FDTD_FLOAT)0 : kap_term_PP_raw;
				const FDTD_FLOAT psi_term_PP = flags.disable_psi ? (FDTD_FLOAT)0 : psi_PP_new;
				const FDTD_FLOAT V_PP_after  = V_PP_before + kap_term_PP + psi_term_PP;
				eng->EngType::SetVolt(m_nyPP, pos, V_PP_after);

				// --- accumulators (per-thread slice, k spans full depth) ---
				sum_psi_V_sq += (double)psi_P_new  * (double)psi_P_new
				             +  (double)psi_PP_new * (double)psi_PP_new;
				sum_V_sq     += (double)V_P_after  * (double)V_P_after
				             +  (double)V_PP_after * (double)V_PP_after;
				sum_diff_sq  += (double)diff_P  * (double)diff_P
				             +  (double)diff_PP * (double)diff_PP;
				if (std::fabs((double)psi_P_new ) > max_psi_V) max_psi_V = std::fabs((double)psi_P_new);
				if (std::fabs((double)psi_PP_new) > max_psi_V) max_psi_V = std::fabs((double)psi_PP_new);
				if (std::fabs((double)V_P_after ) > max_V    ) max_V     = std::fabs((double)V_P_after);
				if (std::fabs((double)V_PP_after) > max_V    ) max_V     = std::fabs((double)V_PP_after);

				// --- per-cell spot-check print ---
				if (print_step && is_spot)
				{
					const char* loc = is_corner ? "C" : "M";
					const char* edg = (k == 0) ? "I" : "O";
					std::cerr << "[CPML-V] ts=" << ts
					          << " " << edg << "/" << loc
					          << " ijk=(" << i << "," << j << "," << k << ")"
					          << " pos_ny=" << pos[m_ny]
					          << " back_ny=" << pos_back[m_ny]
					          << " | dP=" << diff_P
					          << " psi_P[old=" << psi_P_old
					          << " dec=" << psi_P_decay
					          << " drv=" << psi_P_drive
					          << " new=" << psi_P_new << "]"
					          << " kap_P=" << kap_term_P
					          << " V_P[" << V_P_before << "->" << V_P_after << "]"
					          << " | dPP=" << diff_PP
					          << " psi_PP_new=" << psi_PP_new
					          << " V_PP[" << V_PP_before << "->" << V_PP_after << "]"
					          << std::endl;
				}
			}
		}
	}

	// --- per-step summary across this thread's PML region ---
	if (print_step)
	{
		std::cerr << "[CPML-V-SUM] ts=" << ts
		          << " thr=" << threadID
		          << " sum_psi2=" << sum_psi_V_sq
		          << " sum_V2=" << sum_V_sq
		          << " sum_dI2=" << sum_diff_sq
		          << " max_psi=" << max_psi_V
		          << " max_V=" << max_V
		          << std::endl;
	}

	// --- nonfinite watchdog (always on; cheap because we already have maxima) ---
	if (flags.warn_nonfinite && threadID == 0
	    && (!std::isfinite(max_psi_V) || !std::isfinite(max_V)))
	{
		std::cerr << "[CPML-V-NAN] ts=" << ts
		          << " max_psi=" << max_psi_V
		          << " max_V=" << max_V
		          << " (non-finite detected!)" << std::endl;
	}
}

template <typename EngType>
void Engine_Ext_Absorbing_BC::ApplyCPMLCurrentUpdateImpl(EngType* eng, int threadID)
{
	// Currents live on the dual grid: the engine uses pos[m_ny]+1 (forward
	// neighbour). At the trailing boundary the engine bounds the loop at
	// numLines-1, so we mirror that by skipping cells where pos[m_ny]+1 is
	// out of range.
	const unsigned int i_stop = std::min(
		m_threadStartLine.at(threadID) + m_linesPerThread.at(threadID),
		m_numLines[0] - 1);
	const unsigned int i_start = m_threadStartLine.at(threadID);

	const unsigned int n_lines_ny = m_Op_ABC->m_Op->GetNumberOfLines(m_ny, true);

	const unsigned int ts = m_Eng ? m_Eng->GetNumberOfTimesteps() : 0;
	const bool print_step = (threadID == 0) && (
	    (ts < 30) ||
	    (ts < 300 && (ts % 10) == 0) ||
	    ((ts % 200) == 0));
	const unsigned int i_mid = m_numLines[0] / 2;
	const unsigned int j_mid = m_numLines[1] / 2;
	const CpmlDiagFlags& flags = cpml_flags();

	double sum_psi_I_sq = 0.0;
	double sum_I_sq     = 0.0;
	double sum_diff_sq  = 0.0;
	double max_psi_I    = 0.0;
	double max_I        = 0.0;

	unsigned int pos[3];
	unsigned int pos_fwd[3];

	for (unsigned int i = i_start; i < i_stop; ++i)
	{
		pos[m_nyP] = pos_fwd[m_nyP] = m_posStart[m_nyP] + i;
		for (unsigned int j = 0; j + 1 < m_numLines[1]; ++j)
		{
			pos[m_nyPP] = pos_fwd[m_nyPP] = m_posStart[m_nyPP] + j;
			for (unsigned int k = 0; k < m_pmlDepth; ++k)
			{
				const int p_ny = (int)m_posStart[m_ny] + m_pmlStepSign * (int)k;
				pos[m_ny] = (unsigned int)p_ny;
				if (p_ny + 1 >= (int)n_lines_ny)
					pos_fwd[m_ny] = (unsigned int)p_ny;     // engine pins forward to self
				else
					pos_fwd[m_ny] = (unsigned int)(p_ny + 1);

				const bool is_corner = (i == i_start && j == 0);
				const bool is_mid    = (i == i_mid && j == j_mid);
				const bool is_spot   = (is_corner || is_mid)
				                       && (k == 0 || k == m_pmlDepth - 1);

				// I[m_nyP] m_ny-term involves volt(m_nyPP, pos+e_ny).
				const FDTD_FLOAT diff_P =
					eng->EngType::GetVolt(m_nyPP, pos_fwd)
					- eng->EngType::GetVolt(m_nyPP, pos);

				const FDTD_FLOAT I_P_before  = eng->EngType::GetCurr(m_nyP, pos);
				const FDTD_FLOAT psi_P_old   = m_psi_I_nyP(i, j, k);
				const FDTD_FLOAT psi_P_decay = (FDTD_FLOAT)m_Op_ABC->m_pml_b_z[k] * psi_P_old;
				const FDTD_FLOAT psi_P_drive = m_Op_ABC->m_pml_ci_nyP(i, j, k) * diff_P;
				const FDTD_FLOAT psi_P_new   = psi_P_decay + psi_P_drive;
				m_psi_I_nyP(i, j, k) = psi_P_new;

				const FDTD_FLOAT kap_term_P_raw = m_Op_ABC->m_pml_kapI_nyP(i, j, k) * diff_P;
				const FDTD_FLOAT kap_term_P  = flags.disable_kap ? (FDTD_FLOAT)0 : kap_term_P_raw;
				const FDTD_FLOAT psi_term_P  = flags.disable_psi ? (FDTD_FLOAT)0 : psi_P_new;
				const FDTD_FLOAT I_P_after   = I_P_before + kap_term_P + psi_term_P;
				eng->EngType::SetCurr(m_nyP, pos, I_P_after);

				// I[m_nyPP] m_ny-term involves volt(m_nyP, pos+e_ny).
				const FDTD_FLOAT diff_PP =
					eng->EngType::GetVolt(m_nyP, pos_fwd)
					- eng->EngType::GetVolt(m_nyP, pos);

				const FDTD_FLOAT I_PP_before  = eng->EngType::GetCurr(m_nyPP, pos);
				const FDTD_FLOAT psi_PP_old   = m_psi_I_nyPP(i, j, k);
				const FDTD_FLOAT psi_PP_decay = (FDTD_FLOAT)m_Op_ABC->m_pml_b_z[k] * psi_PP_old;
				const FDTD_FLOAT psi_PP_drive = m_Op_ABC->m_pml_ci_nyPP(i, j, k) * diff_PP;
				const FDTD_FLOAT psi_PP_new   = psi_PP_decay + psi_PP_drive;
				m_psi_I_nyPP(i, j, k) = psi_PP_new;

				const FDTD_FLOAT kap_term_PP_raw = m_Op_ABC->m_pml_kapI_nyPP(i, j, k) * diff_PP;
				const FDTD_FLOAT kap_term_PP = flags.disable_kap ? (FDTD_FLOAT)0 : kap_term_PP_raw;
				const FDTD_FLOAT psi_term_PP = flags.disable_psi ? (FDTD_FLOAT)0 : psi_PP_new;
				const FDTD_FLOAT I_PP_after  = I_PP_before + kap_term_PP + psi_term_PP;
				eng->EngType::SetCurr(m_nyPP, pos, I_PP_after);

				sum_psi_I_sq += (double)psi_P_new  * (double)psi_P_new
				             +  (double)psi_PP_new * (double)psi_PP_new;
				sum_I_sq     += (double)I_P_after  * (double)I_P_after
				             +  (double)I_PP_after * (double)I_PP_after;
				sum_diff_sq  += (double)diff_P  * (double)diff_P
				             +  (double)diff_PP * (double)diff_PP;
				if (std::fabs((double)psi_P_new ) > max_psi_I) max_psi_I = std::fabs((double)psi_P_new);
				if (std::fabs((double)psi_PP_new) > max_psi_I) max_psi_I = std::fabs((double)psi_PP_new);
				if (std::fabs((double)I_P_after ) > max_I    ) max_I     = std::fabs((double)I_P_after);
				if (std::fabs((double)I_PP_after) > max_I    ) max_I     = std::fabs((double)I_PP_after);

				if (print_step && is_spot)
				{
					const char* loc = is_corner ? "C" : "M";
					const char* edg = (k == 0) ? "I" : "O";
					std::cerr << "[CPML-I] ts=" << ts
					          << " " << edg << "/" << loc
					          << " ijk=(" << i << "," << j << "," << k << ")"
					          << " pos_ny=" << pos[m_ny]
					          << " fwd_ny=" << pos_fwd[m_ny]
					          << " | dP=" << diff_P
					          << " psi_P[old=" << psi_P_old
					          << " dec=" << psi_P_decay
					          << " drv=" << psi_P_drive
					          << " new=" << psi_P_new << "]"
					          << " kap_P=" << kap_term_P
					          << " I_P[" << I_P_before << "->" << I_P_after << "]"
					          << " | dPP=" << diff_PP
					          << " psi_PP_new=" << psi_PP_new
					          << " I_PP[" << I_PP_before << "->" << I_PP_after << "]"
					          << std::endl;
				}
			}
		}
	}

	if (print_step)
	{
		std::cerr << "[CPML-I-SUM] ts=" << ts
		          << " thr=" << threadID
		          << " sum_psi2=" << sum_psi_I_sq
		          << " sum_I2=" << sum_I_sq
		          << " sum_dV2=" << sum_diff_sq
		          << " max_psi=" << max_psi_I
		          << " max_I=" << max_I
		          << std::endl;
	}

	if (flags.warn_nonfinite && threadID == 0
	    && (!std::isfinite(max_psi_I) || !std::isfinite(max_I)))
	{
		std::cerr << "[CPML-I-NAN] ts=" << ts
		          << " max_psi=" << max_psi_I
		          << " max_I=" << max_I
		          << " (non-finite detected!)" << std::endl;
	}
}
