/*
*	Copyright (C) 2025 Gadi Lahav (gadi@rfwithcare.com)
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

/*
 * Excitation-based waveguide modal absorber with temporal H-field averaging.
 *
 * Algorithm per absorber layer:
 *   1. DoPreCurrentUpdates: store H^n at absorber H-plane (before H update)
 *   2. Apply2Current (after H update):
 *      - Compute Emm = overlap(E, mode_E) at absorber E-plane
 *      - Compute Hmm = overlap(0.5*(H^n + H^{n+1}), mode_H) at absorber H-plane
 *      - a_bwd = 0.5*(Emm - dir * Zw * Hmm)
 *      - Subtractive write-back: E -= a_bwd * mode_E, H += dir * a_bwd/Zw * mode_H
 *
 * Priority is set lower than excitation so absorber runs AFTER sources are applied.
 */

#include "engine_ext_waveguide_absorber.h"
#include "operator_ext_waveguide_absorber.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"

Engine_Ext_WaveguideAbsorber::Engine_Ext_WaveguideAbsorber(Operator_Ext_WaveguideAbsorber* op_ext) :
	Engine_Extension(op_ext)
{
	m_Op_WGA = op_ext;

	m_ny   = m_Op_WGA->m_ny;
	m_nyP  = m_Op_WGA->m_nyP;
	m_nyPP = m_Op_WGA->m_nyPP;
	m_dir  = m_Op_WGA->m_dir;
	m_absLayers = m_Op_WGA->m_absLayers;

	for (int i = 0; i < 3; ++i)
		m_posStart[i] = m_Op_WGA->m_start[i];

	m_numLines_E[0] = m_Op_WGA->m_numLines_E[0];
	m_numLines_E[1] = m_Op_WGA->m_numLines_E[1];
	m_E_OverlapW[0]  = m_Op_WGA->m_E_OverlapW[0];
	m_E_OverlapW[1]  = m_Op_WGA->m_E_OverlapW[1];
	m_E_SubtractW[0] = m_Op_WGA->m_E_SubtractW[0];
	m_E_SubtractW[1] = m_Op_WGA->m_E_SubtractW[1];

	m_numLines_H[0] = m_Op_WGA->m_numLines_H[0];
	m_numLines_H[1] = m_Op_WGA->m_numLines_H[1];
	m_H_OverlapW[0]  = m_Op_WGA->m_H_OverlapW[0];
	m_H_OverlapW[1]  = m_Op_WGA->m_H_OverlapW[1];
	m_H_SubtractW[0] = m_Op_WGA->m_H_SubtractW[0];
	m_H_SubtractW[1] = m_Op_WGA->m_H_SubtractW[1];

	for (int l = 0; l < 2; ++l)
	{
		m_E_abs_line[l] = m_Op_WGA->m_E_abs_line[l];
		m_H_abs_line[l] = m_Op_WGA->m_H_abs_line[l];
	}

	m_Zw = m_Op_WGA->m_Zw;

	// Allocate H^n storage for temporal averaging
	for (int l = 0; l < 2; ++l)
	{
		for (int f = 0; f < 2; ++f)
		{
			if (l < m_absLayers)
			{
				m_H_stored[l][f] = new double*[m_numLines_H[0]];
				for (unsigned int p = 0; p < m_numLines_H[0]; ++p)
				{
					m_H_stored[l][f][p] = new double[m_numLines_H[1]];
					for (unsigned int pp = 0; pp < m_numLines_H[1]; ++pp)
						m_H_stored[l][f][p][pp] = 0;
				}
			}
			else
			{
				m_H_stored[l][f] = NULL;
			}
		}
	}

	// Run AFTER excitation sources are applied
	SetPriority(ENG_EXT_PRIO_EXCITATION - 100);
	SetNumberOfThreads(1);
}

Engine_Ext_WaveguideAbsorber::~Engine_Ext_WaveguideAbsorber()
{
	for (int l = 0; l < 2; ++l)
	{
		for (int f = 0; f < 2; ++f)
		{
			if (m_H_stored[l][f])
			{
				for (unsigned int p = 0; p < m_numLines_H[0]; ++p)
					delete[] m_H_stored[l][f][p];
				delete[] m_H_stored[l][f];
			}
		}
	}
}

// ============================================================================
// DoPreCurrentUpdates: store H^n at absorber H-planes before FDTD H update.
// ============================================================================

template <typename EngType>
void Engine_Ext_WaveguideAbsorber::DoPreCurrentUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	unsigned int pos[3] = {0, 0, 0};

	for (int layer = 0; layer < m_absLayers; ++layer)
	{
		pos[m_ny] = m_H_abs_line[layer];

		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;

				// Store both transverse H components at time n
				m_H_stored[layer][0][posP][posPP] = eng->EngType::GetCurr(m_nyP,  pos);
				m_H_stored[layer][1][posP][posPP] = eng->EngType::GetCurr(m_nyPP, pos);
			}
		}
	}
}

void Engine_Ext_WaveguideAbsorber::DoPreCurrentUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, threadID);
}

// ============================================================================
// Apply2Current: modal decomposition with temporal averaging + subtractive
// write-back. Runs after H update, so H^{n+1} is available.
// ============================================================================

template <typename EngType>
void Engine_Ext_WaveguideAbsorber::Apply2CurrentImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	unsigned int pos[3] = {0, 0, 0};

	for (int layer = 0; layer < m_absLayers; ++layer)
	{
		// --- Compute E overlap at this layer's E-plane ---
		double Emm = 0;
		pos[m_ny] = m_E_abs_line[layer];

		for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				Emm += eng->EngType::GetVolt(m_nyP, pos)  * m_E_OverlapW[0][posP][posPP];
				Emm += eng->EngType::GetVolt(m_nyPP, pos) * m_E_OverlapW[1][posP][posPP];
			}
		}

		// --- Compute H overlap with temporal averaging at this layer's H-plane ---
		double Hmm = 0;
		pos[m_ny] = m_H_abs_line[layer];

		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;

				// Temporal average: 0.5*(H^n + H^{n+1})
				double H_avg_P  = 0.5 * (m_H_stored[layer][0][posP][posPP] + eng->EngType::GetCurr(m_nyP,  pos));
				double H_avg_PP = 0.5 * (m_H_stored[layer][1][posP][posPP] + eng->EngType::GetCurr(m_nyPP, pos));

				Hmm += H_avg_P  * m_H_OverlapW[0][posP][posPP];
				Hmm += H_avg_PP * m_H_OverlapW[1][posP][posPP];
			}
		}

		// --- Directional decomposition: absorb backward wave ---
		double a_bwd = 0.5 * (Emm - m_dir * m_Zw * Hmm);

		// --- Subtractive write-back for E at absorber E-plane ---
		pos[m_ny] = m_E_abs_line[layer];

		for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;

				eng->EngType::SetVolt(m_nyP, pos,
					eng->EngType::GetVolt(m_nyP, pos) - a_bwd * m_E_SubtractW[0][posP][posPP]);
				eng->EngType::SetVolt(m_nyPP, pos,
					eng->EngType::GetVolt(m_nyPP, pos) - a_bwd * m_E_SubtractW[1][posP][posPP]);
			}
		}

		// --- Subtractive write-back for H at absorber H-plane ---
		double h_coeff = m_dir * a_bwd / m_Zw;
		pos[m_ny] = m_H_abs_line[layer];

		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;

				eng->EngType::SetCurr(m_nyP, pos,
					eng->EngType::GetCurr(m_nyP, pos) + h_coeff * m_H_SubtractW[0][posP][posPP]);
				eng->EngType::SetCurr(m_nyPP, pos,
					eng->EngType::GetCurr(m_nyPP, pos) + h_coeff * m_H_SubtractW[1][posP][posPP]);
			}
		}
	}
}

void Engine_Ext_WaveguideAbsorber::Apply2Current(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, threadID);
}
