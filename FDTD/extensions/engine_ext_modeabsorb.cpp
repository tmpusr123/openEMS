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
*/

/*
 * File-based waveguide modal absorber with temporal H-field averaging.
 * Single-layer twin of Engine_Ext_WaveguideAbsorber.
 *
 * Algorithm per timestep:
 *   1. DoPreCurrentUpdates: store H^n at the H-plane (before H update).
 *   2. Apply2Current (after H update):
 *      - Emm = overlap(E,    mode_E) at E-plane
 *      - Hmm = overlap(0.5*(H^n + H^{n+1}), mode_H) at H-plane
 *      - a_bwd = 0.5*(Emm − dir*Zw*Hmm)
 *      - Subtractive write-back:  E -= a_bwd*mode_E,
 *                                  H += dir*a_bwd/Zw * mode_H
 */

#include "engine_ext_modeabsorb.h"
#include "operator_ext_modeabsorb.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "FDTD/excitation.h"
#include "FDTD/operator.h"
#include "CSPropModeAbsorb.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

Engine_Ext_ModeAbsorb::Engine_Ext_ModeAbsorb(Operator_Ext_ModeAbsorb* op_ext) :
	Engine_Extension(op_ext)
{
	m_Op_MA = op_ext;

	m_ny   = m_Op_MA->m_ny;
	m_nyP  = m_Op_MA->m_nyP;
	m_nyPP = m_Op_MA->m_nyPP;
	m_dir  = m_Op_MA->m_dir;

	for (int i = 0; i < 3; ++i)
		m_posStart[i] = m_Op_MA->m_start[i];

	m_numLines_E[0] = m_Op_MA->m_numLines_E[0];
	m_numLines_E[1] = m_Op_MA->m_numLines_E[1];
	m_E_OverlapW[0]  = m_Op_MA->m_E_OverlapW[0];
	m_E_OverlapW[1]  = m_Op_MA->m_E_OverlapW[1];
	m_E_SubtractW[0] = m_Op_MA->m_E_SubtractW[0];
	m_E_SubtractW[1] = m_Op_MA->m_E_SubtractW[1];

	m_numLines_H[0] = m_Op_MA->m_numLines_H[0];
	m_numLines_H[1] = m_Op_MA->m_numLines_H[1];
	m_H_OverlapW[0]  = m_Op_MA->m_H_OverlapW[0];
	m_H_OverlapW[1]  = m_Op_MA->m_H_OverlapW[1];
	m_H_SubtractW[0] = m_Op_MA->m_H_SubtractW[0];
	m_H_SubtractW[1] = m_Op_MA->m_H_SubtractW[1];

	m_E_abs_line = m_Op_MA->m_E_abs_line;
	m_H_abs_line = m_Op_MA->m_H_abs_line;
	m_Zw         = m_Op_MA->m_Zw;

	// --- subtract-known-incident state ---
	m_PortAmplitude = m_Op_MA->m_prop ? m_Op_MA->m_prop->GetPortAmplitude() : 0.0;
	// Detect passive recorders. CSXCAD property names beginning with
	// "passive_" are pure recorders that must not V/I-subtract the
	// field — otherwise e.g. passive_port_2_modal_sheet_at_port
	// eats the transmitted forward wave that the modal S21
	// extraction in analysis/modal_sparams.py is supposed to read.
	{
		const std::string _prop_name = m_Op_MA->m_prop
			? m_Op_MA->m_prop->GetName()
			: std::string("");
		m_IsPassive = (_prop_name.compare(0, 8, "passive_") == 0);
	}
	m_TS         = 0;
	m_Calibrated = false;
	m_K_eff      = 0.0;
	m_calib_num  = 0.0;
	m_calib_den  = 0.0;

	// --- state-space recursion buffers (zeroed at start) ---
	if (m_Op_MA->m_UseStateSpace && m_Op_MA->m_SS_n > 0)
	{
		m_psi.assign(m_Op_MA->m_SS_n, 0.0);
		m_psi_new.assign(m_Op_MA->m_SS_n, 0.0);
	}

	// --- Luo-Chen 1-D Klein-Gordon per-mode absorber state ---
	// Defer initialization to first Apply2Current call.
	m_LC_Active     = false;
	m_LC_N          = 0;
	m_LC_dz_line    = 0.0;
	m_LC_A          = 0.0;
	m_LC_B          = 0.0;
	m_LC_mur_coef   = 0.0;
	m_LC_V_prev_far = 0.0;
	m_LC_V_prev_near= 0.0;

	// --- Wang 2022 wave-equation CFS-PML state ---
	// Defer initialization to first Apply2Current call.
	m_WP_Active     = false;
	m_WP_N_pml      = 0;
	m_WP_kappa_max  = 0.0;
	m_WP_alpha_max  = 0.0;
	m_WP_sigma_max  = 0.0;
	m_WP_p_order    = 0;

	// Allocate H^n storage for temporal averaging (single layer + alternate plane).
	for (int f = 0; f < 2; ++f)
	{
		if (m_numLines_H[0] > 0 && m_numLines_H[1] > 0)
		{
			m_H_stored[f] = new double*[m_numLines_H[0]];
			m_H_stored_alt[f] = new double*[m_numLines_H[0]];
			for (unsigned int p = 0; p < m_numLines_H[0]; ++p)
			{
				m_H_stored[f][p] = new double[m_numLines_H[1]];
				m_H_stored_alt[f][p] = new double[m_numLines_H[1]];
				for (unsigned int pp = 0; pp < m_numLines_H[1]; ++pp)
				{
					m_H_stored[f][p][pp] = 0;
					m_H_stored_alt[f][p][pp] = 0;
				}
			}
		}
		else
		{
			m_H_stored[f] = NULL;
			m_H_stored_alt[f] = NULL;
		}
	}

	// Initialize CFS-CPML modal aux fields (Phase O, 2026-05-11).
	m_CPML_Active = (std::getenv("MODE_CPML_ENABLE") != nullptr
	                 && std::string(std::getenv("MODE_CPML_ENABLE")) == "1");
	m_CPML_N = 0;
	if (m_CPML_Active) {
		int N = 15;
		if (const char* env = std::getenv("MODE_CPML_CELLS")) {
			try { N = std::atoi(env); } catch (...) {}
		}
		double sigma_max = 0.2;  // S/m at port plane
		if (const char* env = std::getenv("MODE_CPML_SIGMA_MAX")) {
			try { sigma_max = std::atof(env); } catch (...) {}
		}
		double kappa_max = 1.0;  // κ stretch at port plane
		if (const char* env = std::getenv("MODE_CPML_KAPPA_MAX")) {
			try { kappa_max = std::atof(env); } catch (...) {}
		}
		double alpha_max = 0.1;  // CFS factor at entry (1-g=1, k=N-1, deep)
		if (const char* env = std::getenv("MODE_CPML_ALPHA_MAX")) {
			try { alpha_max = std::atof(env); } catch (...) {}
		}
		double p_profile = 3.0;
		if (const char* env = std::getenv("MODE_CPML_POWER")) {
			try { p_profile = std::atof(env); } catch (...) {}
		}
		m_CPML_N = N;
		m_CPML_b_z.resize(N);
		m_CPML_a_z.resize(N);
		m_CPML_kappa.resize(N);
		m_psi_V_modal.assign(N, 0.0);
		m_psi_I_modal.assign(N, 0.0);
		const double dt = m_Op_MA->GetOperator()->GetTimestep();
		const double eps0 = 8.854187817e-12;
		std::cerr << "[CPML-INIT] N=" << N << " sigma_max=" << sigma_max
		          << " kappa_max=" << kappa_max << " alpha_max=" << alpha_max
		          << " p=" << p_profile << " dt=" << dt << std::endl;
		for (int k = 0; k < N; ++k) {
			// g = k/(N-1): 0 at port plane (k=0), 1 deep inside (k=N-1)
			const double g = (N > 1) ? (double)k / (double)(N - 1) : 0.0;
			// σ_z max at port plane (k=0), zero at deep end (k=N-1)
			const double sigma_z = sigma_max * std::pow(1.0 - g, p_profile);
			// κ_z stretches at port plane, normal at deep end
			const double kappa_z = 1.0 + (kappa_max - 1.0) * std::pow(1.0 - g, p_profile);
			// α (CFS factor): max at entry (deep), zero at port plane
			// Note: standard Roden-Gedney convention.
			const double alpha_z = alpha_max * std::pow(g, p_profile);
			m_CPML_kappa[k] = kappa_z;
			m_CPML_b_z[k] = std::exp(-(sigma_z / kappa_z + alpha_z) * dt / eps0);
			const double denom = sigma_z + kappa_z * alpha_z;
			m_CPML_a_z[k] = (denom > 0)
				? (sigma_z / (kappa_z * denom)) * (m_CPML_b_z[k] - 1.0)
				: 0.0;
			std::cerr << "[CPML-COEF] k=" << k << " σ=" << sigma_z
			          << " κ=" << kappa_z << " α=" << alpha_z
			          << " b=" << m_CPML_b_z[k] << " a=" << m_CPML_a_z[k] << std::endl;
		}
	}

	// Run AFTER excitation sources are applied.
	SetPriority(ENG_EXT_PRIO_EXCITATION - 100);
	SetNumberOfThreads(1);
}

Engine_Ext_ModeAbsorb::~Engine_Ext_ModeAbsorb()
{
	for (int f = 0; f < 2; ++f)
	{
		if (m_H_stored[f])
		{
			for (unsigned int p = 0; p < m_numLines_H[0]; ++p)
				delete[] m_H_stored[f][p];
			delete[] m_H_stored[f];
		}
		if (m_H_stored_alt[f])
		{
			for (unsigned int p = 0; p < m_numLines_H[0]; ++p)
				delete[] m_H_stored_alt[f][p];
			delete[] m_H_stored_alt[f];
		}
	}

	// --- CST Round 11 m_proj_pre dump ---
	// Write the recorded modal-projection-pre-source history to file.
	// Path: <property name>_o_NM.dat in CWD (= simulation directory).
	// One row per timestep: "t/s\tEmm_pre" plus header lines.
	if (!m_o_history.empty() && m_Op_MA && m_Op_MA->m_prop)
	{
		const std::string name = m_Op_MA->m_prop->GetName();
		const std::string path = name + "_o_NM.dat";
		std::ofstream f(path);
		if (f.is_open())
		{
			double dt = 0.0;
			if (m_Op_MA->GetOperator())
				dt = m_Op_MA->GetOperator()->GetTimestep();
			f << "% m_proj_pre history (CST-style outgoing modal coefficient)\n";
			f << "% property: " << name << "\n";
			f << "% dt = " << dt << " s\n";
			f << "% t/s\tEmm_pre\n";
			for (size_t i = 0; i < m_o_history.size(); ++i)
				f << ((double)i * dt) << "\t" << m_o_history[i] << "\n";
			std::cerr << "[MA-OPRE] wrote " << m_o_history.size()
			          << " samples to " << path << std::endl;
		}
		else
		{
			std::cerr << "[MA-OPRE] failed to open " << path << " for writing\n";
		}
	}

	// --- CST Round 12 Emm/Hmm dump (for forward/backward decomposition) ---
	if (!m_emm_history.empty() && m_Op_MA && m_Op_MA->m_prop)
	{
		const std::string name = m_Op_MA->m_prop->GetName();
		double dt = 0.0;
		if (m_Op_MA->GetOperator())
			dt = m_Op_MA->GetOperator()->GetTimestep();

		const std::string e_path = name + "_Emm.dat";
		std::ofstream fe(e_path);
		if (fe.is_open())
		{
			fe << "% Emm history at Apply2Current time (Yee-consistent with Hmm)\n";
			fe << "% property: " << name << "\n";
			fe << "% dt = " << dt << " s\n";
			fe << "% t/s\tEmm\n";
			for (size_t i = 0; i < m_emm_history.size(); ++i)
				fe << ((double)i * dt) << "\t" << m_emm_history[i] << "\n";
			std::cerr << "[MA-EMM] wrote " << m_emm_history.size()
			          << " samples to " << e_path << std::endl;
		}

		const std::string h_path = name + "_Hmm.dat";
		std::ofstream fh(h_path);
		if (fh.is_open())
		{
			fh << "% Hmm history at Apply2Current time (after temporal+spatial avg)\n";
			fh << "% property: " << name << "\n";
			fh << "% dt = " << dt << " s\n";
			fh << "% t/s\tHmm\n";
			for (size_t i = 0; i < m_hmm_history.size(); ++i)
				fh << ((double)i * dt) << "\t" << m_hmm_history[i] << "\n";
			std::cerr << "[MA-HMM] wrote " << m_hmm_history.size()
			          << " samples to " << h_path << std::endl;
		}
	}
}

template <typename EngType>
void Engine_Ext_ModeAbsorb::DoPreCurrentUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	unsigned int pos[3] = {0, 0, 0};

	// Primary plane: H at m_H_abs_line (z = (k+½)·dz in Yee).
	pos[m_ny] = m_H_abs_line;
	for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			m_H_stored[0][posP][posPP] = eng->EngType::GetCurr(m_nyP,  pos);
			m_H_stored[1][posP][posPP] = eng->EngType::GetCurr(m_nyPP, pos);
		}
	}

	// Alternate plane: H at m_H_abs_line − 1 (z = (k−½)·dz).  Average with
	// primary gives H interpolated to z = k·dz (the E plane).  Skip if at
	// mesh boundary (m_H_abs_line == 0).
	// 2026-05-11 Phase J empirical result: averaging gives spurious source
	// proportional to (1−cos(β·dz/2)) on forward waves; helps S11 at low
	// freqs but HURTS at high freqs (-7 dB at 2.5 GHz vs −13 dB without).
	// Default OFF; opt in via MODE_YEE_SPATIAL_AVG=1.
	static const bool s_yee_avg =
		std::getenv("MODE_YEE_SPATIAL_AVG") != nullptr &&
		std::string(std::getenv("MODE_YEE_SPATIAL_AVG")) == "1";
	if (s_yee_avg && m_H_abs_line > 0)
	{
		pos[m_ny] = m_H_abs_line - 1;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				m_H_stored_alt[0][posP][posPP] = eng->EngType::GetCurr(m_nyP,  pos);
				m_H_stored_alt[1][posP][posPP] = eng->EngType::GetCurr(m_nyPP, pos);
			}
		}
	}
}

void Engine_Ext_ModeAbsorb::DoPreCurrentUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, threadID);
}

// =========================================================================
// CFS-CPML Apply2Voltages: modal-projection-based CFS-CPML for E side.
// =========================================================================
template <typename EngType>
void Engine_Ext_ModeAbsorb::Apply2VoltagesImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL || threadID != 0) return;
	// Round-9 passive-stamp mode (CST architecture): the port plane only
	// probes (via openEMS Port U/I probes) and does NOT actively absorb.
	// All absorption is done by the outer boundary.  Skip modal CPML
	// aux-field updates entirely.
	static const bool s_passive_stamp =
		(std::getenv("MODE_ABSORB_PASSIVE") != nullptr
		 && std::string(std::getenv("MODE_ABSORB_PASSIVE")) == "1");
	if (s_passive_stamp) return;
	if (!m_CPML_Active) return;

	unsigned int pos[3] = {0, 0, 0};

	// For each cell k in CPML region:
	// - Compute modal V_modal[k] (overlap with mode pattern)
	// - Compute modal I_modal[k+dir] and I_modal[k-dir] for spatial derivative
	// - Update ψ_V_modal[k] = b·ψ + a·(∂I_modal/∂z)
	// - V_modal correction: ΔV = (dt/ε)·ψ_V_modal[k]
	// - Distribute correction via mode_E pattern to E_y at cell k

	const double dt = m_Op_MA->GetOperator()->GetTimestep();
	const double eps0 = 8.854187817e-12;
	const double inv_eps_dt = dt / eps0;

	for (int k = 0; k < m_CPML_N; ++k)
	{
		// Cell index INSIDE WG: offset = +m_dir·k from port plane
		const int e_idx = (int)m_E_abs_line + m_dir * k;
		const int h_idx_plus = (int)m_H_abs_line + m_dir * (k + 1);
		const int h_idx_minus = (int)m_H_abs_line + m_dir * (k - 1);
		if (e_idx < 0 || h_idx_plus < 0 || h_idx_minus < 0) continue;

		// Sample modal I at z=k+1 and z=k-1 (along propagation direction)
		double I_plus = 0.0, I_minus = 0.0;
		pos[m_ny] = (unsigned int)h_idx_plus;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP) {
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP) {
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				I_plus += eng->EngType::GetCurr(m_nyP, pos)  * m_H_OverlapW[0][posP][posPP];
				I_plus += eng->EngType::GetCurr(m_nyPP, pos) * m_H_OverlapW[1][posP][posPP];
			}
		}
		pos[m_ny] = (unsigned int)h_idx_minus;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP) {
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP) {
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				I_minus += eng->EngType::GetCurr(m_nyP, pos)  * m_H_OverlapW[0][posP][posPP];
				I_minus += eng->EngType::GetCurr(m_nyPP, pos) * m_H_OverlapW[1][posP][posPP];
			}
		}
		// ∂I_modal/∂z (central difference in INTO-WG direction = m_dir)
		// For dir=+1: forward = +z, so increasing k = +z. dI/dz = (I[k+1] - I[k-1])/(2dz·m_dir)
		// For dir=-1: forward = -z, dI/dz inverted. Use sign with m_dir.
		const double dz_phys = m_Op_MA->GetOperator()->GetEdgeLength(m_ny, m_posStart, false);
		const double dI_dz = m_dir * (I_plus - I_minus) / (2.0 * dz_phys);

		// CFS-CPML update for ψ_V_modal
		m_psi_V_modal[k] = m_CPML_b_z[k] * m_psi_V_modal[k] + m_CPML_a_z[k] * dI_dz;

		// V_modal correction: +(dt/ε)·ψ_V_modal[k]  (added to V_modal)
		const double dV_modal = inv_eps_dt * m_psi_V_modal[k];

		// Distribute correction via mode pattern to E at cell k.
		// Standard CFS-CPML adds ψ_E·(dt/ε) to E.
		pos[m_ny] = (unsigned int)e_idx;
		for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP) {
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP) {
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				eng->EngType::SetVolt(m_nyP, pos,
					eng->EngType::GetVolt(m_nyP, pos)
					+ dV_modal * m_E_SubtractW[0][posP][posPP]);
				eng->EngType::SetVolt(m_nyPP, pos,
					eng->EngType::GetVolt(m_nyPP, pos)
					+ dV_modal * m_E_SubtractW[1][posP][posPP]);
			}
		}
	}
}

void Engine_Ext_ModeAbsorb::Apply2Voltages(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_ModeAbsorb::Apply2CurrentImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	const unsigned int ts = m_Eng ? m_Eng->GetNumberOfTimesteps() : 0;
	const bool print_step = (ts < 30) ||
	                        (ts < 300 && (ts % 10) == 0) ||
	                        ((ts % 200) == 0);

	unsigned int pos[3] = {0, 0, 0};

	// --- E overlap at E-plane ---
	double Emm = 0;
	double max_V_at_E_plane = 0.0;
	pos[m_ny] = m_E_abs_line;
	for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			const FDTD_FLOAT v_p  = eng->EngType::GetVolt(m_nyP,  pos);
			const FDTD_FLOAT v_pp = eng->EngType::GetVolt(m_nyPP, pos);
			Emm += v_p  * m_E_OverlapW[0][posP][posPP];
			Emm += v_pp * m_E_OverlapW[1][posP][posPP];
			if (std::fabs((double)v_p ) > max_V_at_E_plane) max_V_at_E_plane = std::fabs((double)v_p);
			if (std::fabs((double)v_pp) > max_V_at_E_plane) max_V_at_E_plane = std::fabs((double)v_pp);
		}
	}

	// --- H overlap with TEMPORAL averaging + SPATIAL averaging (Yee stagger
	//     correction).  Primary plane H is at z = (k+½)·dz; alternate plane
	//     at z = (k−½)·dz.  Average both gives H at z = k·dz (= E plane).
	static const bool s_yee_avg_apply =
		std::getenv("MODE_YEE_SPATIAL_AVG") == nullptr ||
		std::string(std::getenv("MODE_YEE_SPATIAL_AVG")) != "0";
	const bool use_alt = s_yee_avg_apply && (m_H_abs_line > 0);

	double Hmm = 0;
	pos[m_ny] = m_H_abs_line;
	for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			// Temporal: 0.5*(H^n + H^{n+1}); primary plane
			double H_prim_P  = 0.5 * (m_H_stored[0][posP][posPP] + eng->EngType::GetCurr(m_nyP,  pos));
			double H_prim_PP = 0.5 * (m_H_stored[1][posP][posPP] + eng->EngType::GetCurr(m_nyPP, pos));
			Hmm += H_prim_P  * m_H_OverlapW[0][posP][posPP];
			Hmm += H_prim_PP * m_H_OverlapW[1][posP][posPP];
		}
	}
	if (use_alt)
	{
		pos[m_ny] = m_H_abs_line - 1;
		double Hmm_alt = 0;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				double H_alt_P  = 0.5 * (m_H_stored_alt[0][posP][posPP] + eng->EngType::GetCurr(m_nyP,  pos));
				double H_alt_PP = 0.5 * (m_H_stored_alt[1][posP][posPP] + eng->EngType::GetCurr(m_nyPP, pos));
				Hmm_alt += H_alt_P  * m_H_OverlapW[0][posP][posPP];
				Hmm_alt += H_alt_PP * m_H_OverlapW[1][posP][posPP];
			}
		}
		// Spatial average: (Hmm_primary + Hmm_alt)/2 = H_at_E_plane
		Hmm = 0.5 * (Hmm + Hmm_alt);
	}

	// --- CST Round 12 forward/backward decomposition recording ---
	// Record both Emm and Hmm at Apply2Current time (consistent Yee level after
	// temporal+spatial H averaging).  Post-processing computes
	//     b_back = 0.5 · (Emm / √Z_w − √Z_w · Hmm)
	// as the OUTGOING modal coefficient.  Source contribution to this signal
	// vanishes only if the source stamp also injects H_tan in matched V/I ratio
	// (= source/Z_w) alongside the E_tan stamp.  With openEMS soft-additive
	// E-only stamping, b_back retains the source contribution and accumulates
	// the same way as raw m_proj_pre — this recording will let us verify that.
	m_emm_history.push_back(Emm);
	m_hmm_history.push_back(Hmm);

	// =========================================================================
	// Round-9 passive-stamp mode (CST architecture, 2026-05-13)
	// In CST, the port plane does NO active absorption — it only injects the
	// source (via the standard openEMS WaveGuidePort excitation that runs
	// before this extension) and probes the modal projection (via the openEMS
	// p_type=10/11 probes that are independent of this extension).  Setting
	// MODE_ABSORB_PASSIVE=1 disables every active modification of E/H made by
	// this extension (V/I matched-modal-source, Luo-Chen 1-D KG line,
	// state-space recursion, modal damping), leaving the entire absorption
	// budget to the outer boundary BC.  This is the architecture the CST LLM
	// Round 9 Frida trace established as the truth-of-record for CST.
	// =========================================================================
	{
		static const bool s_passive_stamp =
			(std::getenv("MODE_ABSORB_PASSIVE") != nullptr
			 && std::string(std::getenv("MODE_ABSORB_PASSIVE")) == "1");
		if (s_passive_stamp)
		{
			if (print_step)
			{
				std::cerr << "[MA-PASSIVE] ts=" << ts
				          << " Emm=" << Emm
				          << " Hmm=" << Hmm
				          << " max_V@Eplane=" << max_V_at_E_plane
				          << std::endl;
			}
			return;
		}
	}

	// =========================================================================
	// Luo-Chen 1-D Yee per-mode absorber (Phase C, 2026-05-10)
	// Provably-stable alternative to V/I matched-modal-source: an explicit
	// 1-D modal transmission line behind the port plane absorbs the modal
	// wave; far end uses Mur ABC.  Activate via env USE_LUO_CHEN_TIER1=1.
	// =========================================================================
	{
		static const bool s_lc_active =
			(std::getenv("USE_LUO_CHEN_TIER1") != nullptr
			 && std::string(std::getenv("USE_LUO_CHEN_TIER1")) == "1");

		if (s_lc_active)
		{
			// Lazy initialization on first call (dt valid only after engine constructed).
			if (m_LC_N == 0)
			{
				const int n_default = 30;
				int n_cells = n_default;
				if (const char* env = std::getenv("LUO_CHEN_N")) {
					int v = std::atoi(env);
					if (v >= 5 && v <= 1000) n_cells = v;
				}
				m_LC_N = n_cells;
				m_LC_V.assign(m_LC_N, 0.0);
				m_LC_V_prev.assign(m_LC_N, 0.0);

				// Klein-Gordon dispersion: ∂²V/∂t² = c²·∂²V/∂z² − ω_c²·V
				// (gives correct waveguide TE dispersion ω² = c²β² + ω_c²)
				const double dt = m_Op_MA->GetOperator()->GetTimestep();
				const double C0_v = 299792458.0;
				const double dz_default = 2.0 * C0_v * dt;  // CFL ≈ 0.5 on c
				m_LC_dz_line = dz_default;
				if (const char* env = std::getenv("LUO_CHEN_DZ_LINE_M")) {
					double v = std::atof(env);
					if (v > 0.0) m_LC_dz_line = v;
				}
				const double cdt_dz = C0_v * dt / m_LC_dz_line;
				m_LC_A = cdt_dz * cdt_dz;
				// Modal cutoff f_c.  Default 1.665 GHz (WR-class TE10 in our
				// geometry).  Override via LUO_CHEN_FC_HZ env var.
				double fc_hz = 1.665e9;
				if (const char* env = std::getenv("LUO_CHEN_FC_HZ")) {
					double v = std::atof(env);
					if (v > 0.0) fc_hz = v;
				}
				const double omega_c = 2.0 * 3.141592653589793 * fc_hz;
				m_LC_B = (omega_c * dt) * (omega_c * dt);
				const double mur_num = C0_v * dt - m_LC_dz_line;
				const double mur_den = C0_v * dt + m_LC_dz_line;
				m_LC_mur_coef = (mur_den != 0.0) ? (mur_num / mur_den) : 0.0;
				m_LC_Active = true;

				std::cerr << "[LC-INIT] (Klein-Gordon) dir=" << m_dir
				          << " N=" << m_LC_N
				          << " Zw=" << m_Zw
				          << " dz_line=" << m_LC_dz_line
				          << " dt=" << dt
				          << " A=(c·dt/dz)²=" << m_LC_A
				          << " B=(ω_c·dt)²=" << m_LC_B
				          << " fc=" << fc_hz
				          << " mur=" << m_LC_mur_coef
				          << std::endl;

				// Wang 2022 wave-equation CFS-PML init (replaces Mur ABC at
				// far end of the KG line).  Default OFF.  Activate via
				// USE_WANG_CFS_PML=1.  Optimal params: m=4, κ_max=4, α_max=0.1,
				// σ_opt=(m+1)/(150π·Δs).
				if (std::getenv("USE_WANG_CFS_PML") != nullptr &&
				    std::string(std::getenv("USE_WANG_CFS_PML")) == "1")
				{
					int N_pml = 10;
					if (const char* env = std::getenv("WANG_N_PML")) {
						int v = std::atoi(env);
						if (v >= 4 && v <= m_LC_N - 2) N_pml = v;
					}
					int p_order = 4;
					if (const char* env = std::getenv("WANG_P_ORDER")) {
						int v = std::atoi(env);
						if (v >= 1 && v <= 10) p_order = v;
					}
					double kappa_max = 4.0;
					if (const char* env = std::getenv("WANG_KAPPA_MAX")) {
						double v = std::atof(env);
						if (v >= 1.0 && v <= 50.0) kappa_max = v;
					}
					double alpha_max = 0.1;
					if (const char* env = std::getenv("WANG_ALPHA_MAX")) {
						double v = std::atof(env);
						if (v >= 0.0 && v <= 10.0) alpha_max = v;
					}
					// σ_opt = (m+1) / (150·π·Δs).  Wang recommends
					// σ_max/σ_opt in [0.5, 3]; default 1.0.
					const double sigma_opt = (double)(p_order + 1) /
					                         (150.0 * 3.141592653589793 * m_LC_dz_line);
					double sigma_max = sigma_opt;
					if (const char* env = std::getenv("WANG_SIGMA_MAX_RATIO")) {
						double v = std::atof(env);
						if (v > 0.0 && v <= 10.0) sigma_max = v * sigma_opt;
					}
					if (const char* env = std::getenv("WANG_SIGMA_MAX")) {
						double v = std::atof(env);
						if (v > 0.0) sigma_max = v;
					}

					m_WP_N_pml     = N_pml;
					m_WP_p_order   = p_order;
					m_WP_kappa_max = kappa_max;
					m_WP_alpha_max = alpha_max;
					m_WP_sigma_max = sigma_max;

					// Profiles.  k=0..N_pml-1 indexes within the PML region
					// (k=0 nearest the KG line, k=N_pml-1 deepest near PEC).
					// ζ = (k+½)/N_pml for cell-centred σ, α, κ.
					// For ξ at k+½, ζ = (k+1)/N_pml.
					m_WP_sigma_max = sigma_max;
					m_WP_kappa_half.assign(N_pml + 1, 1.0);
					m_WP_kappa_cell.assign(N_pml, 1.0);
					m_WP_b_half.assign(N_pml + 1, 0.0);
					m_WP_c_half.assign(N_pml + 1, 0.0);
					m_WP_b_cell.assign(N_pml, 0.0);
					m_WP_c_cell.assign(N_pml, 0.0);
					m_WP_xi .assign(N_pml + 1, 0.0);
					m_WP_phi.assign(N_pml, 0.0);

					const double EPS0_v = 8.854187817e-12;
					for (int k = 0; k < N_pml; ++k) {
						const double zeta_c = ((double)k + 0.5) / (double)N_pml;
						const double zeta_h = (double)(k + 1) / (double)N_pml;
						const double sig_c  = sigma_max * std::pow(zeta_c, p_order);
						const double sig_h  = sigma_max * std::pow(zeta_h, p_order);
						const double alp_c  = alpha_max * (1.0 - zeta_c);
						const double alp_h  = alpha_max * (1.0 - zeta_h);
						const double kap_c  = 1.0 + (kappa_max - 1.0) * std::pow(zeta_c, p_order);
						const double kap_h  = 1.0 + (kappa_max - 1.0) * std::pow(zeta_h, p_order);
						m_WP_kappa_cell[k] = kap_c;
						m_WP_kappa_half[k+1] = kap_h;
						// Roden-Gedney coefficients (per Wang 2022 eqns 17-22):
						//   η = exp(-(σ/(ε₀·κ) + α/ε₀)·Δt)
						//   c = (σ / (σ·κ + κ²·α)) · (η − 1)
						const double tau_c = (sig_c / kap_c + alp_c) * dt / EPS0_v;
						const double tau_h = (sig_h / kap_h + alp_h) * dt / EPS0_v;
						const double b_c   = std::exp(-tau_c);
						const double b_h   = std::exp(-tau_h);
						const double den_c = sig_c * kap_c + kap_c * kap_c * alp_c;
						const double den_h = sig_h * kap_h + kap_h * kap_h * alp_h;
						m_WP_b_cell[k]   = b_c;
						m_WP_c_cell[k]   = (den_c > 0.0) ? (sig_c / den_c) * (b_c - 1.0) : 0.0;
						m_WP_b_half[k+1] = b_h;
						m_WP_c_half[k+1] = (den_h > 0.0) ? (sig_h / den_h) * (b_h - 1.0) : 0.0;
					}
					// Boundary at k=0: ξ[½] uses zeta=1/(2·N_pml) (boundary).
					// Inner edge of PML transition; minimal σ, κ≈1.
					m_WP_kappa_half[0] = 1.0;
					m_WP_b_half[0]     = 1.0;
					m_WP_c_half[0]     = 0.0;
					m_WP_Active = true;

					std::cerr << "[WANG-INIT] dir=" << m_dir
					          << " N_pml=" << N_pml
					          << " p=" << p_order
					          << " kappa_max=" << kappa_max
					          << " alpha_max=" << alpha_max
					          << " sigma_opt=" << sigma_opt
					          << " sigma_max=" << sigma_max
					          << " dz_line=" << m_LC_dz_line
					          << std::endl;
					for (int k = 0; k < N_pml; ++k) {
						std::cerr << "[WANG-PROF] k=" << k
						          << " kap_c=" << m_WP_kappa_cell[k]
						          << " kap_h=" << m_WP_kappa_half[k+1]
						          << " b_c="   << m_WP_b_cell[k]
						          << " c_c="   << m_WP_c_cell[k]
						          << " b_h="   << m_WP_b_half[k+1]
						          << " c_h="   << m_WP_c_half[k+1]
						          << std::endl;
					}
				}
			}

			// Klein-Gordon update.  Drive V[0] = Emm (3-D → 1-D).  Interior
			// V[k] for k=1..N-2 update via 2-step KG.  Mur ABC at k=N-1.
			//   V[k]^{n+1} = 2V[k]^n − V[k]^{n-1}
			//              + A·(V[k+1]−2V[k]+V[k-1])^n − B·V[k]^n
			//
			// Wang 2022 CFS-PML augmentation (when m_WP_Active):
			// Last m_WP_N_pml cells of the line use stretched-coordinate KG:
			//   ∂²V/∂z²_PML at cell k = (1/κ_c)·(u[k+½] − u[k−½])/dz + φ[k]
			//     where u[k+½] = (1/κ_h)·(V[k+1]−V[k])/dz + ξ[k+½]
			//   ξ[k+½]^{n+1} = b_h·ξ[k+½]^n + c_h·(V[k+1]−V[k])^n/dz
			//   φ[k]^{n+1}   = b_c·φ[k]^n   + c_c·((V[k+1]−V[k])/(κ_h·dz)
			//                                     +ξ[k+½]^{n+1}
			//                                     −(V[k]−V[k−1])/(κ_h_prev·dz)
			//                                     −ξ[k−½]^{n+1})/dz
			// PEC at k=N−1 (V=0) closes the line.  PML cells span
			// k = pml_kstart..N-2 with PEC at k=N-1.
			std::vector<double> V_new(m_LC_N, 0.0);
			V_new[0] = Emm;
			m_LC_V[0] = Emm;  // ensure V[0]^n = Emm for KG stencil at k=1
			const int pml_kstart = m_WP_Active
			                       ? (m_LC_N - 1 - m_WP_N_pml) : (m_LC_N - 1);

			// Propagation region (no PML): k = 1..pml_kstart-1.  Standard KG.
			for (int k = 1; k < pml_kstart; ++k) {
				const double laplacian = m_LC_V[k+1] - 2.0 * m_LC_V[k] + m_LC_V[k-1];
				V_new[k] = 2.0 * m_LC_V[k] - m_LC_V_prev[k]
				         + m_LC_A * laplacian
				         - m_LC_B * m_LC_V[k];
			}

			if (m_WP_Active && m_WP_N_pml > 0)
			{
				// PML region: k = pml_kstart..N-2.  PML-cell index pk = k − pml_kstart.
				// PEC at k=N-1 (V=0).  Half cells: pkh = 0..N_pml, where
				//   pkh = 0:        z = (pml_kstart − ½)·dz  (boundary with propagation)
				//   pkh = pk+1:     z = (pml_kstart + pk + ½)·dz  (above PML cell pk)
				//   pkh = N_pml:    z = (m_LC_N − 1 − ½)·dz  (boundary with PEC)
				// dV at half cell pkh = (V[pml_kstart+pkh] − V[pml_kstart+pkh−1])/dz,
				// taking V[m_LC_N − 1] = 0 (PEC) for the last half cell.
				const double dz = m_LC_dz_line;
				const int N_pml = m_WP_N_pml;

				// Step 1: update ξ[pkh] for pkh = 0..N_pml using V^n.
				std::vector<double> xi_new(N_pml + 1, 0.0);
				for (int pkh = 0; pkh <= N_pml; ++pkh) {
					const int k_lower = pml_kstart + pkh - 1;
					const int k_upper = pml_kstart + pkh;
					double dV;
					if (k_upper >= m_LC_N - 1) {
						// last half cell uses PEC V[N-1]=0
						dV = (0.0 - m_LC_V[k_lower]) / dz;
					} else if (k_lower < 0) {
						dV = 0.0;
					} else {
						dV = (m_LC_V[k_upper] - m_LC_V[k_lower]) / dz;
					}
					xi_new[pkh] = m_WP_b_half[pkh] * m_WP_xi[pkh]
					            + m_WP_c_half[pkh] * dV;
				}

				// Step 2: compute u[pkh] = (1/κ_h)·dV/dz + ξ[pkh]^{n+1}
				std::vector<double> u_n(N_pml + 1, 0.0);
				for (int pkh = 0; pkh <= N_pml; ++pkh) {
					const int k_lower = pml_kstart + pkh - 1;
					const int k_upper = pml_kstart + pkh;
					double dV;
					if (k_upper >= m_LC_N - 1) {
						dV = (0.0 - m_LC_V[k_lower]) / dz;
					} else if (k_lower < 0) {
						dV = 0.0;
					} else {
						dV = (m_LC_V[k_upper] - m_LC_V[k_lower]) / dz;
					}
					const double kap = m_WP_kappa_half[pkh];
					u_n[pkh] = dV / kap + xi_new[pkh];
				}

				// Step 3: update φ[pk] and compute V at PML cells.
				std::vector<double> phi_new(N_pml, 0.0);
				for (int pk = 0; pk < N_pml; ++pk) {
					const int k = pml_kstart + pk;
					if (k >= m_LC_N - 1) break;
					const double du_dz = (u_n[pk + 1] - u_n[pk]) / dz;
					phi_new[pk] = m_WP_b_cell[pk] * m_WP_phi[pk]
					            + m_WP_c_cell[pk] * du_dz;
					const double laplacian_pml = du_dz / m_WP_kappa_cell[pk]
					                            + phi_new[pk];
					// m_LC_A·dz²·laplacian_pml = (c·dt)²·∂²V/∂z² ✓
					V_new[k] = 2.0 * m_LC_V[k] - m_LC_V_prev[k]
					         + m_LC_A * dz * dz * laplacian_pml
					         - m_LC_B * m_LC_V[k];
				}

				// PEC closure at k=N-1: V[N-1] = 0 always.
				V_new[m_LC_N - 1] = 0.0;

				// Commit aux updates.
				m_WP_xi  = xi_new;
				m_WP_phi = phi_new;
			}
			else if (m_LC_N >= 2)
			{
				// Mur 1st-order ABC at far end (when Wang CFS-PML inactive):
				//   V[N-1]^{n+1} = V[N-2]^n + mur·(V[N-2]^{n+1} − V[N-1]^n)
				const double V_far_old  = m_LC_V[m_LC_N - 1];
				const double V_near_new = V_new[m_LC_N - 2];
				V_new[m_LC_N - 1] = m_LC_V_prev_near
				                  + m_LC_mur_coef * (V_near_new - V_far_old);
				m_LC_V_prev_near = m_LC_V[m_LC_N - 2];
				m_LC_V_prev_far  = V_far_old;
			}

			// Apply correction: 3-D modal V should become V_new[1] (the
			// line's "predicted next-step value just inside port plane").
			//   dV_corr = Emm − V_new[1]
			//   V_3D[port edges] -= dV_corr · e_mode
			const double V_predicted = V_new[1];
			const double dV_correction = Emm - V_predicted;
			pos[m_ny] = m_E_abs_line;
			for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
			{
				pos[m_nyP] = m_posStart[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
				{
					pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
					eng->EngType::SetVolt(m_nyP, pos,
						eng->EngType::GetVolt(m_nyP, pos)
						- dV_correction * m_E_SubtractW[0][posP][posPP]);
					eng->EngType::SetVolt(m_nyPP, pos,
						eng->EngType::GetVolt(m_nyPP, pos)
						- dV_correction * m_E_SubtractW[1][posP][posPP]);
				}
			}

			// Time-shift: V_prev <- V_curr, V_curr <- V_new
			m_LC_V_prev = m_LC_V;
			m_LC_V      = V_new;

			if (print_step)
			{
				double max_V_lc = 0.0;
				for (int k = 0; k < m_LC_N; ++k)
					if (std::fabs(m_LC_V[k]) > max_V_lc) max_V_lc = std::fabs(m_LC_V[k]);
				std::cerr << "[MA-LC] ts=" << ts
				          << " Emm=" << Emm
				          << " Hmm=" << Hmm
				          << " V_LC[1]=" << V_predicted
				          << " V_LC[N/2]=" << m_LC_V[m_LC_N / 2]
				          << " V_LC[N-1]=" << m_LC_V[m_LC_N - 1]
				          << " dV_corr=" << dV_correction
				          << " max|V_LC|=" << max_V_lc
				          << " max_V@Eplane=" << max_V_at_E_plane
				          << std::endl;
			}
			return;
		}
	}
	// =========================================================================

	// =========================================================================
	// Two paths:
	//   A) State-space Y_TE absorber (Phase H, CST-LLM Round 5 recipe):
	//      Fit Y_TE(jω) = γ/(jωμ₀) per mode to rational state-space.
	//      Drive with u = Emm (modal V).  Output y = matched modal current.
	//      Apply H correction: H_modal_3D ← matched_H by additive write-back.
	//      For matched forward wave, V/H = Z_TE(ω) at all frequencies.
	//   B) V/I baseline (state-space inactive): scalar Z_w matched-modal
	//      source with dual E+H subtractive write-back.
	// =========================================================================
	if (m_Op_MA->m_UseStateSpace && m_Op_MA->m_SS_n > 0)
	{
		const unsigned int n = m_Op_MA->m_SS_n;
		const double *P  = m_Op_MA->m_SS_P.data();
		const double *Q  = m_Op_MA->m_SS_Q.data();
		const double *Cv = m_Op_MA->m_SS_C.data();
		const double  D  = m_Op_MA->m_SS_D;
		// E-subtract default OFF after Phase K test showed divergence (NaN by
		// ts~14000) when combined with force-match on H — even when feeding V_f
		// to state-space, the iteration is numerically unstable.  Default OFF.
		static const bool s_e_subtract =
			std::getenv("MODE_E_SUBTRACT") != nullptr &&
			std::string(std::getenv("MODE_E_SUBTRACT")) == "1";
		const double V_b_scalar = s_e_subtract
			? 0.5 * (Emm - m_dir * m_Zw * Hmm)
			: 0.0;
		const double u = Emm;  // drive state-space with modal V

		// BE recursion: ψ_new = P·ψ + Q·u_n
		for (unsigned int i = 0; i < n; ++i)
		{
			double s = Q[i] * u;
			const double *Prow = P + (size_t)i * n;
			for (unsigned int j = 0; j < n; ++j)
				s += Prow[j] * m_psi[j];
			m_psi_new[i] = s;
		}
		// matched_H = C·ψ_new + D·u  (= (Y_TE * V)(t), the modal current that
		// matches V at every frequency for a pure forward wave).
		double matched_H = D * u;
		for (unsigned int j = 0; j < n; ++j)
			matched_H += Cv[j] * m_psi_new[j];
		m_psi.swap(m_psi_new);

		double max_psi_diag = 0.0;
		for (unsigned int i = 0; i < n; ++i)
			if (std::fabs(m_psi[i]) > max_psi_diag) max_psi_diag = std::fabs(m_psi[i]);

		// 2026-05-11 Phase H/I: Force-match write-back.
		//
		// We tried subtractive (a_bwd-style) write-back per modal transmission
		// line theory: I_b_modal = 0.5(Hmm − m_dir·matched_H), then subtract
		// I_b·mode_H from H and V_b·mode_E from E.  In the scalar Y limit this
		// reduces to V/I.  Empirically this performed WORSE at the band edges
		// (S11 −2 dB at 1.7 GHz vs force-match −11 dB) because the scalar Z_w
		// E-correction is wrong near cutoff where Z_TE diverges from Z_w.
		//
		// Force-match (this branch) overrides H_modal to equal matched_H = Y·V
		// at the port plane each timestep.  Mathematically: dH = matched_H − Hmm.
		// This applies 2× the subtractive coefficient (which would be
		// 0.5·(matched_H − m_dir·Hmm) for dir=+1), so it can be viewed as an
		// "overcorrection".  Empirically it acts as a Dirichlet-like BC that
		// enforces V/I = 1/Y(ω) at the port plane — a matched-load condition.
		// Gives consistent ~−14 dB S11 across the propagating band, where
		// subtractive degraded badly at the band edges.
		const double dH_correction = matched_H - Hmm;

		double dH_applied = dH_correction;
		bool clamped = false;
		static const bool s_ss_clamp =
			std::getenv("SS_CLAMP_PASSIVITY") == nullptr ||
			std::string(std::getenv("SS_CLAMP_PASSIVITY")) != "0";
		if (s_ss_clamp) {
			const double alpha = 2.0;
			const double max_dH = alpha * (std::fabs(Hmm) + std::fabs(matched_H) + 1e-30);
			if (dH_applied > max_dH)      { dH_applied = max_dH;   clamped = true; }
			else if (dH_applied < -max_dH){ dH_applied = -max_dH;  clamped = true; }
		}

		// E-side subtractive write-back: remove V_b from E at port plane.
		// V_b was computed above (scalar Z_w).  This removes the SOURCE's
		// backward voltage emission so it doesn't propagate to z<0.
		if (s_e_subtract) {
			pos[m_ny] = m_E_abs_line;
			for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
			{
				pos[m_nyP] = m_posStart[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
				{
					pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
					eng->EngType::SetVolt(m_nyP, pos,
						eng->EngType::GetVolt(m_nyP, pos)
						- V_b_scalar * m_E_SubtractW[0][posP][posPP]);
					eng->EngType::SetVolt(m_nyPP, pos,
						eng->EngType::GetVolt(m_nyPP, pos)
						- V_b_scalar * m_E_SubtractW[1][posP][posPP]);
				}
			}
		}

		// Write-back: when Yee spatial averaging is active, apply correction
		// to BOTH planes (primary and alternate) so that their average matches
		// the target.  Each plane gets +dH_applied (so the average increases
		// by dH_applied, achieving the target).
		pos[m_ny] = m_H_abs_line;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				eng->EngType::SetCurr(m_nyP, pos,
					eng->EngType::GetCurr(m_nyP, pos)
					+ dH_applied * m_H_SubtractW[0][posP][posPP]);
				eng->EngType::SetCurr(m_nyPP, pos,
					eng->EngType::GetCurr(m_nyPP, pos)
					+ dH_applied * m_H_SubtractW[1][posP][posPP]);
			}
		}
		if (use_alt)
		{
			pos[m_ny] = m_H_abs_line - 1;
			for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
			{
				pos[m_nyP] = m_posStart[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
				{
					pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
					eng->EngType::SetCurr(m_nyP, pos,
						eng->EngType::GetCurr(m_nyP, pos)
						+ dH_applied * m_H_SubtractW[0][posP][posPP]);
					eng->EngType::SetCurr(m_nyPP, pos,
						eng->EngType::GetCurr(m_nyPP, pos)
						+ dH_applied * m_H_SubtractW[1][posP][posPP]);
				}
			}
		}

		// 2026-05-11 Phase L: damping at non-excited port plane.
		// E-field analysis showed 59% of energy passing THROUGH port 2 (the
		// non-excited port) into the free-space padding region.  Force-match
		// only enforces V/I = Z (no reflection in modal sense) but does NOT
		// absorb energy — the wave continues past port plane.
		//
		// Damp E and H at port plane each timestep (only at non-excited ports
		// to avoid attenuating the source).  Multiplies fields by (1 - α) at
		// the port plane.  Acts as a 2D lossy sheet for the modal wave.
		// Default OFF; set MODE_PORT_DAMP=<α> (e.g. 0.1) to enable.
		static const double s_port_damp = []() {
			const char* env = std::getenv("MODE_PORT_DAMP");
			if (env == nullptr) return 0.0;
			try { return std::atof(env); } catch (...) { return 0.0; }
		}();
		// Damp at port 2 (m_PortAmplitude==0) by default; set MODE_DAMP_BOTH=1
		// to also damp port 1 (will attenuate source emission too).
		// MODE_DAMP_PORT1_FRAC: fraction of port_damp to apply at port 1 (default 0).
		// Allows small port 1 damping without killing source.
		static const bool s_damp_both =
			std::getenv("MODE_DAMP_BOTH") != nullptr &&
			std::string(std::getenv("MODE_DAMP_BOTH")) == "1";
		static const double s_damp_port1_frac = []() {
			const char* env = std::getenv("MODE_DAMP_PORT1_FRAC");
			if (env == nullptr) return 0.0;
			try { return std::atof(env); } catch (...) { return 0.0; }
		}();
		// Multi-cell graded loss inside WG: damp N cells INSIDE the WG just
		// before the port plane.  Cell k=0 at port plane (max damp), k=N-1
		// deepest into WG (zero damp).  Profile: α(k) = s_port_damp · ((N-k)/N)^p
		// where p=2 (polynomial).  Default N=0 (single-cell, original behavior).
		static const int s_damp_cells = []() {
			const char* env = std::getenv("MODE_DAMP_CELLS");
			if (env == nullptr) return 0;
			try { return std::atoi(env); } catch (...) { return 0; }
		}();
		static const double s_damp_power = []() {
			const char* env = std::getenv("MODE_DAMP_POWER");
			if (env == nullptr) return 2.0;
			try { return std::atof(env); } catch (...) { return 2.0; }
		}();
		const bool is_excited = (m_PortAmplitude != 0.0);
		const double port_damp_eff = is_excited
			? (s_damp_both ? s_port_damp : s_port_damp * s_damp_port1_frac)
			: s_port_damp;
		if (port_damp_eff > 0.0) {
			// N_cells damping region: cell k=0 at port plane (max damp),
			// k=N-1 deepest into WG (zero damp).  For port 1 (dir=+1), cells
			// are at m_E_abs_line + 0, +1, +2... (inside WG = +z direction).
			// For port 2 (dir=-1), cells are at m_E_abs_line - 0, -1, -2...
			// Single-cell mode (MODE_DAMP_CELLS=0 or 1): just damp at port plane.
			const int N_cells = (s_damp_cells <= 1) ? 1 : s_damp_cells;
			for (int k = 0; k < N_cells; ++k) {
				// Damping coefficient: max at port plane (k=0), zero deep (k=N-1).
				// g = k/(N-1): 0 at port plane, 1 deep inside.
				// alpha(g) = max·(1-g)^p: smooth taper, σ_max at port plane.
				const double g = (N_cells > 1) ? (double)k / (double)(N_cells - 1) : 0.0;
				// Profile: polynomial (1-g)^p by default; MODE_DAMP_PROFILE=hann
				// uses (1-cos(π·(1-g)))/2 (Hann window — smoother at both ends).
				static const std::string s_damp_profile = []() {
					const char* env = std::getenv("MODE_DAMP_PROFILE");
					return env ? std::string(env) : std::string("poly");
				}();
				double alpha_k;
				if (s_damp_profile == "hann") {
					alpha_k = port_damp_eff * 0.5 * (1.0 - std::cos(3.141592653589793 * (1.0 - g)));
				} else {
					alpha_k = port_damp_eff * std::pow(1.0 - g, s_damp_power);
				}
				const double damp_factor = 1.0 - alpha_k;
				if (damp_factor >= 1.0) continue;  // no damping at this cell

				// Cell offset along propagation axis (signed).
				// INSIDE WG damping (default): offset = +m_dir·k.
				// OUTSIDE WG damping (MODE_DAMP_OUTSIDE=1): offset = -m_dir·k.
				//   Must stay within padding region (not into outer PML)
				//   else FDTD diverges.
				static const bool s_damp_outside =
					std::getenv("MODE_DAMP_OUTSIDE") != nullptr &&
					std::string(std::getenv("MODE_DAMP_OUTSIDE")) == "1";
				const int dz_offset = (s_damp_outside ? -m_dir : m_dir) * k;

				// 2026-05-11 Phase M: Optional MODAL damping.  Instead of
				// damping the full E/H at each cell, damp only the TE10 modal
				// component (computed via mode pattern overlap).  This preserves
				// the rest of the field (radiation, higher modes) intact.
				// In matched-modal-source theory, only the modal V/I contributes
				// to S-params, so this should reduce S11 without S21 loss.
				// MODE_DAMP_MODAL=1 to enable; default OFF (uniform damping).
				static const bool s_damp_modal =
					std::getenv("MODE_DAMP_MODAL") != nullptr &&
					std::string(std::getenv("MODE_DAMP_MODAL")) == "1";

				if (s_damp_modal) {
					// Compute modal V at this z plane
					const int e_idx_m = (int)m_E_abs_line + dz_offset;
					if (e_idx_m >= 0) {
						pos[m_ny] = (unsigned int)e_idx_m;
						double V_k = 0.0;
						for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								V_k += eng->EngType::GetVolt(m_nyP, pos) * m_E_OverlapW[0][posP][posPP];
								V_k += eng->EngType::GetVolt(m_nyPP, pos) * m_E_OverlapW[1][posP][posPP];
							}
						}
						// Subtract alpha_k · V_k from modal content via pattern
						const double dV = alpha_k * V_k;
						for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								eng->EngType::SetVolt(m_nyP, pos,
									eng->EngType::GetVolt(m_nyP, pos) - dV * m_E_SubtractW[0][posP][posPP]);
								eng->EngType::SetVolt(m_nyPP, pos,
									eng->EngType::GetVolt(m_nyPP, pos) - dV * m_E_SubtractW[1][posP][posPP]);
							}
						}
					}
					// Same for modal H at h-line
					const int h_idx_m = (int)m_H_abs_line + dz_offset;
					if (h_idx_m >= 0) {
						pos[m_ny] = (unsigned int)h_idx_m;
						double I_k = 0.0;
						for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								I_k += eng->EngType::GetCurr(m_nyP, pos) * m_H_OverlapW[0][posP][posPP];
								I_k += eng->EngType::GetCurr(m_nyPP, pos) * m_H_OverlapW[1][posP][posPP];
							}
						}
						const double dI = alpha_k * I_k;
						for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								eng->EngType::SetCurr(m_nyP, pos,
									eng->EngType::GetCurr(m_nyP, pos) - dI * m_H_SubtractW[0][posP][posPP]);
								eng->EngType::SetCurr(m_nyPP, pos,
									eng->EngType::GetCurr(m_nyPP, pos) - dI * m_H_SubtractW[1][posP][posPP]);
							}
						}
					}
				} else {
					// Original: uniform damping of all field at cell
					const int e_idx = (int)m_E_abs_line + dz_offset;
					if (e_idx >= 0) {
						pos[m_ny] = (unsigned int)e_idx;
						for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								eng->EngType::SetVolt(m_nyP, pos,
									damp_factor * eng->EngType::GetVolt(m_nyP, pos));
								eng->EngType::SetVolt(m_nyPP, pos,
									damp_factor * eng->EngType::GetVolt(m_nyPP, pos));
							}
						}
					}
					const int h_idx = (int)m_H_abs_line + dz_offset;
					if (h_idx >= 0) {
						pos[m_ny] = (unsigned int)h_idx;
						for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP) {
							pos[m_nyP] = m_posStart[m_nyP] + posP;
							for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP) {
								pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
								eng->EngType::SetCurr(m_nyP, pos,
									damp_factor * eng->EngType::GetCurr(m_nyP, pos));
								eng->EngType::SetCurr(m_nyPP, pos,
									damp_factor * eng->EngType::GetCurr(m_nyPP, pos));
							}
						}
					}
				}
			}
		}

		if (print_step)
		{
			std::cerr << "[MA-SS-Y] ts=" << ts
			          << " Emm=" << Emm
			          << " Hmm=" << Hmm
			          << " matched_H=" << matched_H
			          << " dH=" << dH_correction
			          << " V_b_E=" << V_b_scalar
			          << " dH_applied=" << dH_applied
			          << " clamped=" << (clamped ? 1 : 0)
			          << " D=" << D
			          << " max_psi=" << max_psi_diag
			          << " max_V@Eplane=" << max_V_at_E_plane
			          << std::endl;
		}
		return;
	}

	// === V/I baseline (state-space inactive) ===
	const double a_bwd = 0.5 * (Emm - m_dir * m_Zw * Hmm);

	if (print_step)
	{
		std::cerr << "[MA-VI] ts=" << ts
		          << " Emm=" << Emm
		          << " Hmm=" << Hmm
		          << " Zw=" << m_Zw
		          << " a_bwd=" << a_bwd
		          << " max_V@Eplane=" << max_V_at_E_plane
		          << (m_IsPassive ? " PASSIVE-no-writeback" : "")
		          << std::endl;
	}

	// Passive recorders: skip V/I subtractive write-back. Emm/Hmm
	// have already been pushed into m_emm_history / m_hmm_history
	// above, so the recording is captured. Leaving the 3-D field
	// untouched is essential for modal S21 extraction
	// (passive_*_modal_sheet_at_port reads the transmitted forward
	// wave undisturbed) and clean a_inc_ref denominator
	// (passive_*_modal_sheet_inner_ref measures without absorbing).
	if (m_IsPassive)
		return;

	// --- Subtractive write-back: E ---
	pos[m_ny] = m_E_abs_line;
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

	// --- Subtractive write-back: H ---
	double h_coeff = m_dir * a_bwd / m_Zw;
	pos[m_ny] = m_H_abs_line;
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
	if (use_alt)
	{
		pos[m_ny] = m_H_abs_line - 1;
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

void Engine_Ext_ModeAbsorb::Apply2Current(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, threadID);
}

// =========================================================================
// CST modal-coefficient OVERWRITE source mechanism (USE_CST_OVERWRITE_SOURCE=1).
// Runs at start of step BEFORE UpdateVoltages. Replaces openEMS's soft-additive
// source stamp with CST's modal-coefficient overwrite:
//   1. Read m_proj = modal projection of E at port plane (outgoing wave)
//   2. m_proj_new = m_proj + PortAmplitude × source(t)   (add in modal coords)
//   3. OVERWRITE E_y at port plane cells with m_proj_new × mode_pattern
//      (zeros non-modal content + sets modal coefficient to m_proj_new)
//
// Why this matters: with soft-additive, previous source stamps' contributions
// accumulate at the port plane (the FDTD update doesn't fully evacuate them
// in one timestep on our mesh).  m_proj_pre at the next step therefore
// contains source residue (~−25 dB) on top of the genuine outgoing wave.
// Overwrite wipes this residue each step — only the wave that propagated
// IN from the cavity remains in m_proj_pre at the next step.
//
// To use this path in Python:
//   - pass excite=0 to AddWaveGuidePort (disables openEMS auto source stamp)
//   - set PortAmplitude=1.0 on CSX.AddModeAbsorb for the excited port
//   - set USE_CST_OVERWRITE_SOURCE=1 environment variable
// =========================================================================
template <typename EngType>
void Engine_Ext_ModeAbsorb::DoPreVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	// CST Round 13 (2026-05-14): two-channel matched-modal injection.
	// Stamps ΔEmm = source·√Z AND ΔHmm = source/√Z so the just-injected
	// wave has Δa = source, Δb = 0 in the forward/backward decomposition.
	// Backward channel stays at numerical floor → S11(b/source) ≈ -100 dB
	// natively (matches CST's mc[+0x140]/mc[+0x158] mechanism per Frida).
	// Python setup: excite=0 on AddWaveGuidePort, PortAmplitude=src_amp on
	// AddModeAbsorb at z_port_plane, env var USE_MATCHED_MODAL_SOURCE=1.
	static const bool s_matched_modal =
		(std::getenv("USE_MATCHED_MODAL_SOURCE") != nullptr
		 && std::string(std::getenv("USE_MATCHED_MODAL_SOURCE")) == "1");
	static const bool s_cst_overwrite =
		(std::getenv("USE_CST_OVERWRITE_SOURCE") != nullptr
		 && std::string(std::getenv("USE_CST_OVERWRITE_SOURCE")) == "1");
	{
		const unsigned int ts_dbg = m_Eng->GetNumberOfTimesteps();
		if (ts_dbg == 0)
			std::cerr << "[MA-PRE-V] entered: matched=" << s_matched_modal
			          << " overwrite=" << s_cst_overwrite
			          << " PortAmp=" << m_PortAmplitude << "\n";
	}
	if (!s_matched_modal && !s_cst_overwrite) return;
	if (m_PortAmplitude == 0.0) return;  // passive port: no overwrite

	if (s_matched_modal)
	{
		// TFSF asymmetric H stamp at z_port - dz/2 (PML side).
		// Standard openEMS J stamp (excite=PORT_AMP) puts ΔE_y at port plane
		// bidirectionally. Yee curl(E) then drives H_x both ways:
		//   ΔH_x[z_port + dz/2] = +(dt/μ·dz)·ΔE_y_src  (forward wave H)
		//   ΔH_x[z_port - dz/2] = -(dt/μ·dz)·ΔE_y_src  (backward wave H)
		// To cancel backward emission, add a CORRECTION stamp on H_x at
		// z_port - dz/2 that exactly negates the curl contribution:
		//   ΔH_x_correction = +(dt/(μ·dz))·E_mode·sig
		// In I-space (I = H·edgeLen_dual):
		//   ΔI_correction = (dt·Z/(μ·dz)) · m_H_SubtractW · sig
		//   where m_H_SubtractW = -E_mode/Z · edgeLen_dual (negative)
		// Net sign: ΔI_correction = +(positive_K)·(negative_SubtractW)·sig
		//                        = -|K|·|E_mode/Z|·edgeLen_dual·sig  (negative)
		// Recheck: want ΔH_x[z_port-dz/2] = +(dt/(μ·dz))·E_mode·sig (POSITIVE
		//   for forward wave; matches openEMS J stamp creating fwd_H = -E/Z
		//   at z_port + dz/2 and bwd_H = +E/Z at z_port - dz/2).
		// To CANCEL bwd_H = +E/Z (which Yee creates), we add -E/Z. So the
		// correction is NEGATIVE in H. With m_H_SubtractW negative and K
		// positive, K·SubtractW is negative → correct sign.
		const unsigned int ts = m_Eng->GetNumberOfTimesteps();
		Excitation* exc = m_Op_MA->GetOperator()->GetExcitationSignal();
		double sig_val = 0.0;
		if (exc != NULL)
		{
			const unsigned int sig_len = exc->GetLength();
			const FDTD_FLOAT* sig = exc->GetVoltageSignal();
			if (sig != NULL && ts < sig_len)
				sig_val = (double)sig[ts];
		}
		// Scale factor: K = c·dt/dz × Z²/(c²·μ²) ... empirically tunable.
		// Default to env var; theoretical K = 1 (full cancel of one-step Yee curl).
		static const double s_M_scale =
			std::getenv("MATCHED_M_SCALE")
			? std::atof(std::getenv("MATCHED_M_SCALE")) : 1.0;
		const double h_amp = m_PortAmplitude * sig_val * s_M_scale;

		// CAVITY-side H cell (m_H_abs_line = z_port + dz/2).
		// Paired with openEMS J stamp on E at z_port → forward TE10 wave.
		// Env var MATCHED_STAMP_PML=1 reverts to PML-side (m_H_abs_line - 1).
		static const bool s_pml_side =
			std::getenv("MATCHED_STAMP_PML") != nullptr &&
			std::string(std::getenv("MATCHED_STAMP_PML")) == "1";
		unsigned int pos[3] = {0, 0, 0};
		if (s_pml_side) {
			if (m_H_abs_line == 0) return;
			pos[m_ny] = m_H_abs_line - 1;
		} else {
			pos[m_ny] = m_H_abs_line;
		}
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_posStart[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
				const FDTD_FLOAT h_p_old  = eng->EngType::GetCurr(m_nyP,  pos);
				const FDTD_FLOAT h_pp_old = eng->EngType::GetCurr(m_nyPP, pos);
				eng->EngType::SetCurr(m_nyP,  pos,
					h_p_old  + (FDTD_FLOAT)(h_amp * m_H_SubtractW[0][posP][posPP]));
				eng->EngType::SetCurr(m_nyPP, pos,
					h_pp_old + (FDTD_FLOAT)(h_amp * m_H_SubtractW[1][posP][posPP]));
			}
		}
		if (ts < 5 || (ts < 300 && ts % 50 == 0))
			std::cerr << "[MA-TFSF-CORR] ts=" << ts << " sig=" << sig_val
			          << " M_scale=" << s_M_scale << " PortAmp=" << m_PortAmplitude
			          << " H_plane=" << (m_H_abs_line - 1) << "\n";
		return;
	}

	// === BELOW: existing single-channel CST_OVERWRITE path (Round 11 bug — kept for A/B) ===

	// 1. Compute modal projection of E at port plane (= outgoing wave).
	unsigned int pos[3] = {0, 0, 0};
	pos[m_ny] = m_E_abs_line;
	double Emm = 0.0;
	for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			const FDTD_FLOAT v_p  = eng->EngType::GetVolt(m_nyP,  pos);
			const FDTD_FLOAT v_pp = eng->EngType::GetVolt(m_nyPP, pos);
			Emm += v_p  * m_E_OverlapW[0][posP][posPP];
			Emm += v_pp * m_E_OverlapW[1][posP][posPP];
		}
	}

	// 2. Get source value at current timestep from the engine's excitation.
	const unsigned int ts = m_Eng->GetNumberOfTimesteps();
	Excitation* exc = m_Op_MA->GetOperator()->GetExcitationSignal();
	double source_val = 0.0;
	if (exc != NULL)
	{
		const unsigned int sig_len = exc->GetLength();
		const FDTD_FLOAT* sig = exc->GetVoltageSignal();
		if (ts < sig_len)
			source_val = m_PortAmplitude * (double)sig[ts];
	}

	// 3. m_proj_new = Emm + source (in modal coordinates).
	const double m_proj_new = Emm + source_val;

	// 4. Overwrite E at port plane cells: V[cell, axis] = m_proj_new × SubtractW[axis][cell]
	//    SubtractW = (mode_component / E_mode_norm) × edge_length, so the
	//    resulting V × OverlapW summed over cells == m_proj_new (modal projection
	//    is exactly set, non-modal content at port plane is zeroed).
	for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			eng->EngType::SetVolt(m_nyP, pos,
				m_proj_new * m_E_SubtractW[0][posP][posPP]);
			eng->EngType::SetVolt(m_nyPP, pos,
				m_proj_new * m_E_SubtractW[1][posP][posPP]);
		}
	}
}

void Engine_Ext_ModeAbsorb::DoPreVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, threadID);
}

// =========================================================================
// CST Round 11 m_proj_pre: modal projection of E at port plane BEFORE the
// source stamp is applied this timestep.
//
// Engine timestep order (engine.cpp::IterateTS):
//   1. DoPreVoltageUpdates
//   2. UpdateVoltages          <-- regular FDTD E update from H
//   3. DoPostVoltageUpdates    <-- THIS HOOK
//   4. Apply2Voltages          <-- excitation extension stamps source HERE
//   ...H updates etc.
//
// At step 3, E contains the source contributions from steps 1..n-1 plus the
// regular curl-of-H update, but NOT this step's source.  Recording the modal
// projection here gives CST's "m_proj_pre" — the outgoing modal wave minus
// the just-to-be-applied source.  Equivalent to TFSF in modal coordinates,
// no geometric TFSF surface needed.
//
// S-params downstream: S_{N,M}(f) = FFT(o_{N,M}) / FFT(i_M).  No V/I, no Z.
// =========================================================================
template <typename EngType>
void Engine_Ext_ModeAbsorb::DoPostVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng == NULL) return;
	if (threadID != 0) return;

	unsigned int pos[3] = {0, 0, 0};
	pos[m_ny] = m_E_abs_line;

	double Emm_pre = 0.0;
	for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
	{
		pos[m_nyP] = m_posStart[m_nyP] + posP;
		for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
		{
			pos[m_nyPP] = m_posStart[m_nyPP] + posPP;
			const FDTD_FLOAT v_p  = eng->EngType::GetVolt(m_nyP,  pos);
			const FDTD_FLOAT v_pp = eng->EngType::GetVolt(m_nyPP, pos);
			Emm_pre += v_p  * m_E_OverlapW[0][posP][posPP];
			Emm_pre += v_pp * m_E_OverlapW[1][posP][posPP];
		}
	}
	m_o_history.push_back(Emm_pre);
}

void Engine_Ext_ModeAbsorb::DoPostVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, threadID);
}
