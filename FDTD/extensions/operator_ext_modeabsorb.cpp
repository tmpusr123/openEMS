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

#include "operator_ext_modeabsorb.h"
#include "engine_ext_modeabsorb.h"

#include "tools/array_ops.h"
#include "tools/constants.h"

#include "CSPrimBox.h"
#include "CSPropModeAbsorb.h"
#include "CSModeFileParser.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using std::cerr;
using std::endl;

Operator_Ext_ModeAbsorb::Operator_Ext_ModeAbsorb(Operator* op) : Operator_Extension(op)
{
	Initialize();
}

Operator_Ext_ModeAbsorb::Operator_Ext_ModeAbsorb(Operator* op, Operator_Ext_ModeAbsorb* op_ext)
	: Operator_Extension(op, op_ext)
{
	Initialize();
	m_prop = op_ext->m_prop;
	m_prim = op_ext->m_prim;
}

Operator_Ext_ModeAbsorb::~Operator_Ext_ModeAbsorb()
{
	for (int n = 0; n < 2; ++n)
	{
		Delete2DArray<double>(m_E_OverlapW[n],  m_numLines_E);
		Delete2DArray<double>(m_E_SubtractW[n], m_numLines_E);
		Delete2DArray<double>(m_H_OverlapW[n],  m_numLines_H);
		Delete2DArray<double>(m_H_SubtractW[n], m_numLines_H);
	}
}

bool Operator_Ext_ModeAbsorb::SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop)
{
	m_prim = prim;
	m_prop = prop;
	return (prim != NULL && prop != NULL);
}

Operator_Extension* Operator_Ext_ModeAbsorb::Clone(Operator* op)
{
	return new Operator_Ext_ModeAbsorb(op, this);
}

void Operator_Ext_ModeAbsorb::Initialize()
{
	m_prop = NULL;
	m_prim = NULL;
	m_ny = m_nyP = m_nyPP = -1;
	m_dir = 1;
	m_srcLine = 0;
	m_E_abs_line = 0;
	m_H_abs_line = 0;
	m_Zw = 376.73;

	for (int i = 0; i < 3; ++i)
	{
		m_start[i] = 0;
		m_stop[i]  = 0;
	}

	m_numLines_E[0] = m_numLines_E[1] = 0;
	m_numLines_H[0] = m_numLines_H[1] = 0;

	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n]  = NULL;
		m_E_SubtractW[n] = NULL;
		m_H_OverlapW[n]  = NULL;
		m_H_SubtractW[n] = NULL;
	}

	m_UseStateSpace = false;
	m_SS_n = 0;
	m_SS_D = 0.0;
	m_SS_P.clear();
	m_SS_Q.clear();
	m_SS_C.clear();
}

bool Operator_Ext_ModeAbsorb::BuildExtension()
{
	if (!m_prop || !m_prim)
	{
		SetActive(false);
		return false;
	}

	// 1. Snap user box (zero-thickness in propagation direction) to mesh.
	CSPrimBox* box = dynamic_cast<CSPrimBox*>(m_prim);
	if (!box)
	{
		cerr << "Operator_Ext_ModeAbsorb::BuildExtension: primitive is not a Box." << endl;
		SetActive(false);
		return false;
	}

	unsigned int boxStart[3], boxStop[3];
	m_Op->SnapBox2Mesh(
		box->GetStartCoord()->GetCoords(m_Op->m_MeshType),
		box->GetStopCoord()->GetCoords(m_Op->m_MeshType),
		boxStart, boxStop, false, true);

	// 2. Find the propagation axis: the one where boxStart == boxStop.
	int zero_axis_count = 0;
	m_ny = -1;
	for (int i = 0; i < 3; ++i)
	{
		if (boxStart[i] == boxStop[i])
		{
			m_ny = i;
			++zero_axis_count;
		}
	}

	if (m_ny < 0 || zero_axis_count != 1)
	{
		cerr << "Operator_Ext_ModeAbsorb::BuildExtension: cannot determine propagation axis "
		        "from box geometry (need exactly one degenerate axis, got "
		     << zero_axis_count << ")." << endl;
		SetActive(false);
		return false;
	}

	m_nyP  = (m_ny + 1) % 3;
	m_nyPP = (m_ny + 2) % 3;

	// Direction: NormalSignPositive == true means we're absorbing waves
	// travelling in +m_ny direction (forward), so the absorber sheet sits at
	// the −m_ny side of the source; the kernel's a_bwd formula uses the
	// propagation sign of the wave it absorbs.
	m_dir = m_prop->GetNormalSignPositive() ? +1 : -1;

	// Sheet position in m_ny direction
	m_srcLine = boxStart[m_ny];

	for (int i = 0; i < 3; ++i)
	{
		m_start[i] = boxStart[i];
		m_stop[i]  = boxStop[i];
	}

	// Single-layer absorber: E plane at the user-supplied sheet, H plane
	// configurable.  2026-05-11 Phase G: previously H was offset by 1 cell
	// from E (asymmetric: -1 for dir=+1, 0 for dir=-1), creating a ~1.5·dz
	// spatial offset between V and I samples (cell offset + Yee dual-mesh
	// dz/2).  At f=2 GHz this gives a 22° phase mismatch in the
	// a_bwd = 0.5(V − Z·I) formula, contributing to the −10 dB |S11| floor.
	// Default now: H at the same line as E (only the Yee dz/2 dual-mesh
	// offset remains).  Set MODE_H_PLANE_OFFSET=1 to revert old behavior.
	m_E_abs_line = m_srcLine;
	int _h_offset = 0;
	if (const char* env = std::getenv("MODE_H_PLANE_OFFSET")) {
		try { _h_offset = std::atoi(env); } catch (...) {}
	}
	if (_h_offset != 0) {
		m_H_abs_line = (m_dir > 0)
		             ? (m_srcLine > 0 ? m_srcLine - _h_offset : m_srcLine)
		             :  m_srcLine + _h_offset;
	} else {
		m_H_abs_line = m_srcLine;
	}
	std::cerr << "[ModeAbsorb] H plane offset from E: " << _h_offset
	          << "  E_abs_line=" << m_E_abs_line
	          << "  H_abs_line=" << m_H_abs_line
	          << "  dir=" << m_dir << std::endl;

	unsigned int Ncells[3];
	for (int i = 0; i < 3; ++i)
		Ncells[i] = (m_stop[i] >= m_start[i]) ? (m_stop[i] - m_start[i] + 1) : 1;

	m_numLines_E[0] = Ncells[m_nyP];
	m_numLines_E[1] = Ncells[m_nyPP];

	m_numLines_H[0] = Ncells[m_nyP]  > 0 ? Ncells[m_nyP]  - 1 : 0;
	m_numLines_H[1] = Ncells[m_nyPP] > 0 ? Ncells[m_nyPP] - 1 : 0;

	// 3. Wave impedance: take from CSPropModeAbsorb (set by user from mode
	// solver). Fall back to free-space if zero/missing.
	double zw_user = m_prop->GetWaveImpedance();
	m_Zw = (zw_user > 0.0) ? zw_user : 376.73;

	// 3b. State-space mode (CST CBBPortOperatorStateSpace equivalent).  When
	//     UseStateSpace=true, the engine bypasses V/I matched-modal-source and
	//     runs the per-mode Crank-Nicolson recursion using pre-computed P, Q,
	//     C, D matrices supplied from the Python side (Gustavsen vector fit
	//     of tanh(γ_m(ω)·Δz) target).  Validate sizes; if anything looks off,
	//     warn and fall back to V/I.
	m_UseStateSpace = m_prop->GetUseStateSpace();
	m_SS_n          = m_prop->GetSSOrder();
	m_SS_P          = m_prop->GetSSP();
	m_SS_Q          = m_prop->GetSSQ();
	m_SS_C          = m_prop->GetSSC();
	m_SS_D          = m_prop->GetSSD();
	if (m_UseStateSpace)
	{
		bool ok = (m_SS_n > 0
		           && m_SS_P.size() == (size_t)m_SS_n * m_SS_n
		           && m_SS_Q.size() == (size_t)m_SS_n
		           && m_SS_C.size() == (size_t)m_SS_n);
		if (!ok)
		{
			cerr << "Operator_Ext_ModeAbsorb: UseStateSpace=true but matrix sizes "
			     << "are inconsistent (n=" << m_SS_n << ", |P|=" << m_SS_P.size()
			     << ", |Q|=" << m_SS_Q.size() << ", |C|=" << m_SS_C.size()
			     << ").  Falling back to V/I matched-modal-source." << endl;
			m_UseStateSpace = false;
			m_SS_n = 0;
			m_SS_P.clear();
			m_SS_Q.clear();
			m_SS_C.clear();
		}
	}

	// 4. Parse the E and H CSV mode files. The Python side wrote them as
	//    columns:  x, y, V_first_transverse, V_second_transverse
	// where x,y are in dimensions_unit, shifted so the box's corner is at 0.
	std::string E_file = m_prop->GetEModeFileName();
	std::string H_file = m_prop->GetHModeFileName();

	if (E_file.empty() || H_file.empty())
	{
		cerr << "Operator_Ext_ModeAbsorb::BuildExtension: missing E/H mode file name." << endl;
		SetActive(false);
		return false;
	}

	CSModeFileParser E_parser(E_file);
	CSModeFileParser H_parser(H_file);

	if (!E_parser.IsFileParsed())
	{
		cerr << "Operator_Ext_ModeAbsorb::BuildExtension: failed to parse E mode file '" << E_file << "'." << endl;
		SetActive(false);
		return false;
	}
	if (!H_parser.IsFileParsed())
	{
		cerr << "Operator_Ext_ModeAbsorb::BuildExtension: failed to parse H mode file '" << H_file << "'." << endl;
		SetActive(false);
		return false;
	}

	// The CSV's first two columns are x,y in DRAWING UNITS, with the origin
	// shifted to the LOW corner of the box that genPortGeoFromSTLs sampled
	// the mode on (which is the WG cross-section, not the full simulation
	// transverse domain — so the CSV bounds are just [0, a] × [0, b], where
	// a, b are the WG dimensions). Anchor our lookup origin at the box's
	// own start mesh-line position; that puts xq=0 at csv_x=0, xq=a at
	// csv_x=a, lined up with the stored samples.
	double x0_csv = m_Op->GetDiscLine(m_nyP,  m_start[m_nyP],  false);
	double y0_csv = m_Op->GetDiscLine(m_nyPP, m_start[m_nyPP], false);

	// 5. Allocate weight arrays.
	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n]  = Create2DArray<double>(m_numLines_E);
		m_E_SubtractW[n] = Create2DArray<double>(m_numLines_E);
	}

	// 6. Build E-mode distribution on primary mesh. Normalisation matches
	// the WaveguideAbsorber template: L2-normalise on the primary mesh, then
	// reuse that norm for the H side so a_bwd vanishes for a pure forward
	// wave.
	double E_mode_norm = 0;
	{
		double** modeTmp[2];
		for (int n = 0; n < 2; ++n)
			modeTmp[n] = Create2DArray<double>(m_numLines_E);

		unsigned int pos[3] = {0, 0, 0};
		pos[m_ny] = m_start[m_ny];
		double norm = 0;

		for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
		{
			pos[m_nyP] = m_start[m_nyP] + posP;
			double xq = m_Op->GetDiscLine(m_nyP, pos[m_nyP], false) - x0_csv;
			for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
			{
				pos[m_nyPP] = m_start[m_nyPP] + posPP;
				double yq = m_Op->GetDiscLine(m_nyPP, pos[m_nyPP], false) - y0_csv;

				double mP = 0, mPP = 0;
				E_parser.LinInterp2(xq, yq, mP, mPP);
				modeTmp[0][posP][posPP] = mP;
				modeTmp[1][posP][posPP] = mPP;

				double area = m_Op->GetNodeArea(m_ny, pos, false);
				norm += (mP*mP + mPP*mPP) * area;
			}
		}
		norm = std::sqrt(norm);
		E_mode_norm = norm;

		cerr << "[ModeAbsorb] " << E_file
		     << "  (m_ny=" << m_ny << " dir=" << m_dir
		     << " sheet=" << m_srcLine << " Zw=" << m_Zw
		     << ")  E_mode_norm=" << norm
		     << "  size=" << m_numLines_E[0] << "x" << m_numLines_E[1]
		     << endl;
		cerr << "[ModeAbsorb] state-space: "
		     << (m_UseStateSpace ? "ACTIVE" : "INACTIVE-V/I-fallback")
		     << "  SS_n=" << m_SS_n
		     << "  D=" << m_SS_D
		     << "  PortAmplitude=" << (m_prop ? m_prop->GetPortAmplitude() : 0.0)
		     << endl;

		if (norm > 0)
		{
			for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
			{
				pos[m_nyP] = m_start[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
				{
					pos[m_nyPP] = m_start[m_nyPP] + posPP;

					double area      = m_Op->GetNodeArea(m_ny, pos, false);
					double edgeLen_P  = m_Op->GetEdgeLength(m_nyP,  pos, false);
					double edgeLen_PP = m_Op->GetEdgeLength(m_nyPP, pos, false);

					double mP  = modeTmp[0][posP][posPP] / norm;
					double mPP = modeTmp[1][posP][posPP] / norm;

					m_E_OverlapW[0][posP][posPP]  = (edgeLen_P  > 0) ? mP  * area / edgeLen_P  : 0;
					m_E_OverlapW[1][posP][posPP]  = (edgeLen_PP > 0) ? mPP * area / edgeLen_PP : 0;
					m_E_SubtractW[0][posP][posPP] = mP  * edgeLen_P;
					m_E_SubtractW[1][posP][posPP] = mPP * edgeLen_PP;
				}
			}
		}

		for (int n = 0; n < 2; ++n)
			Delete2DArray<double>(modeTmp[n], m_numLines_E);
	}

	// 7. Build H-mode distribution on dual mesh.
	//
	// 2026-05-10: switched from independent L2-normalisation to
	// power-coupled normalisation per CST-LLM Round-2 review (Step 2).
	// Old: H_mode_norm = √∫|h_pattern|² (L2 of H separately).  This
	// makes a_bwd = 0 only at the single frequency where the mode was
	// solved (Z_TE(f_mode_solver) ≈ ratio of L2(E)/L2(H) for physical
	// field amplitudes).  At off-mode frequencies, V/I residual exists
	// regardless of how good Z_w is.
	// New: H_mode_norm = E_mode_norm / Z_w  (H side scaled by E side via
	// the impedance used in the V/I formula).  This is the standard
	// "modal power = unity" convention from CST and implements
	// `Re(Emm·Hmm*) = forward modal power` directly.  With Z_w
	// recomputed at the band's geometric center (see ports.py), the
	// V/I formula a_bwd = 0.5(Emm − dir·Z_w·Hmm) is properly matched
	// across the propagating band, not just at f_mode_solver.
	for (int n = 0; n < 2; ++n)
	{
		m_H_OverlapW[n]  = Create2DArray<double>(m_numLines_H);
		m_H_SubtractW[n] = Create2DArray<double>(m_numLines_H);
	}

	{
		double** modeTmp[2];
		for (int n = 0; n < 2; ++n)
			modeTmp[n] = Create2DArray<double>(m_numLines_H);

		unsigned int pos[3] = {0, 0, 0};
		pos[m_ny] = m_start[m_ny];

		double H_norm_sq = 0.0;
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_start[m_nyP] + posP;
			double xq = m_Op->GetDiscLine(m_nyP, pos[m_nyP], true) - x0_csv;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_start[m_nyPP] + posPP;
				double yq = m_Op->GetDiscLine(m_nyPP, pos[m_nyPP], true) - y0_csv;

				double mP = 0, mPP = 0;
				H_parser.LinInterp2(xq, yq, mP, mPP);
				modeTmp[0][posP][posPP] = mP;
				modeTmp[1][posP][posPP] = mPP;

				double area = m_Op->GetNodeArea(m_ny, pos, true);
				H_norm_sq += (mP*mP + mPP*mPP) * area;
			}
		}

		const double norm_l2  = std::sqrt(H_norm_sq);
		const double norm_pwr = (m_Zw > 0.0) ? (E_mode_norm / m_Zw) : norm_l2;
		// Power-coupled normalisation (default).  Set
		// MODE_ABSORB_USE_L2_HNORM=1 to revert to old independent-L2.
		const bool use_l2 = (std::getenv("MODE_ABSORB_USE_L2_HNORM") != nullptr
		                     && std::string(std::getenv("MODE_ABSORB_USE_L2_HNORM")) == "1");
		double norm = use_l2 ? norm_l2 : norm_pwr;
		cerr << "[ModeAbsorb] " << H_file
		     << "  H_mode_norm_L2=" << norm_l2
		     << "  H_mode_norm_pwr=" << norm_pwr
		     << "  used=" << (use_l2 ? "L2" : "pwr")
		     << "  E_norm/H_norm=" << (norm_l2 > 0 ? E_mode_norm/norm_l2 : 0.0)
		     << " (≈Z_TE@f_mode_solver)"
		     << "  size=" << m_numLines_H[0] << "x" << m_numLines_H[1]
		     << endl;

		if (norm > 0)
		{
			for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
			{
				pos[m_nyP] = m_start[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
				{
					pos[m_nyPP] = m_start[m_nyPP] + posPP;

					double area       = m_Op->GetNodeArea(m_ny, pos, true);
					double edgeLen_P  = m_Op->GetEdgeLength(m_nyP,  pos, true);
					double edgeLen_PP = m_Op->GetEdgeLength(m_nyPP, pos, true);

					double mP  = modeTmp[0][posP][posPP] / norm;
					double mPP = modeTmp[1][posP][posPP] / norm;

					m_H_OverlapW[0][posP][posPP]  = (edgeLen_P  > 0) ? mP  * area / edgeLen_P  : 0;
					m_H_OverlapW[1][posP][posPP]  = (edgeLen_PP > 0) ? mPP * area / edgeLen_PP : 0;
					m_H_SubtractW[0][posP][posPP] = mP  * edgeLen_P;
					m_H_SubtractW[1][posP][posPP] = mPP * edgeLen_PP;
				}
			}
		}

		for (int n = 0; n < 2; ++n)
			Delete2DArray<double>(modeTmp[n], m_numLines_H);
	}

	return true;
}

Engine_Extension* Operator_Ext_ModeAbsorb::CreateEngineExtention()
{
	return new Engine_Ext_ModeAbsorb(this);
}

void Operator_Ext_ModeAbsorb::ShowStat(std::ostream& ostr) const
{
	Operator_Extension::ShowStat(ostr);
	ostr << " Propagation dir: " << m_ny << " sign: " << m_dir << endl;
	ostr << " Sheet line:      " << m_srcLine << endl;
	ostr << " E-mode lines:    " << m_numLines_E[0] << " x " << m_numLines_E[1] << endl;
	ostr << " H-mode lines:    " << m_numLines_H[0] << " x " << m_numLines_H[1] << endl;
	ostr << " Wave impedance:  " << m_Zw << " Ohm" << endl;
	ostr << " E_abs_line=" << m_E_abs_line << " H_abs_line=" << m_H_abs_line << endl;
}
