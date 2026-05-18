/*
*	Copyright (C) 2025
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#include "operator_ext_modalfdtd.h"
#include "engine_ext_modalfdtd.h"

#include "tools/array_ops.h"
#include "tools/constants.h"

#include "CSPrimBox.h"
#include "CSPropModeAbsorb.h"
#include "CSModeFileParser.h"

#include <cmath>
#include <iostream>

using std::cerr;
using std::endl;

Operator_Ext_ModalFDTD::Operator_Ext_ModalFDTD(Operator* op) : Operator_Extension(op)
{
	Initialize();
}

Operator_Ext_ModalFDTD::Operator_Ext_ModalFDTD(Operator* op, Operator_Ext_ModalFDTD* op_ext)
	: Operator_Extension(op, op_ext)
{
	Initialize();
	m_prop = op_ext->m_prop;
	m_prim = op_ext->m_prim;
}

Operator_Ext_ModalFDTD::~Operator_Ext_ModalFDTD()
{
	for (int n = 0; n < 2; ++n)
	{
		Delete2DArray<double>(m_E_OverlapW[n],  m_numLines_E);
		Delete2DArray<double>(m_E_SubtractW[n], m_numLines_E);
		Delete2DArray<double>(m_H_OverlapW[n],  m_numLines_H);
		Delete2DArray<double>(m_H_SubtractW[n], m_numLines_H);
	}
}

void Operator_Ext_ModalFDTD::Initialize()
{
	m_prop = NULL;
	m_prim = NULL;
	m_ny = m_nyP = m_nyPP = -1;
	m_dir = 1;
	m_srcLine = 0;
	m_H_in_line = 0;
	m_H_override_line = 0;
	for (int i = 0; i < 3; ++i) { m_start[i] = 0; m_stop[i] = 0; }
	m_numLines_E[0] = m_numLines_E[1] = 0;
	m_numLines_H[0] = m_numLines_H[1] = 0;
	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n]  = NULL; m_E_SubtractW[n] = NULL;
		m_H_OverlapW[n]  = NULL; m_H_SubtractW[n] = NULL;
	}
	m_N1D       = 1000;
	m_dz        = 0;
	m_dt        = 0;
	m_kc        = 0;
	m_omega_c   = 0;
	m_c_centre  = 0;
	m_c_neigh   = 0;
	m_mur_coef  = 0;
}

bool Operator_Ext_ModalFDTD::SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop)
{
	m_prim = prim;
	m_prop = prop;
	return (prim != NULL && prop != NULL);
}

Operator_Extension* Operator_Ext_ModalFDTD::Clone(Operator* op)
{
	return new Operator_Ext_ModalFDTD(op, this);
}

bool Operator_Ext_ModalFDTD::BuildExtension()
{
	if (!m_prop || !m_prim)
	{
		SetActive(false);
		return false;
	}

	// 1. Snap user box to mesh; identify propagation and transverse axes
	CSPrimBox* box = dynamic_cast<CSPrimBox*>(m_prim);
	if (!box)
	{
		cerr << "Operator_Ext_ModalFDTD: primitive is not a Box." << endl;
		SetActive(false);
		return false;
	}

	unsigned int boxStart[3], boxStop[3];
	m_Op->SnapBox2Mesh(
		box->GetStartCoord()->GetCoords(m_Op->m_MeshType),
		box->GetStopCoord()->GetCoords(m_Op->m_MeshType),
		boxStart, boxStop, false, true);

	int zero_axis_count = 0;
	m_ny = -1;
	for (int i = 0; i < 3; ++i)
	{
		if (boxStart[i] == boxStop[i]) { m_ny = i; ++zero_axis_count; }
	}
	if (m_ny < 0 || zero_axis_count != 1)
	{
		cerr << "Operator_Ext_ModalFDTD: cannot determine propagation axis "
		        "(need exactly one degenerate axis, got " << zero_axis_count
		     << ")." << endl;
		SetActive(false);
		return false;
	}

	m_nyP  = (m_ny + 1) % 3;
	m_nyPP = (m_ny + 2) % 3;
	m_dir  = m_prop->GetNormalSignPositive() ? +1 : -1;

	m_srcLine = boxStart[m_ny];
	for (int i = 0; i < 3; ++i)
	{
		m_start[i] = boxStart[i];
		m_stop[i]  = boxStop[i];
	}

	// Cell mapping for transparent boundary.
	//
	// `m_dir` is the V/I forward-propagation convention used by the
	// matched-modal-source absorber: m_dir=+1 means "forward = +z"
	// (structure / WG continues in +z relative to the absorber sheet).
	// In that case the source / source-side structure lies on the +z side
	// of the sheet, so the wave arriving at the absorber comes from +z.
	//
	// Yee staggering: H[i] sits between E[i] and E[i+1].  Therefore
	//   H[m_srcLine]     is on the +z side of the sheet (between absorber
	//                    sheet E plane and the source/structure).
	//   H[m_srcLine - 1] is on the -z side of the sheet (the empty mesh
	//                    that we treat as the 1-D extension's foothold).
	//
	// dir=+1 → wave from +z, extension in -z:
	//   m_H_in_line       = m_srcLine     (sample where wave arrives)
	//   m_H_override_line = m_srcLine - 1 (write 1-D-supplied H here)
	// dir=-1 mirrors this.
	if (m_dir > 0)
	{
		m_H_in_line       = m_srcLine;
		m_H_override_line = (m_srcLine > 0) ? m_srcLine - 1 : 0;
	}
	else
	{
		m_H_in_line       = m_srcLine - 1;
		m_H_override_line = m_srcLine;
	}

	unsigned int Ncells[3];
	for (int i = 0; i < 3; ++i)
		Ncells[i] = (m_stop[i] >= m_start[i]) ? (m_stop[i] - m_start[i] + 1) : 1;
	m_numLines_E[0] = Ncells[m_nyP];
	m_numLines_E[1] = Ncells[m_nyPP];
	m_numLines_H[0] = Ncells[m_nyP]  > 0 ? Ncells[m_nyP]  - 1 : 0;
	m_numLines_H[1] = Ncells[m_nyPP] > 0 ? Ncells[m_nyPP] - 1 : 0;

	// 2. Modal cutoff and timestep / cell size for 1-D Yee
	m_kc      = m_prop->GetKc();
	m_omega_c = m_kc * __C0__;
	m_dt      = m_Op->GetTimestep();

	// dz: 3-D mesh spacing along m_ny at port plane (between m_srcLine and
	// the line one cell toward the source side). GetDiscLine returns
	// positions in DRAWING UNITS (e.g. mm) — multiply by GetGridDelta() to
	// convert to METERS, so the Courant formula `r = c * dt / dz` is
	// dimensionally consistent (dt seconds, c m/s, dz meters).
	{
		double z0 = m_Op->GetDiscLine(m_ny, m_srcLine, false);
		unsigned int neighbor = (m_dir > 0) ? (m_srcLine - 1) : (m_srcLine + 1);
		double z1 = m_Op->GetDiscLine(m_ny, neighbor, false);
		m_dz = std::abs(z1 - z0) * m_Op->GetGridDelta();
	}
	if (m_dz <= 0)
	{
		cerr << "Operator_Ext_ModalFDTD: invalid dz at port plane." << endl;
		SetActive(false);
		return false;
	}

	double rate = __C0__ * m_dt / m_dz;
	m_c_centre = 2.0 * (1.0 - rate*rate - 0.5 * m_omega_c*m_omega_c * m_dt*m_dt);
	m_c_neigh  = rate * rate;
	m_mur_coef = (rate - 1.0) / (rate + 1.0);

	unsigned int n1d_user = m_prop->GetN1D();
	m_N1D = (n1d_user > 0) ? n1d_user : 2000u;
	if (m_N1D < 100) m_N1D = 100;

	// 3. Parse mode files and build projection weights at port plane
	std::string E_file = m_prop->GetEModeFileName();
	if (E_file.empty())
	{
		cerr << "Operator_Ext_ModalFDTD: missing E mode file name." << endl;
		SetActive(false);
		return false;
	}
	CSModeFileParser E_parser(E_file);
	if (!E_parser.IsFileParsed())
	{
		cerr << "Operator_Ext_ModalFDTD: failed to parse E mode file '"
		     << E_file << "'." << endl;
		SetActive(false);
		return false;
	}

	double x0_csv = m_Op->GetDiscLine(m_nyP,  m_start[m_nyP],  false);
	double y0_csv = m_Op->GetDiscLine(m_nyPP, m_start[m_nyPP], false);

	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n]  = Create2DArray<double>(m_numLines_E);
		m_E_SubtractW[n] = Create2DArray<double>(m_numLines_E);
	}

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

	cerr << "[ModalFDTD] " << E_file
	     << "  (m_ny=" << m_ny << " dir=" << m_dir
	     << " sheet=" << m_srcLine << " kc=" << m_kc << ")"
	     << "  E_mode_norm=" << norm
	     << "  size=" << m_numLines_E[0] << "x" << m_numLines_E[1]
	     << "  dz=" << m_dz << " dt=" << m_dt
	     << "  rate=" << rate << " N1D=" << m_N1D
	     << " c_centre=" << m_c_centre << " c_neigh=" << m_c_neigh
	     << " mur_coef=" << m_mur_coef
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

	// 4. Parse H mode and build H weights (analogous to Operator_Ext_ModeAbsorb).
	// We need m_H_OverlapW (project 3-D H onto h_mode at m_H_in_line) and
	// m_H_SubtractW (reconstruct H pattern from a[1] at m_H_override_line).
	std::string H_file = m_prop->GetHModeFileName();
	if (H_file.empty())
	{
		cerr << "Operator_Ext_ModalFDTD: missing H mode file name." << endl;
		SetActive(false);
		return false;
	}
	CSModeFileParser H_parser(H_file);
	if (!H_parser.IsFileParsed())
	{
		cerr << "Operator_Ext_ModalFDTD: failed to parse H mode file '"
		     << H_file << "'." << endl;
		SetActive(false);
		return false;
	}

	for (int n = 0; n < 2; ++n)
	{
		m_H_OverlapW[n]  = Create2DArray<double>(m_numLines_H);
		m_H_SubtractW[n] = Create2DArray<double>(m_numLines_H);
	}

	double** modeTmpH[2];
	for (int n = 0; n < 2; ++n)
		modeTmpH[n] = Create2DArray<double>(m_numLines_H);

	double H_norm_sq = 0.0;
	{
		unsigned int posH[3] = {0, 0, 0};
		posH[m_ny] = m_start[m_ny];
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			posH[m_nyP] = m_start[m_nyP] + posP;
			double xq = m_Op->GetDiscLine(m_nyP, posH[m_nyP], true) - x0_csv;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				posH[m_nyPP] = m_start[m_nyPP] + posPP;
				double yq = m_Op->GetDiscLine(m_nyPP, posH[m_nyPP], true) - y0_csv;
				double mP = 0, mPP = 0;
				H_parser.LinInterp2(xq, yq, mP, mPP);
				modeTmpH[0][posP][posPP] = mP;
				modeTmpH[1][posP][posPP] = mPP;
				double area = m_Op->GetNodeArea(m_ny, posH, true);
				H_norm_sq += (mP*mP + mPP*mPP) * area;
			}
		}
	}
	double H_norm = std::sqrt(H_norm_sq);
	cerr << "[ModalFDTD] " << H_file
	     << "  H_mode_norm=" << H_norm
	     << "  size=" << m_numLines_H[0] << "x" << m_numLines_H[1]
	     << "  H_in_line=" << m_H_in_line
	     << " H_override_line=" << m_H_override_line << endl;

	if (H_norm > 0)
	{
		unsigned int posH[3] = {0, 0, 0};
		posH[m_ny] = m_start[m_ny];
		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			posH[m_nyP] = m_start[m_nyP] + posP;
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				posH[m_nyPP] = m_start[m_nyPP] + posPP;
				double area      = m_Op->GetNodeArea(m_ny, posH, true);
				double edgeLen_P  = m_Op->GetEdgeLength(m_nyP,  posH, true);
				double edgeLen_PP = m_Op->GetEdgeLength(m_nyPP, posH, true);
				double mP  = modeTmpH[0][posP][posPP] / H_norm;
				double mPP = modeTmpH[1][posP][posPP] / H_norm;
				m_H_OverlapW[0][posP][posPP]  = (edgeLen_P  > 0) ? mP  * area / edgeLen_P  : 0;
				m_H_OverlapW[1][posP][posPP]  = (edgeLen_PP > 0) ? mPP * area / edgeLen_PP : 0;
				m_H_SubtractW[0][posP][posPP] = mP  * edgeLen_P;
				m_H_SubtractW[1][posP][posPP] = mPP * edgeLen_PP;
			}
		}
	}
	for (int n = 0; n < 2; ++n)
		Delete2DArray<double>(modeTmpH[n], m_numLines_H);

	return true;
}

Engine_Extension* Operator_Ext_ModalFDTD::CreateEngineExtention()
{
	return new Engine_Ext_ModalFDTD(this);
}

void Operator_Ext_ModalFDTD::ShowStat(std::ostream& ostr) const
{
	Operator_Extension::ShowStat(ostr);
	ostr << " Propagation dir: " << m_ny << " sign: " << m_dir << endl;
	ostr << " Sheet line:      " << m_srcLine << endl;
	ostr << " 1-D cells (N1D): " << m_N1D << endl;
	ostr << " kc:              " << m_kc << " rad/m" << endl;
	ostr << " dz, dt:          " << m_dz << " m, " << m_dt << " s" << endl;
}
