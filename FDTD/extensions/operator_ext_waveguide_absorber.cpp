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

#include "operator_ext_waveguide_absorber.h"
#include "engine_ext_waveguide_absorber.h"

#include "tools/array_ops.h"
#include "tools/constants.h"

#include "CSPrimBox.h"
#include "CSPropExcitation.h"
#include "CSFunctionParser.h"

#include <cmath>

using std::cerr;
using std::endl;

Operator_Ext_WaveguideAbsorber::Operator_Ext_WaveguideAbsorber(Operator* op, CSPropExcitation* exc_prop)
	: Operator_Extension(op)
{
	Initialize();
	m_exc_prop = exc_prop;
	m_absLayers = exc_prop->GetAbsorbLayers();
}

Operator_Ext_WaveguideAbsorber::~Operator_Ext_WaveguideAbsorber()
{
	for (int n = 0; n < 2; ++n)
	{
		Delete2DArray<double>(m_E_OverlapW[n], m_numLines_E);
		Delete2DArray<double>(m_E_SubtractW[n], m_numLines_E);
		Delete2DArray<double>(m_H_OverlapW[n], m_numLines_H);
		Delete2DArray<double>(m_H_SubtractW[n], m_numLines_H);
	}
}

Operator_Ext_WaveguideAbsorber::Operator_Ext_WaveguideAbsorber(Operator* op, Operator_Ext_WaveguideAbsorber* op_ext)
	: Operator_Extension(op, op_ext)
{
	Initialize();
	m_exc_prop = op_ext->m_exc_prop;
	m_absLayers = op_ext->m_absLayers;
}

Operator_Extension* Operator_Ext_WaveguideAbsorber::Clone(Operator* op)
{
	return new Operator_Ext_WaveguideAbsorber(op, this);
}

void Operator_Ext_WaveguideAbsorber::Initialize()
{
	m_exc_prop = NULL;
	m_ny = m_nyP = m_nyPP = -1;
	m_dir = 1;
	m_absLayers = 0;
	m_srcLine = 0;
	m_Zw = 376.73;

	for (int i = 0; i < 2; ++i)
	{
		m_E_abs_line[i] = 0;
		m_H_abs_line[i] = 0;
	}

	for (int i = 0; i < 3; ++i)
	{
		m_start[i] = 0;
		m_stop[i] = 0;
	}

	m_numLines_E[0] = m_numLines_E[1] = 0;
	m_numLines_H[0] = m_numLines_H[1] = 0;

	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n] = NULL;
		m_E_SubtractW[n] = NULL;
		m_H_OverlapW[n] = NULL;
		m_H_SubtractW[n] = NULL;
	}
}

bool Operator_Ext_WaveguideAbsorber::BuildExtension()
{
	if (!m_exc_prop)
	{
		SetActive(false);
		return false;
	}

	// 1. Determine propagation direction from PropagationDir vector
	m_ny = -1;
	int checkSum = 0;
	for (int i = 0; i < 3; ++i)
	{
		double pd = m_exc_prop->GetPropagationDir(i);
		if (pd != 0)
		{
			m_ny = i;
			m_dir = (pd > 0) ? 1 : -1;
			checkSum++;
		}
	}

	if (m_ny < 0 || checkSum != 1)
	{
		cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: Cannot determine propagation direction." << endl;
		SetActive(false);
		return false;
	}

	m_nyP  = (m_ny + 1) % 3;
	m_nyPP = (m_ny + 2) % 3;

	// 2. Find excitation plane from the primitive box
	std::vector<CSPrimitives*> prims = m_exc_prop->GetAllPrimitives();
	if (prims.empty())
	{
		cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: No primitives found." << endl;
		SetActive(false);
		return false;
	}

	CSPrimBox* box = dynamic_cast<CSPrimBox*>(prims[0]);
	if (!box)
	{
		cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: First primitive is not a box." << endl;
		SetActive(false);
		return false;
	}

	unsigned int boxStart[3], boxStop[3];
	m_Op->SnapBox2Mesh(
		box->GetStartCoord()->GetCoords(m_Op->m_MeshType),
		box->GetStopCoord()->GetCoords(m_Op->m_MeshType),
		boxStart, boxStop, false, true);

	// Source plane is where start==stop in the propagation direction
	m_srcLine = boxStart[m_ny];

	// Cross-section bounds from the box
	for (int i = 0; i < 3; ++i)
	{
		m_start[i] = boxStart[i];
		m_stop[i]  = boxStop[i];
	}

	// 3. Compute absorber plane positions
	// The absorber sits on the interior side of the source, intercepting
	// backward-traveling waves coming from the domain interior.
	// E at primary index k pairs with H at dual index k-dir (physically
	// at k - dir + 0.5), collocating them for the directional decomposition.
	if (m_absLayers == 1)
	{
		m_E_abs_line[0] = m_srcLine + m_dir;
		m_H_abs_line[0] = m_E_abs_line[0] - m_dir;
	}
	else if (m_absLayers == 2)
	{
		m_E_abs_line[0] = m_srcLine + 2 * m_dir;
		m_H_abs_line[0] = m_E_abs_line[0] - m_dir;
		m_E_abs_line[1] = m_srcLine + m_dir;
		m_H_abs_line[1] = m_E_abs_line[1] - m_dir;
	}

	// 4. Cross-section dimensions
	unsigned int Ncells[3];
	for (int i = 0; i < 3; ++i)
		Ncells[i] = m_stop[i] - m_start[i] + 1;

	m_numLines_E[0] = Ncells[m_nyP];
	m_numLines_E[1] = Ncells[m_nyPP];

	// H mode lives on dual mesh, so 1 fewer line
	m_numLines_H[0] = Ncells[m_nyP]  > 0 ? Ncells[m_nyP]  - 1 : 0;
	m_numLines_H[1] = Ncells[m_nyPP] > 0 ? Ncells[m_nyPP] - 1 : 0;

	// 5. Determine wave impedance.
	// Use the user-supplied value if available (from the mode solver or
	// analytical formula).  The Python port layer stores it in the
	// "WaveImpedance" attribute.  Otherwise, fall back to computing it from
	// FDTD grid coefficients.
	std::string zw_attr = m_exc_prop->GetAttributeValue("WaveImpedance");
	if (!zw_attr.empty())
	{
		m_Zw = std::stod(zw_attr);
		cerr << "  Waveguide absorber: using supplied Zw=" << m_Zw << " Ohm" << endl;
	}
	else
	{
		// Compute from grid coefficients.
		// Sample eps_r from the FDTD VI coefficient at non-PEC cells in the
		// cross-section.  For lossless dielectric: VI = dT * dL / (eps0*epsR*dA)
		// so epsR = dT * dL / (eps0 * VI * dA).
		double dT = m_Op->GetTimestep();
		unsigned int pos_tmp[3] = {m_start[0], m_start[1], m_start[2]};
		double dL = m_Op->GetEdgeLength(m_ny, pos_tmp, false);
		if (dL <= 0)
		{
			cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: Invalid edge length." << endl;
			SetActive(false);
			return false;
		}

		// Mode-weighted average of eps_r over the cross-section
		double epsR_eff = 1.0;
		{
			CSFunctionParser* tmpParser[2];
			for (int n = 0; n < 2; ++n)
				tmpParser[n] = new CSFunctionParser();

			std::string tmpE_func[3];
			for (int i = 0; i < 3; ++i)
				tmpE_func[i] = m_exc_prop->GetWeightFunction(i);

			bool parsedOK = true;
			for (int n = 0; n < 2; ++n)
			{
				int ny = (n == 0) ? m_nyP : m_nyPP;
				if (tmpParser[n]->Parse(tmpE_func[ny], "x,y,z,rho,a,r,t") >= 0)
					parsedOK = false;
			}

			double sum_wt = 0, sum_eps_wt = 0;
			unsigned int spos[3];
			spos[m_ny] = m_srcLine;

			for (unsigned int pP = 0; pP < Ncells[m_nyP]; ++pP)
			{
				spos[m_nyP] = m_start[m_nyP] + pP;
				for (unsigned int pPP = 0; pPP < Ncells[m_nyPP]; ++pPP)
				{
					spos[m_nyPP] = m_start[m_nyPP] + pPP;

					double vi_val = m_Op->GetVI(m_nyP, spos[0], spos[1], spos[2]);
					if (vi_val <= 0) continue;

					double eL = m_Op->GetEdgeLength(m_nyP, spos, false);
					double eA = m_Op->GetEdgeArea(m_nyP, spos, false);
					if (eL <= 0 || eA <= 0) continue;

					double local_eps = dT * eL / (__EPS0__ * vi_val * eA);

					double mode_mag2 = 1.0;
					if (parsedOK)
					{
						double discLine[3];
						for (int i = 0; i < 3; ++i)
							discLine[i] = m_Op->GetDiscLine(i, spos[i], false);
						double var[7];
						var[0] = discLine[0];
						var[1] = discLine[1];
						var[2] = discLine[2];
						var[3] = sqrt(var[0]*var[0] + var[1]*var[1]);
						var[4] = atan2(var[1], var[0]);
						var[5] = sqrt(var[0]*var[0] + var[1]*var[1] + var[2]*var[2]);
						var[6] = (var[3] > 0) ? asin(1) - atan(var[2]/var[3]) : 0;
						double m0 = tmpParser[0]->Eval(var);
						double m1 = tmpParser[1]->Eval(var);
						mode_mag2 = m0*m0 + m1*m1;
					}

					sum_eps_wt += local_eps * mode_mag2;
					sum_wt    += mode_mag2;
				}
			}

			if (sum_wt > 0)
				epsR_eff = sum_eps_wt / sum_wt;
			else
				cerr << "  Waveguide absorber: Warning - zero total mode weight for eps_r sampling" << endl;

			for (int n = 0; n < 2; ++n)
				delete tmpParser[n];
		}

		double S = __C0__ * dT / (dL * sqrt(epsR_eff));
		m_Zw = sqrt(__MUE0__ / __EPS0__) / sqrt(epsR_eff) * sqrt(1.0 - S * S / 4.0);
		cerr << "  Waveguide absorber: computed Zw=" << m_Zw << " Ohm (epsR_eff=" << epsR_eff << ")" << endl;
	}

	// 6. Parse E-field mode weight functions and build E mode distribution
	CSFunctionParser* E_parser[2];
	for (int n = 0; n < 2; ++n)
		E_parser[n] = new CSFunctionParser();

	std::string E_func[3];
	for (int i = 0; i < 3; ++i)
		E_func[i] = m_exc_prop->GetWeightFunction(i);

	// Parse transverse E components
	for (int n = 0; n < 2; ++n)
	{
		int ny = (n == 0) ? m_nyP : m_nyPP;
		int res = E_parser[n]->Parse(E_func[ny], "x,y,z,rho,a,r,t");
		if (res >= 0)
		{
			cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: Error parsing E weight function for component " << ny << endl;
			cerr << E_func[ny] << "\n" << std::string(res, ' ') << "^\n" << E_parser[n]->ErrorMsg() << endl;
			for (int k = 0; k < 2; ++k) delete E_parser[k];
			SetActive(false);
			return false;
		}
	}

	// Allocate E mode arrays
	for (int n = 0; n < 2; ++n)
	{
		m_E_OverlapW[n]  = Create2DArray<double>(m_numLines_E);
		m_E_SubtractW[n] = Create2DArray<double>(m_numLines_E);
	}

	// Build E mode distribution on primary mesh.
	// The E norm is also used for H normalization to keep them consistent.
	double E_mode_norm = 0;
	{
		double** modeTmp[2];
		for (int n = 0; n < 2; ++n)
			modeTmp[n] = Create2DArray<double>(m_numLines_E);

		unsigned int pos[3] = {0, 0, 0};
		double discLine[3] = {0, 0, 0};
		pos[m_ny] = m_start[m_ny];
		discLine[m_ny] = m_Op->GetDiscLine(m_ny, pos[m_ny], false);
		double norm = 0;

		for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
		{
			pos[m_nyP] = m_start[m_nyP] + posP;
			discLine[m_nyP] = m_Op->GetDiscLine(m_nyP, pos[m_nyP], false);
			for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
			{
				pos[m_nyPP] = m_start[m_nyPP] + posPP;
				discLine[m_nyPP] = m_Op->GetDiscLine(m_nyPP, pos[m_nyPP], false);

				double var[7];
				var[0] = discLine[0];
				var[1] = discLine[1];
				var[2] = discLine[2];
				var[3] = sqrt(discLine[0]*discLine[0] + discLine[1]*discLine[1]);
				var[4] = atan2(discLine[1], discLine[0]);
				var[5] = sqrt(var[0]*var[0] + var[1]*var[1] + var[2]*var[2]);
				var[6] = (var[3] > 0) ? asin(1) - atan(var[2]/var[3]) : 0;

				for (int n = 0; n < 2; ++n)
					modeTmp[n][posP][posPP] = E_parser[n]->Eval(var);

				double area = m_Op->GetNodeArea(m_ny, pos, false);
				for (int n = 0; n < 2; ++n)
					norm += modeTmp[n][posP][posPP] * modeTmp[n][posP][posPP] * area;
			}
		}

		norm = sqrt(norm);
		E_mode_norm = norm;
		if (norm > 0)
		{
			for (unsigned int posP = 0; posP < m_numLines_E[0]; ++posP)
			{
				pos[m_nyP] = m_start[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_E[1]; ++posPP)
				{
					pos[m_nyPP] = m_start[m_nyPP] + posPP;

					double area = m_Op->GetNodeArea(m_ny, pos, false);
					double edgeLen_P  = m_Op->GetEdgeLength(m_nyP,  pos, false);
					double edgeLen_PP = m_Op->GetEdgeLength(m_nyPP, pos, false);

					double mP  = modeTmp[0][posP][posPP] / norm;
					double mPP = modeTmp[1][posP][posPP] / norm;

					m_E_OverlapW[0][posP][posPP] = (edgeLen_P  > 0) ? mP  * area / edgeLen_P  : 0;
					m_E_OverlapW[1][posP][posPP] = (edgeLen_PP > 0) ? mPP * area / edgeLen_PP : 0;

					m_E_SubtractW[0][posP][posPP] = mP  * edgeLen_P;
					m_E_SubtractW[1][posP][posPP] = mPP * edgeLen_PP;
				}
			}
		}

		for (int n = 0; n < 2; ++n)
			Delete2DArray<double>(modeTmp[n], m_numLines_E);
	}

	for (int n = 0; n < 2; ++n)
		delete E_parser[n];

	// 7. Parse H-field mode weight functions and build H mode distribution
	CSFunctionParser* H_parser[2];
	for (int n = 0; n < 2; ++n)
		H_parser[n] = new CSFunctionParser();

	std::string H_func[3];
	for (int i = 0; i < 3; ++i)
		H_func[i] = m_exc_prop->GetH_WeightFunction(i);

	for (int n = 0; n < 2; ++n)
	{
		int ny = (n == 0) ? m_nyP : m_nyPP;
		int res = H_parser[n]->Parse(H_func[ny], "x,y,z,rho,a,r,t");
		if (res >= 0)
		{
			cerr << "Operator_Ext_WaveguideAbsorber::BuildExtension: Error parsing H weight function for component " << ny << endl;
			cerr << H_func[ny] << "\n" << std::string(res, ' ') << "^\n" << H_parser[n]->ErrorMsg() << endl;
			for (int k = 0; k < 2; ++k) delete H_parser[k];
			SetActive(false);
			return false;
		}
	}

	// Allocate H mode arrays
	for (int n = 0; n < 2; ++n)
	{
		m_H_OverlapW[n]  = Create2DArray<double>(m_numLines_H);
		m_H_SubtractW[n] = Create2DArray<double>(m_numLines_H);
	}

	// Build H mode distribution on dual mesh.
	// Use E_mode_norm for normalization to keep E/H mode overlap integrals
	// consistent.  On the Yee grid, the E-mode on the primary mesh and the
	// H-mode on the dual mesh have different L2 norms even for the same
	// analytic function.  Using the E norm for both ensures that
	// a_bwd = 0.5*(Emm - Zw*Hmm) = 0 for a pure forward wave.
	{
		double** modeTmp[2];
		for (int n = 0; n < 2; ++n)
			modeTmp[n] = Create2DArray<double>(m_numLines_H);

		unsigned int pos[3] = {0, 0, 0};
		double discLine[3] = {0, 0, 0};
		pos[m_ny] = m_start[m_ny];
		discLine[m_ny] = m_Op->GetDiscLine(m_ny, pos[m_ny], true);

		for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
		{
			pos[m_nyP] = m_start[m_nyP] + posP;
			discLine[m_nyP] = m_Op->GetDiscLine(m_nyP, pos[m_nyP], true);
			for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
			{
				pos[m_nyPP] = m_start[m_nyPP] + posPP;
				discLine[m_nyPP] = m_Op->GetDiscLine(m_nyPP, pos[m_nyPP], true);

				double var[7];
				var[0] = discLine[0];
				var[1] = discLine[1];
				var[2] = discLine[2];
				var[3] = sqrt(discLine[0]*discLine[0] + discLine[1]*discLine[1]);
				var[4] = atan2(discLine[1], discLine[0]);
				var[5] = sqrt(var[0]*var[0] + var[1]*var[1] + var[2]*var[2]);
				var[6] = (var[3] > 0) ? asin(1) - atan(var[2]/var[3]) : 0;

				for (int n = 0; n < 2; ++n)
					modeTmp[n][posP][posPP] = H_parser[n]->Eval(var);
			}
		}

		double norm = E_mode_norm;
		if (norm > 0)
		{
			for (unsigned int posP = 0; posP < m_numLines_H[0]; ++posP)
			{
				pos[m_nyP] = m_start[m_nyP] + posP;
				for (unsigned int posPP = 0; posPP < m_numLines_H[1]; ++posPP)
				{
					pos[m_nyPP] = m_start[m_nyPP] + posPP;

					double area = m_Op->GetNodeArea(m_ny, pos, true);
					double edgeLen_P  = m_Op->GetEdgeLength(m_nyP,  pos, true);
					double edgeLen_PP = m_Op->GetEdgeLength(m_nyPP, pos, true);

					double mP  = modeTmp[0][posP][posPP] / norm;
					double mPP = modeTmp[1][posP][posPP] / norm;

					m_H_OverlapW[0][posP][posPP] = (edgeLen_P  > 0) ? mP  * area / edgeLen_P  : 0;
					m_H_OverlapW[1][posP][posPP] = (edgeLen_PP > 0) ? mPP * area / edgeLen_PP : 0;

					m_H_SubtractW[0][posP][posPP] = mP  * edgeLen_P;
					m_H_SubtractW[1][posP][posPP] = mPP * edgeLen_PP;
				}
			}
		}

		for (int n = 0; n < 2; ++n)
			Delete2DArray<double>(modeTmp[n], m_numLines_H);
	}

	for (int n = 0; n < 2; ++n)
		delete H_parser[n];

	return true;
}

Engine_Extension* Operator_Ext_WaveguideAbsorber::CreateEngineExtention()
{
	Engine_Ext_WaveguideAbsorber* eng_ext = new Engine_Ext_WaveguideAbsorber(this);
	return eng_ext;
}

void Operator_Ext_WaveguideAbsorber::ShowStat(std::ostream &ostr) const
{
	Operator_Extension::ShowStat(ostr);
	ostr << " Propagation dir: " << m_ny << " sign: " << m_dir << endl;
	ostr << " Absorber layers: " << m_absLayers << endl;
	ostr << " Source line: " << m_srcLine << endl;
	ostr << " E-mode lines: " << m_numLines_E[0] << " x " << m_numLines_E[1] << endl;
	ostr << " H-mode lines: " << m_numLines_H[0] << " x " << m_numLines_H[1] << endl;
	ostr << " Wave impedance: " << m_Zw << " Ohm" << endl;
	for (int l = 0; l < m_absLayers; ++l)
		ostr << " Layer " << l << ": E_abs=" << m_E_abs_line[l] << " H_abs=" << m_H_abs_line[l] << endl;
}
