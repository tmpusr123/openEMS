/*
*	Copyright (C) 2012-2025 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_lossymetal.h"
#include "tools/array_ops.h"
#include "tools/constants.h"
#include "lossy_metal_parameter.h"

#include "CSPropLossyMetal.h"

#include <cmath>

using std::cerr;
using std::cout;
using std::endl;

Operator_Ext_LossyMetal::Operator_Ext_LossyMetal(Operator* op, double f_max) : Operator_Ext_LorentzMaterial(op)
{
	m_f_max = f_max;
}

Operator_Ext_LossyMetal::Operator_Ext_LossyMetal(Operator* op, Operator_Ext_LossyMetal* op_ext) : Operator_Ext_LorentzMaterial(op, op_ext)
{
	m_f_max = op_ext->m_f_max;
}

Operator_Extension* Operator_Ext_LossyMetal::Clone(Operator* op)
{
	return new Operator_Ext_LossyMetal(op, this);
}

int Operator_Ext_LossyMetal::GetSurfaceInfo(int n, unsigned int pos[3], CSPrimitives* prim,
                                             double &surface_width)
{
	int nP = (n+1)%3;
	int nPP = (n+2)%3;

	double coord[3];
	double shift_coord[3];

	if (m_Op->GetYeeCoords(n, pos, coord, false)==false)
	{
		surface_width = 0;
		return 0;
	}

	double half_width;
	int num_outside = 0;
	surface_width = 0;

	// Check nP direction
	half_width = 0.5 * m_Op->GetEdgeLength(nP, pos, false);

	shift_coord[0] = coord[0]; shift_coord[1] = coord[1]; shift_coord[2] = coord[2];
	shift_coord[nP] += half_width;
	bool nP_plus_outside = !prim->IsInside(shift_coord);

	shift_coord[0] = coord[0]; shift_coord[1] = coord[1]; shift_coord[2] = coord[2];
	shift_coord[nP] -= half_width;
	bool nP_minus_outside = !prim->IsInside(shift_coord);

	// Check nPP direction
	half_width = 0.5 * m_Op->GetEdgeLength(nPP, pos, false);

	shift_coord[0] = coord[0]; shift_coord[1] = coord[1]; shift_coord[2] = coord[2];
	shift_coord[nPP] += half_width;
	bool nPP_plus_outside = !prim->IsInside(shift_coord);

	shift_coord[0] = coord[0]; shift_coord[1] = coord[1]; shift_coord[2] = coord[2];
	shift_coord[nPP] -= half_width;
	bool nPP_minus_outside = !prim->IsInside(shift_coord);

	if (nP_plus_outside || nP_minus_outside)
	{
		num_outside++;
		surface_width += m_Op->GetEdgeLength(nPP, pos, false);
	}
	if (nPP_plus_outside || nPP_minus_outside)
	{
		num_outside++;
		surface_width += m_Op->GetEdgeLength(nP, pos, false);
	}

	return num_outside;
}

bool Operator_Ext_LossyMetal::BuildExtension()
{
	double dT = m_Op->GetTimestep();
	unsigned int pos[] = {0,0,0};
	double coord[3];
	unsigned int numLines[3] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};

	double omega_max = 2.0 * PI * m_f_max;

	// Phase 1: collect surface edge positions and properties
	unsigned int numSurfaceEdges = 0;
	unsigned int numInteriorPEC = 0;

	std::vector<unsigned int> v_pos[3];
	std::vector<double> v_sigma;
	std::vector<double> v_surface_width;
	std::vector<double> v_dl;
	std::vector<int> v_dir;

	CSPrimitives* found_prim = NULL;
	bool disable_pos;

	for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
	{
		for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
		{
			std::vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(
				pos[0], pos[1], -1,
				(CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL)
			);

			for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
			{
				disable_pos = false;

				for (int m=0; m<3; ++m)
					if ((pos[m]<=(unsigned int)m_Op->GetBCSize(2*m)) || (pos[m]>=(numLines[m]-m_Op->GetBCSize(2*m+1)-1)))
						disable_pos = true;

				for (int n=0; n<3; ++n)
				{
					if (m_Op->GetYeeCoords(n,pos,coord,false)==false)
						continue;

					if (m_CC_R0_included && (n==2) && (pos[0]==0))
						disable_pos = true;

					CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, false, &found_prim);
					CSPropLossyMetal* lm_prop = dynamic_cast<CSPropLossyMetal*>(prop);
					if (lm_prop)
					{
						if (found_prim==NULL)
							return false;

						found_prim->SetPrimitiveUsed(true);

						double sigma = lm_prop->GetConductivity();

						double surface_width = 0;
						int num_outside = GetSurfaceInfo(n, pos, found_prim, surface_width);

						if (num_outside == 0 || disable_pos || sigma <= 0)
						{
							// Interior edge: set to PEC
							m_Op->SetVV(n,pos[0],pos[1],pos[2], 0);
							m_Op->SetVI(n,pos[0],pos[1],pos[2], 0);
							++m_Op->m_Nr_PEC[n];
							++numInteriorPEC;
							continue;
						}

						double dl = m_Op->GetEdgeLength(n, pos);

						if (dl <= 0 || surface_width <= 0)
						{
							m_Op->SetVV(n,pos[0],pos[1],pos[2], 0);
							m_Op->SetVI(n,pos[0],pos[1],pos[2], 0);
							++m_Op->m_Nr_PEC[n];
							++numInteriorPEC;
							continue;
						}

						// Store surface edge info for ADE setup
						v_pos[0].push_back(pos[0]);
						v_pos[1].push_back(pos[1]);
						v_pos[2].push_back(pos[2]);
						v_sigma.push_back(sigma);
						v_surface_width.push_back(surface_width);
						v_dl.push_back(dl);
						v_dir.push_back(n);
						++numSurfaceEdges;
					}
				}
			}
		}
	}

	if (numSurfaceEdges == 0 && numInteriorPEC == 0)
		return false;

	cout << "Operator_Ext_LossyMetal::BuildExtension: " << numInteriorPEC << " interior PEC edges, "
	     << numSurfaceEdges << " surface SIBC edges." << endl;

	if (numSurfaceEdges == 0)
	{
		m_Order = 0;
		return true;
	}

	// Phase 2: Set up ADE with 2 parallel RL branches
	//
	// Surface admittance per square:
	//   Y_s(w) = G0 + 1/(R1+jw*L1) + 1/(R2+jw*L2)
	//
	// Physical scaling from lossy_metal_parameter.h:
	//   delta = sqrt(mu0/sigma)
	//   G0_phys = G0_hat / (delta * sqrt(omega_max))   [S/sq]
	//   Rk_phys = Rk_hat * delta * sqrt(omega_max)     [Ohm/sq]
	//   Lk_phys = Lk_hat * delta / sqrt(omega_max)     [H/sq]
	//
	// Cell-level (geometric scaling):
	//   G0_cell = G0_sq * surface_width / dl            [S]
	//   Rk_cell = Rk_sq * dl / surface_width            [Ohm]
	//   Lk_cell = Lk_sq * dl / surface_width            [H]
	//
	// Operator setup (stability correction from conducting sheet):
	//   EC_G += G0_cell
	//   EC_C += dT^2/4 * (16/Lmin + 1/L1 + 1/L2)
	//   -> VV, VI via Calc_ECOperatorPos
	//
	// ADE coefficients for each RL branch:
	//   v_int = (2*L - dT*R) / (2*L + dT*R)
	//   v_ext = dT / (L + dT*R/2) * VI

	m_Order = 2;
	size_t numCS = numSurfaceEdges;

	m_LM_Count.push_back(numCS);
	m_LM_Count.push_back(numCS);

	m_volt_ADE_On = new bool[m_Order];
	m_volt_ADE_On[0] = m_volt_ADE_On[1] = true;
	m_curr_ADE_On = new bool[m_Order];
	m_curr_ADE_On[0] = m_curr_ADE_On[1] = false;

	m_volt_Lor_ADE_On = new bool[m_Order];
	m_volt_Lor_ADE_On[0] = m_volt_Lor_ADE_On[1] = false;
	m_curr_Lor_ADE_On = new bool[m_Order];
	m_curr_Lor_ADE_On[0] = m_curr_Lor_ADE_On[1] = false;

	m_LM_pos = new unsigned int**[m_Order];
	m_LM_pos[0] = new unsigned int*[3];
	m_LM_pos[1] = new unsigned int*[3];

	v_int_ADE = new FDTD_FLOAT**[m_Order];
	v_ext_ADE = new FDTD_FLOAT**[m_Order];

	v_int_ADE[0] = new FDTD_FLOAT*[3];
	v_ext_ADE[0] = new FDTD_FLOAT*[3];
	v_int_ADE[1] = new FDTD_FLOAT*[3];
	v_ext_ADE[1] = new FDTD_FLOAT*[3];

	// Allocate unused arrays to keep parent destructor happy
	i_int_ADE = NULL;
	i_ext_ADE = NULL;
	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	for (int nn=0; nn<3; ++nn)
	{
		m_LM_pos[0][nn] = new unsigned int[numCS];
		m_LM_pos[1][nn] = new unsigned int[numCS];

		v_int_ADE[0][nn] = new FDTD_FLOAT[numCS];
		v_int_ADE[1][nn] = new FDTD_FLOAT[numCS];
		v_ext_ADE[0][nn] = new FDTD_FLOAT[numCS];
		v_ext_ADE[1][nn] = new FDTD_FLOAT[numCS];
	}

	// Normalized RL parameters (from lossy_metal_parameter.h)
	double G0_hat = SIBC_G0_HAT;
	double R_hat[2] = {SIBC_R1_HAT, SIBC_R2_HAT};
	double L_hat[2] = {SIBC_L1_HAT, SIBC_L2_HAT};

	for (unsigned int i=0; i<numCS; ++i)
	{
		pos[0] = v_pos[0][i];
		pos[1] = v_pos[1][i];
		pos[2] = v_pos[2][i];
		int n = v_dir[i];
		double sigma = v_sigma[i];
		double sw = v_surface_width[i];
		double dl = v_dl[i];

		// Physical scaling
		double delta = sqrt(__MUE0__ / sigma);
		double sqrt_wmax = sqrt(omega_max);

		// Physical parameters per square
		double G0_sq = G0_hat / (delta * sqrt_wmax);
		double R_sq[2], L_sq[2];
		for (int k=0; k<2; ++k)
		{
			R_sq[k] = R_hat[k] * delta * sqrt_wmax;
			L_sq[k] = L_hat[k] * delta / sqrt_wmax;
		}

		// Cell-level geometric scaling
		double wtl = sw / dl;   // width-to-length (for admittance)
		double ltw = dl / sw;   // length-to-width (for impedance)

		double G0_cell = G0_sq * wtl;
		double R_cell[2], L_cell[2];
		for (int k=0; k<2; ++k)
		{
			R_cell[k] = R_sq[k] * ltw;
			L_cell[k] = L_sq[k] * ltw;
		}

		unsigned int index = m_Op->MainOp->SetPos(pos[0],pos[1],pos[2]);

		// Stability correction for EC_C and set EC_G
		double Lmin = (L_cell[0] < L_cell[1]) ? L_cell[0] : L_cell[1];
		m_Op->EC_C[n][index] += dT*dT/4.0 * (16.0/Lmin + 1.0/L_cell[0] + 1.0/L_cell[1]);
		m_Op->EC_G[n][index] += G0_cell;
		m_Op->Calc_ECOperatorPos(n, pos);

		double VI = m_Op->GetVI(n, pos[0], pos[1], pos[2]);

		// Store positions (same for both ADE orders)
		m_LM_pos[0][0][i] = pos[0]; m_LM_pos[0][1][i] = pos[1]; m_LM_pos[0][2][i] = pos[2];
		m_LM_pos[1][0][i] = pos[0]; m_LM_pos[1][1][i] = pos[1]; m_LM_pos[1][2][i] = pos[2];

		// Initialize all directions to zero (inactive: v_int=0, v_ext=0)
		for (int nn=0; nn<3; ++nn)
		{
			v_int_ADE[0][nn][i] = 0; v_ext_ADE[0][nn][i] = 0;
			v_int_ADE[1][nn][i] = 0; v_ext_ADE[1][nn][i] = 0;
		}

		// ADE coefficients for RL branch 1: 1/(R1 + jw*L1)
		v_int_ADE[0][n][i] = (2.0*L_cell[0] - dT*R_cell[0]) / (2.0*L_cell[0] + dT*R_cell[0]);
		v_ext_ADE[0][n][i] = dT / (L_cell[0] + dT*R_cell[0]/2.0) * VI;

		// ADE coefficients for RL branch 2: 1/(R2 + jw*L2)
		v_int_ADE[1][n][i] = (2.0*L_cell[1] - dT*R_cell[1]) / (2.0*L_cell[1] + dT*R_cell[1]);
		v_ext_ADE[1][n][i] = dT / (L_cell[1] + dT*R_cell[1]/2.0) * VI;
	}

	cout << "  Frequency-dependent SIBC (N=2 RL ADE)" << endl;
	cout << "  f_max = " << m_f_max << " Hz, omega_max = " << omega_max << " rad/s" << endl;

	return true;
}
