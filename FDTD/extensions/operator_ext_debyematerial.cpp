/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_debyematerial.h"
#include "engine_ext_debyematerial.h"
#include "operator_ext_cylinder.h"
#include "../operator_cylinder.h"

#include "tools/constants.h"
#include "CSPropDebyeMaterial.h"

#include <vector>
#include <string>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::cerr;
using std::endl;

//! A Debye pole needs its relaxation time resolved by the timestep. The
//! trapezoidal update stays stable up to dT/tau of about 0.35, independent of
//! d_eps, so refuse a pole below three timesteps per relaxation time.
// Shortest relaxation time a pole may have, in timesteps; a faster pole is
// skipped with a warning. Upstream chose 3 as an accuracy margin (the update is
// stable for any tau). Measured on the CUDA engine against the analytic slab of
// the SPECIFIED material (one pole, d_eps = 1 on eps_inf = 3, 1-9 GHz):
// simulating the pole beats every alternative down to tau ~ 1 dT (max |T err|
// 0.038 vs 0.078 for folding it into eps_inf at 2 dT, 0.036 vs 0.113 at 3 dT),
// and SKIPPING it -- what a cutoff does -- is worst at every tau (~0.5). Below
// ~1 dT folding wins (0.004 at 0.1 dT). So the cutoff sits at 1: poles that can
// be simulated usefully are; Blit folds the faster ones before they get here.
// The environment variable OPENEMS_DEBYE_MIN_TAU_PER_DT overrides it.
#define DEBYE_MIN_TAU_PER_DT 1.0

Operator_Ext_DebyeMaterial::Operator_Ext_DebyeMaterial(Operator* op) : Operator_Ext_Dispersive(op)
{
	m_PoleCount = 0;
	v_relax_ADE = NULL;
	v_drive_ADE = NULL;
	v_solve_ADE = NULL;
}

Operator_Ext_DebyeMaterial::Operator_Ext_DebyeMaterial(Operator* op, Operator_Ext_DebyeMaterial* op_ext) : Operator_Ext_Dispersive(op,op_ext)
{
	m_PoleCount = 0;
	v_relax_ADE = NULL;
	v_drive_ADE = NULL;
	v_solve_ADE = NULL;
}

void Operator_Ext_DebyeMaterial::DeleteArrays()
{
	for (int o=0; o<m_PoleCount; ++o)
	{
		if (v_relax_ADE!=NULL)
		{
			for (int n=0; n<3; ++n)
			{
				delete[] v_relax_ADE[o][n];
				delete[] v_drive_ADE[o][n];
			}
			delete[] v_relax_ADE[o];
			delete[] v_drive_ADE[o];
		}
	}
	delete[] v_relax_ADE;
	delete[] v_drive_ADE;
	v_relax_ADE = NULL;
	v_drive_ADE = NULL;

	if (v_solve_ADE!=NULL)
		for (int n=0; n<3; ++n)
			delete[] v_solve_ADE[n];
	delete[] v_solve_ADE;
	v_solve_ADE = NULL;
}

Operator_Ext_DebyeMaterial::~Operator_Ext_DebyeMaterial()
{
	DeleteArrays();
	m_PoleCount = 0;
}

Operator_Extension* Operator_Ext_DebyeMaterial::Clone(Operator* op)
{
	return new Operator_Ext_DebyeMaterial(op, this);
}

bool Operator_Ext_DebyeMaterial::BuildExtension()
{
	double dT = m_Op->GetTimestep();
	unsigned int numLines[3] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};

	m_PoleCount = 0;
	std::vector<CSProperties*> props = m_Op->GetGeometryCSX()->GetPropertyByType(CSProperties::DEBYEMATERIAL);
	for (size_t n=0; n<props.size(); ++n)
	{
		CSPropDebyeMaterial* mat = dynamic_cast<CSPropDebyeMaterial*>(props.at(n));
		if (mat==NULL)
			return false; //sanity check, this should not happen
		if (mat->GetDispersionOrder()>m_PoleCount)
			m_PoleCount = mat->GetDispersionOrder();
	}
	if (m_PoleCount<1)
		return false; //no Debye material, drop this extension

	// One entry per active cell, m_PoleCount coefficient sets per direction.
	//
	// Per-cell material detection over the WHOLE volume (point queries per cell per
	// component) is the slow part of setup on a large model -- the same stage the
	// fork parallelized in Operator_Ext_LorentzMaterial (~3.5 h single-threaded on a
	// 1G-cell Debye3 model, which used to run through that extension). Done the
	// same way: parallel over x-planes, each plane collecting into its own
	// buffers, concatenated in x order afterwards. That reproduces the serial
	// push_back order EXACTLY (the engine indexes these arrays linearly), so the
	// result is bit-identical to a serial build. The calls made here are the
	// ones the parallel Lorentz build already makes per cell.
	double min_tau_per_dt = DEBYE_MIN_TAU_PER_DT;
	if (const char* env = getenv("OPENEMS_DEBYE_MIN_TAU_PER_DT"))
		if (atof(env) > 0)
			min_tau_per_dt = atof(env);
	const int P = m_PoleCount;
	const int _nx = (int)numLines[0];
	std::vector< std::vector<unsigned int> > pl_pos(_nx);          // 3 per cell
	std::vector< std::vector<double> > pl_solve(_nx);              // 3 per cell
	std::vector< std::vector<double> > pl_relax(_nx), pl_drive(_nx); // 3*P per cell
	// the first too-fast pole in x order, for the warning
	std::vector<double> pl_warn_tau(_nx, 0.0);
	std::vector<std::string> pl_warn_name(_nx);

#ifdef _OPENMP
	#pragma omp parallel for schedule(dynamic,1)
#endif
	for (int _x=0; _x<_nx; ++_x)
	{
		unsigned int pos[3]; pos[0] = (unsigned int)_x;
		double coord[3];
		double S[3];
		std::vector<double> relax(3*P), drive(3*P);
		for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
		{
			std::vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(
				pos[0], pos[1], -1,
				(CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL)
			);

			for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
			{
				bool b_pos_on = false;

				for (int n=0; n<3; ++n)
				{
					S[n] = 0.0;
					for (int o=0; o<P; ++o)
					{
						relax[n*P+o] = 0.0;
						drive[n*P+o] = 0.0;
					}

					if (m_Op->GetYeeCoords(n,pos,coord,false)==false)
						continue;
					if (m_CC_R0_included && (n==2) && (pos[0]==0))
						coord[1] = m_Op->GetDiscLine(1,0);

					// the cell's own voltage update coefficient: it carries the cell
					// losses and, at r==0 in cylindrical coords, the special case
					double vi;
					if (m_CC_R0_included && (n==2) && (pos[0]==0))
						vi = m_Op_Cyl->m_Cyl_Ext->vi_R0[pos[2]];
					else
						vi = m_Op->GetVI(n,pos[0],pos[1],pos[2]);
					if (vi==0)
						continue;

					CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, true);
					if (prop==NULL)
						continue;
					CSPropDebyeMaterial* mat = prop->ToDebyeMaterial();
					if (mat==NULL)
						continue;

					double A_l = m_Op->GetEdgeArea(n,pos)/m_Op->GetEdgeLength(n,pos);
					int order = mat->GetDispersionOrder();
					if (order>P)
						order = P;

					for (int o=0; o<order; ++o)
					{
						double d_eps   = mat->GetEpsDeltaWeighted(o,n,coord);
						double t_relax = mat->GetEpsRelaxTimeWeighted(o,n,coord);
						if ((d_eps<=0) || (t_relax<=0))
							continue;
						if (t_relax < min_tau_per_dt*dT)
						{
							if (pl_warn_tau[_x]==0.0)
							{
								pl_warn_tau[_x] = t_relax;
								pl_warn_name[_x] = mat->GetName();
							}
							continue;
						}
						// the same eps0 literal the old Lorentz-path Debye code used
						double C_L = 8.85418781762e-12 * d_eps * A_l;
						relax[n*P+o] = (2.0*t_relax - dT)/(2.0*t_relax + dT);
						// drive = r*c2, with r = C_L*vi/dT the branch to cell
						// capacitance ratio and c2 = dT/(2 tau + dT)
						drive[n*P+o] = C_L*vi/(2.0*t_relax + dT);
						S[n] += drive[n*P+o];
						b_pos_on = true;
					}
				}

				if (b_pos_on==false)
					continue;

				for (int n=0; n<3; ++n)
				{
					pl_pos[_x].push_back(pos[n]);
					pl_solve[_x].push_back(1.0/(1.0 + S[n]));
					for (int o=0; o<P; ++o)
					{
						pl_relax[_x].push_back(relax[n*P+o]);
						pl_drive[_x].push_back(drive[n*P+o]);
					}
				}
			}
		}
	}

	for (int _x=0; _x<_nx; ++_x)
	{
		if (pl_warn_tau[_x]>0.0)
		{
			cerr << "Operator_Ext_DebyeMaterial::BuildExtension(): Warning, "
			     << "relaxation time (" << pl_warn_tau[_x] << "s) of material \""
			     << pl_warn_name[_x] << "\" needs at least "
			     << min_tau_per_dt << " timesteps (dT=" << dT
			     << "s), skipping this pole..." << endl;
			break;
		}
	}

	// concatenate the planes in x order: exactly the serial push_back order
	std::vector<unsigned int> v_pos[3];
	std::vector<double> v_solve[3];
	std::vector<std::vector<double> > v_relax[3], v_drive[3];
	for (int n=0; n<3; ++n)
	{
		v_relax[n].resize(P);
		v_drive[n].resize(P);
	}
	for (int _x=0; _x<_nx; ++_x)
	{
		size_t cells = pl_pos[_x].size()/3;
		for (size_t c=0; c<cells; ++c)
			for (int n=0; n<3; ++n)
			{
				v_pos[n].push_back(pl_pos[_x][c*3+n]);
				v_solve[n].push_back(pl_solve[_x][c*3+n]);
				for (int o=0; o<P; ++o)
				{
					v_relax[n][o].push_back(pl_relax[_x][(c*3+n)*P+o]);
					v_drive[n][o].push_back(pl_drive[_x][(c*3+n)*P+o]);
				}
			}
		std::vector<unsigned int>().swap(pl_pos[_x]);
		std::vector<double>().swap(pl_solve[_x]);
		std::vector<double>().swap(pl_relax[_x]);
		std::vector<double>().swap(pl_drive[_x]);
	}

	unsigned int count = v_pos[0].size();
	if (count==0)
		return false; //no active cell, drop this extension

	// The base applies one correction per slot; the implicit solve couples all
	// poles into a single joint correction, so there is exactly one slot and the
	// cell list is shared, not replicated per pole.
	m_Order = 1;
	m_LM_Count.push_back(count);
	m_volt_ADE_On = new bool[1];
	m_curr_ADE_On = new bool[1];
	m_volt_ADE_On[0] = true;
	m_curr_ADE_On[0] = false;

	m_LM_pos = new unsigned int**[1];
	m_LM_pos[0] = new unsigned int*[3];

	v_solve_ADE = new FDTD_FLOAT*[3];
	v_relax_ADE  = new FDTD_FLOAT**[m_PoleCount];
	v_drive_ADE   = new FDTD_FLOAT**[m_PoleCount];
	for (int o=0; o<m_PoleCount; ++o)
	{
		v_relax_ADE[o] = new FDTD_FLOAT*[3];
		v_drive_ADE[o]  = new FDTD_FLOAT*[3];
	}

	for (int n=0; n<3; ++n)
	{
		m_LM_pos[0][n] = new unsigned int[count];
		v_solve_ADE[n]       = new FDTD_FLOAT[count];
		for (unsigned int i=0; i<count; ++i)
		{
			m_LM_pos[0][n][i] = v_pos[n].at(i);
			v_solve_ADE[n][i]       = v_solve[n].at(i);
		}
		for (int o=0; o<m_PoleCount; ++o)
		{
			v_relax_ADE[o][n] = new FDTD_FLOAT[count];
			v_drive_ADE[o][n]  = new FDTD_FLOAT[count];
			for (unsigned int i=0; i<count; ++i)
			{
				v_relax_ADE[o][n][i] = v_relax[n][o].at(i);
				v_drive_ADE[o][n][i]  = v_drive[n][o].at(i);
			}
		}
	}

	return true;
}

Engine_Extension* Operator_Ext_DebyeMaterial::CreateEngineExtention(Engine *engine)
{
	return new Engine_Ext_DebyeMaterial(this);
}

void Operator_Ext_DebyeMaterial::ShowStat(std::ostream &ostr)  const
{
	Operator_Extension::ShowStat(ostr);
	ostr << " Debye poles N       \t: " << m_PoleCount << endl;
	ostr << " Active cells        \t: " << m_LM_Count.at(0) << endl;
}
