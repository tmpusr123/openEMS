/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_lorentzmaterial.h"
#include "engine_ext_lorentzmaterial.h"
#include "operator_ext_cylinder.h"
#include "../operator_cylinder.h"

#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#ifdef _OPENMP
#include <omp.h>
#endif

Operator_Ext_LorentzMaterial::Operator_Ext_LorentzMaterial(Operator* op) : Operator_Ext_Dispersive(op)
{
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	m_curr_Lor_ADE_On = NULL;
	m_curr_Lor_ADE_On = NULL;
}

Operator_Ext_LorentzMaterial::Operator_Ext_LorentzMaterial(Operator* op, Operator_Ext_LorentzMaterial* op_ext) : Operator_Ext_Dispersive(op,op_ext)
{
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	m_curr_Lor_ADE_On = NULL;
	m_curr_Lor_ADE_On = NULL;
}

Operator_Ext_LorentzMaterial::~Operator_Ext_LorentzMaterial()
{
	for (int i=0;i<m_Order;++i)
	{
		for (int n=0; n<3; ++n)
		{
			if (m_volt_ADE_On[i])
			{
				delete[] v_int_ADE[i][n];
				delete[] v_ext_ADE[i][n];
			}
			if (m_curr_ADE_On[i])
			{
				delete[] i_int_ADE[i][n];
				delete[] i_ext_ADE[i][n];
			}
			if (m_volt_Lor_ADE_On[i])
				delete[] v_Lor_ADE[i][n];
			if (m_curr_Lor_ADE_On[i])
				delete[] i_Lor_ADE[i][n];
		}
		if (m_volt_ADE_On[i])
		{
			delete[] v_int_ADE[i];
			delete[] v_ext_ADE[i];
		}
		if (m_curr_ADE_On[i])
		{
			delete[] i_int_ADE[i];
			delete[] i_ext_ADE[i];
		}
		if (m_volt_Lor_ADE_On[i])
			delete[] v_Lor_ADE[i];
		if (m_curr_Lor_ADE_On[i])
			delete[] i_Lor_ADE[i];
	}
	delete[] v_int_ADE;
	delete[] v_ext_ADE;
	delete[] i_int_ADE;
	delete[] i_ext_ADE;
	v_int_ADE = NULL;
	v_ext_ADE = NULL;
	i_int_ADE = NULL;
	i_ext_ADE = NULL;

	delete[] v_Lor_ADE;
	delete[] i_Lor_ADE;
	v_Lor_ADE = NULL;
	i_Lor_ADE = NULL;

	delete[] m_curr_Lor_ADE_On;
	delete[] m_volt_Lor_ADE_On;
	m_curr_Lor_ADE_On = NULL;
	m_curr_Lor_ADE_On = NULL;
}

Operator_Extension* Operator_Ext_LorentzMaterial::Clone(Operator* op)
{
	if (dynamic_cast<Operator_Ext_LorentzMaterial*>(this)==NULL)
		return NULL;
	return new Operator_Ext_LorentzMaterial(op, this);
}

bool Operator_Ext_LorentzMaterial::BuildExtension()
{
	double dT = m_Op->GetTimestep();
	unsigned int pos[] = {0,0,0};
	double coord[3];
	unsigned int numLines[3] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};
	CSPropLorentzMaterial* mat = NULL;
	CSPropDebyeMaterial* debye_mat = NULL;

	bool warn_once = true;

	bool b_pos_on;
	vector<unsigned int> v_pos[3];

	// drude material parameter
	double w_plasma,t_relax;
	double L_D[3], C_D[3];
	double R_D[3], G_D[3];
	vector<double> v_int[3];
	vector<double> v_ext[3];
	vector<double> i_int[3];
	vector<double> i_ext[3];

	//additional Dorentz material parameter
	double w_Lor_Pol;
	double C_L[3];
	double L_L[3];
	vector<double> v_Lor[3];
	vector<double> i_Lor[3];

	m_Order = 0;
	vector<CSProperties*> LD_props = m_Op->CSX->GetPropertyByType(CSProperties::LORENTZMATERIAL);
	for (size_t n=0;n<LD_props.size();++n)
	{
		CSPropLorentzMaterial* LorMat = dynamic_cast<CSPropLorentzMaterial*>(LD_props.at(n));
		if (LorMat==NULL)
			return false; //sanity check, this should not happen
		if (LorMat->GetDispersionOrder()>m_Order)
			m_Order=LorMat->GetDispersionOrder();
	}
	LD_props = m_Op->CSX->GetPropertyByType(CSProperties::DEBYEMATERIAL);
	for (size_t n=0;n<LD_props.size();++n)
	{
		CSPropDebyeMaterial* DebyeMat = dynamic_cast<CSPropDebyeMaterial*>(LD_props.at(n));
		if (DebyeMat==NULL)
			return false; //sanity check, this should not happen
		if (DebyeMat->GetDispersionOrder()>m_Order)
			m_Order=DebyeMat->GetDispersionOrder();
	}

	m_LM_pos = new unsigned int**[m_Order];

	m_volt_ADE_On = new bool[m_Order];
	m_curr_ADE_On = new bool[m_Order];
	m_volt_Lor_ADE_On = new bool[m_Order];
	m_curr_Lor_ADE_On = new bool[m_Order];

	v_int_ADE = new FDTD_FLOAT**[m_Order];
	v_ext_ADE = new FDTD_FLOAT**[m_Order];
	i_int_ADE = new FDTD_FLOAT**[m_Order];
	i_ext_ADE = new FDTD_FLOAT**[m_Order];

	v_Lor_ADE = new FDTD_FLOAT**[m_Order];
	i_Lor_ADE = new FDTD_FLOAT**[m_Order];

	for (int order=0;order<m_Order;++order)
	{
		m_volt_ADE_On[order]=false;
		m_curr_ADE_On[order]=false;

		m_volt_Lor_ADE_On[order]=false;
		m_curr_Lor_ADE_On[order]=false;

		for (int n=0;n<3;++n)
		{
			v_pos[n].clear();

			v_int[n].clear();
			v_ext[n].clear();
			i_int[n].clear();
			i_ext[n].clear();

			v_Lor[n].clear();
			i_Lor[n].clear();
		}

		// Per-cell dispersive-material detection over the WHOLE volume: two
		// GetPropertyByCoordPriority point-queries per cell per component, repeated
		// for EVERY dispersion order. This was the last fully-serial stage of
		// operator setup (~3.5 h single-threaded on a 1G-cell Debye3 model with STL
		// geometry, while 47 of 48 cores idled). Parallelized over x-planes: each
		// plane collects into its own buffers, concatenated in x order afterwards,
		// which reproduces the serial push_back ordering EXACTLY (the engine indexes
		// these arrays linearly) -> bit-identical. The MainOp cursor is read with
		// the stateless, cursor-at-origin GetPos instead of the mutating SetPos.
		//
		// The m_*_ADE_On[order] flags become OR-reductions. Their in-loop READ
		// (e.g. "t_relax>0 && f_volt") is provably benign: it only gates
		// R_D[n]=L_D[n]/t_relax, and L_D[n]>0 implies the flag was set by this very
		// cell, while L_D[n]==0 gives R_D[n]==0 whichever way the flag reads.
		{
		const int _nx = (int)numLines[0];
		std::vector< std::vector<unsigned int> > pl_v_pos(_nx*3);
		std::vector< std::vector<double> > pl_v_int(_nx*3), pl_v_ext(_nx*3),
		                                   pl_i_int(_nx*3), pl_i_ext(_nx*3),
		                                   pl_v_Lor(_nx*3), pl_i_Lor(_nx*3);
		bool f_volt=false, f_curr=false, f_vLor=false, f_iLor=false, f_warn=false;
		m_Op->MainOp->SetPos(0,0,0);   // cursor at origin -> GetPos below is a pure read
#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic,1) \
			reduction(||:f_volt,f_curr,f_vLor,f_iLor,f_warn)
#endif
		for (int _x=0; _x<_nx; ++_x)
		{
			unsigned int pos[3]; pos[0]=(unsigned int)_x;
			double coord[3];
			double L_D[3], C_D[3], R_D[3], G_D[3], C_L[3], L_L[3];
			double w_plasma, t_relax, w_Lor_Pol;
			CSPropLorentzMaterial* mat = NULL;
			CSPropDebyeMaterial* debye_mat = NULL;
			bool b_pos_on;
			for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
			{
				vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(pos[0], pos[1], -1, (CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL));
				for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
				{
					unsigned int index = m_Op->MainOp->GetPos(pos[0],pos[1],pos[2]);
					//calc epsilon lorentz material
					b_pos_on = false;
					for (int n=0; n<3; ++n)
					{
						L_D[n]=0;
						R_D[n]=0;
						C_L[n]=0;
						if (m_Op->GetYeeCoords(n,pos,coord,false)==false)
							continue;
						if (m_CC_R0_included && (n==2) && (pos[0]==0))
							coord[1] = m_Op->GetDiscLine(1,0);

						if (m_Op->GetVI(n,pos[0],pos[1],pos[2])==0)
							continue;

//						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord,(CSProperties::PropertyType)(CSProperties::METAL | CSProperties::MATERIAL), true);
						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, true);

						if (prop==NULL) continue;

						if ((mat = prop->ToLorentzMaterial()))
						{
							w_plasma = mat->GetEpsPlasmaFreqWeighted(order,n,coord) * 2 * PI;
							if ((w_plasma>0) && (m_Op->EC_C[n][index]>0))
							{
								b_pos_on = true;
								f_volt = true;
								L_D[n] = 1/(w_plasma*w_plasma*m_Op->EC_C[n][index]);
							}
							t_relax = mat->GetEpsRelaxTimeWeighted(order,n,coord);
							if ((t_relax>0) && f_volt)
							{
								R_D[n] = L_D[n]/t_relax;
							}
							w_Lor_Pol = mat->GetEpsLorPoleFreqWeighted(order,n,coord) * 2 * PI;
							if ((w_Lor_Pol>0) && (L_D[n]>0))
							{
								f_vLor = true;
								C_L[n] = 1/(w_Lor_Pol*w_Lor_Pol*L_D[n]);
							}
						}
						if ((debye_mat = prop->ToDebyeMaterial()))
						{
							C_L[n] = 8.85418781762e-12*debye_mat->GetEpsDeltaWeighted(order,n,coord) * m_Op->GetEdgeArea(n, pos) / m_Op->GetEdgeLength(n,pos);
							t_relax = debye_mat->GetEpsRelaxTimeWeighted(order,n,coord);
							if (t_relax<2.0*dT)
								f_warn = true;
							if ((C_L[n]>0) && (t_relax>0) && (t_relax>2.0*dT))
							{
								R_D[n] = t_relax/C_L[n];
								b_pos_on = true;
								f_volt = true;
								f_vLor = true;
							}
						}
					}

					for (int n=0; n<3; ++n)
					{
						C_D[n]=0;
						G_D[n]=0;
						L_L[n]=0;
						if (m_Op->GetYeeCoords(n,pos,coord,true)==false)
							continue;
						if (m_Op->GetIV(n,pos[0],pos[1],pos[2])==0)
							continue;

//						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord,(CSProperties::PropertyType)(CSProperties::METAL | CSProperties::MATERIAL), true);
						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrims, true);

						if (prop==NULL) continue;

						if ((mat = prop->ToLorentzMaterial()))
						{
							w_plasma = mat->GetMuePlasmaFreqWeighted(order,n,coord) * 2 * PI;
							if ((w_plasma>0) && (m_Op->EC_L[n][index]>0))
							{
								b_pos_on = true;
								f_curr = true;
								C_D[n] = 1/(w_plasma*w_plasma*m_Op->EC_L[n][index]);
							}
							t_relax = mat->GetMueRelaxTimeWeighted(order,n,coord);
							if ((t_relax>0) && f_curr)
							{
								G_D[n] = C_D[n]/t_relax;
							}
							w_Lor_Pol = mat->GetMueLorPoleFreqWeighted(order,n,coord) * 2 * PI;
							if ((w_Lor_Pol>0) && (C_D[n]>0))
							{
								f_iLor = true;
								L_L[n] = 1/(w_Lor_Pol*w_Lor_Pol*C_D[n]);
							}
						}
					}

					if (b_pos_on) //this position has active drude material
					{
						for (unsigned int n=0; n<3; ++n)
						{
							pl_v_pos[_x*3+n].push_back(pos[n]);
							if (L_D[n]>0)
							{
								pl_v_int[_x*3+n].push_back((2.0*L_D[n]-dT*R_D[n])/(2.0*L_D[n]+dT*R_D[n]));
								// check for r==0 in clyindrical coords and get special VI coefficient
								if (m_CC_R0_included && n==2 && pos[0]==0)
									pl_v_ext[_x*3+n].push_back(dT/(L_D[n]+dT*R_D[n]/2.0)*m_Op_Cyl->m_Cyl_Ext->vi_R0[pos[2]]);
								else
									pl_v_ext[_x*3+n].push_back(dT/(L_D[n]+dT*R_D[n]/2.0)*m_Op->GetVI(n,pos[0],pos[1],pos[2]));
							}
							else if ((R_D[n]>0) && (C_L[n]>0))
							{
								pl_v_int[_x*3+n].push_back((2.0*dT-R_D[n]*C_L[n])/(C_L[n]*R_D[n]));
								pl_v_ext[_x*3+n].push_back(2.0/R_D[n]*m_Op->GetVI(n,pos[0],pos[1],pos[2]));
							}
							else
							{
								pl_v_int[_x*3+n].push_back(1);
								pl_v_ext[_x*3+n].push_back(0);
							}
							if (C_D[n]>0)
							{
								pl_i_int[_x*3+n].push_back((2.0*C_D[n]-dT*G_D[n])/(2.0*C_D[n]+dT*G_D[n]));
								pl_i_ext[_x*3+n].push_back(dT/(C_D[n]+dT*G_D[n]/2.0)*m_Op->GetIV(n,pos[0],pos[1],pos[2]));
							}
							else
							{
								pl_i_int[_x*3+n].push_back(1);
								pl_i_ext[_x*3+n].push_back(0);
							}
							if (C_L[n]>0)
								pl_v_Lor[_x*3+n].push_back(dT/C_L[n]/m_Op->GetVI(n,pos[0],pos[1],pos[2]));
							else
								pl_v_Lor[_x*3+n].push_back(0);
							if (L_L[n]>0)
								pl_i_Lor[_x*3+n].push_back(dT/L_L[n]/m_Op->GetIV(n,pos[0],pos[1],pos[2]));
							else
								pl_i_Lor[_x*3+n].push_back(0);
						}
					}
				}
			}
		}

		// concatenate per-plane buffers in x order == the serial push_back order
		{
			size_t _tot=0;
			for (int _x=0;_x<_nx;++_x) _tot += pl_v_pos[_x*3].size();
			for (int n=0;n<3;++n)
			{
				v_pos[n].reserve(_tot); v_int[n].reserve(_tot); v_ext[n].reserve(_tot);
				i_int[n].reserve(_tot); i_ext[n].reserve(_tot);
				v_Lor[n].reserve(_tot); i_Lor[n].reserve(_tot);
			}
			for (int _x=0;_x<_nx;++_x)
				for (int n=0;n<3;++n)
				{
					const int k=_x*3+n;
					v_pos[n].insert(v_pos[n].end(), pl_v_pos[k].begin(), pl_v_pos[k].end());
					v_int[n].insert(v_int[n].end(), pl_v_int[k].begin(), pl_v_int[k].end());
					v_ext[n].insert(v_ext[n].end(), pl_v_ext[k].begin(), pl_v_ext[k].end());
					i_int[n].insert(i_int[n].end(), pl_i_int[k].begin(), pl_i_int[k].end());
					i_ext[n].insert(i_ext[n].end(), pl_i_ext[k].begin(), pl_i_ext[k].end());
					v_Lor[n].insert(v_Lor[n].end(), pl_v_Lor[k].begin(), pl_v_Lor[k].end());
					i_Lor[n].insert(i_Lor[n].end(), pl_i_Lor[k].begin(), pl_i_Lor[k].end());
					std::vector<unsigned int>().swap(pl_v_pos[k]);   // free as we merge
					std::vector<double>().swap(pl_v_int[k]);
					std::vector<double>().swap(pl_v_ext[k]);
					std::vector<double>().swap(pl_i_int[k]);
					std::vector<double>().swap(pl_i_ext[k]);
					std::vector<double>().swap(pl_v_Lor[k]);
					std::vector<double>().swap(pl_i_Lor[k]);
				}
		}
		if (f_volt) m_volt_ADE_On[order]     = true;
		if (f_curr) m_curr_ADE_On[order]     = true;
		if (f_vLor) m_volt_Lor_ADE_On[order] = true;
		if (f_iLor) m_curr_Lor_ADE_On[order] = true;
		if (f_warn && warn_once)
		{
			warn_once = false;
			cerr << "Operator_Ext_LorentzMaterial::BuildExtension(): Warning, debye relaxation time is to small, skipping..." << endl;
		}
		}

		//copy all vectors into the array's
		m_LM_Count.push_back(v_pos[0].size());

		m_LM_pos[order] = new unsigned int*[3];

		if (m_volt_ADE_On[order])
		{
			v_int_ADE[order] = new FDTD_FLOAT*[3];
			v_ext_ADE[order] = new FDTD_FLOAT*[3];
		}
		else
		{
			v_int_ADE[order] = NULL;
			v_ext_ADE[order] = NULL;
		}

		if (m_curr_ADE_On[order])
		{
			i_int_ADE[order] = new FDTD_FLOAT*[3];
			i_ext_ADE[order] = new FDTD_FLOAT*[3];
		}
		else
		{
			i_int_ADE[order] = NULL;
			i_ext_ADE[order] = NULL;
		}

		if (m_volt_Lor_ADE_On[order])
			v_Lor_ADE[order] = new FDTD_FLOAT*[3];
		else
			v_Lor_ADE[order] = NULL;

		if (m_curr_Lor_ADE_On[order])
			i_Lor_ADE[order] = new FDTD_FLOAT*[3];
		else
			i_Lor_ADE[order] = NULL;

		for (int n=0; n<3; ++n)
		{
			m_LM_pos[order][n] = new unsigned int[m_LM_Count.at(order)];
			for (unsigned int i=0; i<m_LM_Count.at(order); ++i)
				m_LM_pos[order][n][i] = v_pos[n].at(i);
			if (m_volt_ADE_On[order])
			{
				v_int_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				v_ext_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];

				for (unsigned int i=0; i<m_LM_Count.at(order); ++i)
				{
					v_int_ADE[order][n][i] = v_int[n].at(i);
					v_ext_ADE[order][n][i] = v_ext[n].at(i);
				}
			}
			if (m_curr_ADE_On[order])
			{
				i_int_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				i_ext_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];

				for (unsigned int i=0; i<m_LM_Count.at(order); ++i)
				{
					i_int_ADE[order][n][i] = i_int[n].at(i);
					i_ext_ADE[order][n][i] = i_ext[n].at(i);
				}
			}

			if (m_volt_Lor_ADE_On[order])
			{
				v_Lor_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				for (unsigned int i=0; i<m_LM_Count.at(order); ++i)
					v_Lor_ADE[order][n][i] = v_Lor[n].at(i);
			}
			if (m_curr_Lor_ADE_On[order])
			{
				i_Lor_ADE[order][n]  = new FDTD_FLOAT[m_LM_Count.at(order)];
				for (unsigned int i=0; i<m_LM_Count.at(order); ++i)
					i_Lor_ADE[order][n][i] = i_Lor[n].at(i);
			}
		}
	}

	return true;
}

Engine_Extension* Operator_Ext_LorentzMaterial::CreateEngineExtention(Engine *engine)
{
	Engine_Ext_LorentzMaterial* eng_ext_lor = new Engine_Ext_LorentzMaterial(this);
	return eng_ext_lor;
}

void Operator_Ext_LorentzMaterial::ShowStat(ostream &ostr)  const
{
	Operator_Extension::ShowStat(ostr);
	string On_Off[2] = {"Off", "On"};
	ostr << " Max. Dispersion Order N = " << m_Order << endl;
	for (int i=0;i<m_Order;++i)
	{
		ostr << " N=" << i << ":\t Active cells\t\t: " << 	m_LM_Count.at(i) << endl;
		ostr << " N=" << i << ":\t Voltage ADE is \t: " << On_Off[m_volt_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Voltage Lor-ADE is \t: " << On_Off[m_volt_Lor_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Current ADE is \t: " << On_Off[m_curr_ADE_On[i]] << endl;
		ostr << " N=" << i << ":\t Current Lor-ADE is \t: " << On_Off[m_curr_Lor_ADE_On[i]] << endl;
	}
}
