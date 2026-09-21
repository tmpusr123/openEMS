/*
*	Copyright (C) 2026 Gadi Lahav (gadi@rfwithcare.com)
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

#include "operator_ext_invisible_pml.h"
#include "engine_ext_invisible_pml.h"
#include "fparser.hh"

#include "CSPrimBox.h"

using std::cerr;
using std::endl;

Operator_Ext_InvisiblePML::Operator_Ext_InvisiblePML(Operator* op) : Operator_Extension(op)
{
	setlocale(LC_NUMERIC, "en_US.UTF-8");
	m_GradingFunction = new FunctionParser();
	// the default grading function of Operator_Ext_UPML
	SetGradingFunction(" -log(1e-6)*log(2.5)/(2*dl*Z*(pow(2.5,W/dl)-1)) * pow(2.5, D/dl) ");

	m_ny = m_nyP = m_nyPP = -1;
	m_normalSignPositive = true;
	m_numCells = 0;
	m_delta = 0;
	for (int n=0; n<3; ++n)
	{
		m_sheetX0[n] = 0;
		m_sheetX1[n] = 0;
		m_numLines[n] = 0;
	}
}

Operator_Ext_InvisiblePML::~Operator_Ext_InvisiblePML()
{
	delete m_GradingFunction;
	m_GradingFunction = NULL;
}

bool Operator_Ext_InvisiblePML::SetGradingFunction(std::string func)
{
	if (func.empty())
		return true;

	m_GradFunc = func;
	int res = m_GradingFunction->Parse(m_GradFunc.c_str(), "D,dl,W,Z,N");
	if (res < 0) return true;

	cerr << "Operator_Ext_InvisiblePML::SetGradingFunction: Warning, an error occurred parsing the pml grading function (see below) ..." << endl;
	cerr << func << "\n" << std::string(res, ' ') << "^\n" << m_GradingFunction->ErrorMsg() << "\n";
	return false;
}

bool Operator_Ext_InvisiblePML::SetInitParams(CSPrimitives* prim, CSPropAbsorbingBC* abc_prop)
{
	CSPrimBox* cSheet = dynamic_cast<CSPrimBox*>(prim);
	if (!cSheet)
	{
		cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Absorbing sheet is not a box, skipping. "
			 << " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	int Snap_Dimension = m_Op->SnapBox2Mesh(
		cSheet->GetStartCoord()->GetCoords(m_Op->m_MeshType),
		cSheet->GetStopCoord()->GetCoords(m_Op->m_MeshType),
		m_sheetX0, m_sheetX1, false, true);

	if (Snap_Dimension <= 0)
	{
		if (Snap_Dimension >= -1)
			cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Absorbing sheet snapping failed! Dimension is: " << Snap_Dimension << " skipping. "
				 << " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	unsigned int sheetCheck = 0;
	m_ny = -1;
	for (int n=0; n<3; ++n)
	{
		if (m_sheetX1[n] == m_sheetX0[n])
		{
			++sheetCheck;
			m_ny = n;
		}
	}
	if (sheetCheck != 1)
	{
		cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Absorbing sheet is not a sheet! Skipping. "
			 << " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}
	m_nyP  = (m_ny+1)%3;
	m_nyPP = (m_ny+2)%3;

	switch (abc_prop->GetAbsorbingBoundaryType())
	{
	case CSPropAbsorbingBC::PML_8:
		m_numCells = 8;
		break;
	case CSPropAbsorbingBC::PML_16:
		m_numCells = 16;
		break;
	case CSPropAbsorbingBC::PML_32:
		m_numCells = 32;
		break;
	default:
		cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Not an invisible PML type, skipping. "
			 << " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	m_normalSignPositive = abc_prop->GetNormalSignPositive();

	// the first cell in front of the sheet is the template of the virtual PML
	unsigned int k0 = m_sheetX0[m_ny];
	if ((m_normalSignPositive && (k0+2 >= m_Op->GetNumberOfLines(m_ny,true))) || (!m_normalSignPositive && (k0 < 2)))
	{
		cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Absorbing sheet has no room in front of it, skipping. "
			 << " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	for (int n=1; n<3; ++n)
	{
		int nT = (m_ny+n)%3;
		if ((m_sheetX0[nT]!=0) || (m_sheetX1[nT]!=m_Op->GetNumberOfLines(nT,true)-1))
			cerr << "Operator_Ext_InvisiblePML::SetInitParams(): Warning: Sheet does not span the full cross-section in direction " << nT
				 << ". The virtual PML is bounded by the sheet's rim. ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
	}

	prim->SetPrimitiveUsed(true);
	return true;
}

double Operator_Ext_InvisiblePML::GradingKappa(double depthCells) const
{
	if (depthCells <= 0)
		return 0;
	double vars[5] = {depthCells*m_delta, m_delta, m_numCells*m_delta, __Z0__, (double)m_numCells};
	return m_GradingFunction->Eval(vars);
}

bool Operator_Ext_InvisiblePML::BuildExtension()
{
	/* The coefficients are those of Operator_Ext_UPML::BuildExtension(), with the
	   material, the PEC/PMC flags and the "disabled" (kappa==0 or metal) operator
	   taken from the first cell in front of the sheet (the template). This assumes
	   the mesh is uniform along ny over the first two cells in front of the sheet;
	   the virtual PML cells have that same size. */
	if (m_Op==NULL)
		return false;

	int s = m_normalSignPositive ? +1 : -1;
	unsigned int k0 = m_sheetX0[m_ny];

	m_delta = fabs(m_Op->GetDiscLine(m_ny,k0+s) - m_Op->GetDiscLine(m_ny,k0)) * m_Op->GetGridDelta();
	double delta2 = fabs(m_Op->GetDiscLine(m_ny,k0+2*s) - m_Op->GetDiscLine(m_ny,k0+s)) * m_Op->GetGridDelta();
	if (fabs(delta2/m_delta-1) > 0.01)
		cerr << "Operator_Ext_InvisiblePML::BuildExtension(): Warning: The mesh in front of the sheet is not uniform ("
			 << m_delta << " vs. " << delta2 << " m); the virtual PML uses " << m_delta << " m cells." << endl;

	m_numLines[m_ny]   = m_numCells + 2;
	m_numLines[m_nyP]  = m_sheetX1[m_nyP]  - m_sheetX0[m_nyP]  + 1;
	m_numLines[m_nyPP] = m_sheetX1[m_nyPP] - m_sheetX0[m_nyPP] + 1;

	vv_m.Init("vv_m", m_numLines);
	vi_m.Init("vi_m", m_numLines);
	vv.Init("vv", m_numLines);
	vvfo.Init("vvfo", m_numLines);
	vvfn.Init("vvfn", m_numLines);
	ii_m.Init("ii_m", m_numLines);
	iv_m.Init("iv_m", m_numLines);
	ii.Init("ii", m_numLines);
	iifo.Init("iifo", m_numLines);
	iifn.Init("iifn", m_numLines);

	// template positions along ny: node plane / edge (or dual line) in front of the sheet
	unsigned int tNode = k0 + s;
	unsigned int tHalf = m_normalSignPositive ? k0 : k0-1;

	// the ranges along ny of the updated fields, see the header
	unsigned int N = m_numCells;
	unsigned int lNorm0 = m_normalSignPositive ? 0 : 1;	// E normal and H tangential
	unsigned int lNorm1 = m_normalSignPositive ? N-1 : N;

	int bb[3] = {-1,-1,-1};
	bb[m_ny] = tNode;
	std::vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(bb[0], bb[1], bb[2], CSProperties::MATERIAL);

	double dT = m_Op->GetTimestep();
	double eff_Mat[4];
	unsigned int loc[3];
	unsigned int tpos[3];

	for (int n=0; n<3; ++n)
	{
		int nP  = (n+1)%3;
		int nPP = (n+2)%3;
		bool normal = (n==m_ny);

		for (int field=0; field<2; ++field) // 0: voltages, 1: currents
		{
			unsigned int l0, l1, tny;
			bool nodeDepth;
			if (field==0)
			{
				l0 = normal ? lNorm0 : 1;
				l1 = normal ? lNorm1 : N;
				tny = normal ? tHalf : tNode;
				nodeDepth = !normal;
			}
			else
			{
				l0 = normal ? 1 : lNorm0;
				l1 = normal ? N : lNorm1;
				tny = normal ? tNode : tHalf;
				nodeDepth = normal;
			}

			for (loc[m_nyP]=0; loc[m_nyP]<m_numLines[m_nyP]; ++loc[m_nyP])
			{
				tpos[m_nyP] = loc[m_nyP] + m_sheetX0[m_nyP];
				for (loc[m_nyPP]=0; loc[m_nyPP]<m_numLines[m_nyPP]; ++loc[m_nyPP])
				{
					tpos[m_nyPP] = loc[m_nyPP] + m_sheetX0[m_nyPP];
					tpos[m_ny] = tny;

					m_Op->Calc_EffMatPos(n, tpos, eff_Mat, vPrims);

					for (loc[m_ny]=l0; loc[m_ny]<=l1; ++loc[m_ny])
					{
						double depth = nodeDepth ? NodeDepth(loc[m_ny]) : HalfDepth(loc[m_ny]);
						double kappa[3] = {0,0,0};
						kappa[m_ny] = GradingKappa(depth);
						unsigned int i=loc[0], j=loc[1], k=loc[2];

						if (field==0)
						{
							double t_vv = m_Op->GetVV(n, tpos[0], tpos[1], tpos[2]);
							double t_vi = m_Op->GetVI(n, tpos[0], tpos[1], tpos[2]);
							// if eff_Mat[1] > 1e3 assume a metal and disable PML to continue a signal layer
							if ((kappa[m_ny]!=0) && (eff_Mat[1]<1e3))
							{
								// check if the template is on PEC; if so, all coefficients stay zero
								if ((t_vv + t_vi) != 0)
								{
									vv_m(n,i,j,k) = (2*__EPS0__ - kappa[nP]*dT) / (2*__EPS0__ + kappa[nP]*dT);
									vi_m(n,i,j,k) = (2*__EPS0__*dT) / (2*__EPS0__ + kappa[nP]*dT) * m_Op->GetEdgeLength(n,tpos) / m_Op->GetEdgeArea(n,tpos);
									vv(n,i,j,k)   = (2*__EPS0__ - kappa[nPP]*dT) / (2*__EPS0__ + kappa[nPP]*dT);
									vvfn(n,i,j,k) = (2*__EPS0__ + kappa[n]*dT)   / (2*__EPS0__ + kappa[nPP]*dT)/eff_Mat[0];
									vvfo(n,i,j,k) = (2*__EPS0__ - kappa[n]*dT)   / (2*__EPS0__ + kappa[nPP]*dT)/eff_Mat[0];
								}
							}
							else
							{
								// disabled pml: the plain update of the template
								vv(n,i,j,k)   = t_vv;
								vv_m(n,i,j,k) = 0;
								vi_m(n,i,j,k) = t_vi;
								vvfo(n,i,j,k) = 0;
								vvfn(n,i,j,k) = 1;
							}
						}
						else
						{
							double t_ii = m_Op->GetII(n, tpos[0], tpos[1], tpos[2]);
							double t_iv = m_Op->GetIV(n, tpos[0], tpos[1], tpos[2]);
							if (kappa[m_ny]!=0)
							{
								// check if the template is on PMC
								if ((t_ii + t_iv) != 0)
								{
									ii_m(n,i,j,k) = (2*__EPS0__ - kappa[nP]*dT) / (2*__EPS0__ + kappa[nP]*dT);
									iv_m(n,i,j,k) = (2*__EPS0__*dT) / (2*__EPS0__ + kappa[nP]*dT) * m_Op->GetEdgeLength(n,tpos,true) / m_Op->GetEdgeArea(n,tpos,true);
									ii(n,i,j,k)   = (2*__EPS0__ - kappa[nPP]*dT) / (2*__EPS0__ + kappa[nPP]*dT);
									iifn(n,i,j,k) = (2*__EPS0__ + kappa[n]*dT)   / (2*__EPS0__ + kappa[nPP]*dT)/eff_Mat[2];
									iifo(n,i,j,k) = (2*__EPS0__ - kappa[n]*dT)   / (2*__EPS0__ + kappa[nPP]*dT)/eff_Mat[2];
								}
							}
							else
							{
								ii(n,i,j,k)   = t_ii;
								ii_m(n,i,j,k) = 0;
								iv_m(n,i,j,k) = t_iv;
								iifo(n,i,j,k) = 0;
								iifn(n,i,j,k) = 1;
							}
						}
					}
				}
			}
		}
	}
	return true;
}

Engine_Extension* Operator_Ext_InvisiblePML::CreateEngineExtention()
{
	return new Engine_Ext_InvisiblePML(this);
}

void Operator_Ext_InvisiblePML::ShowStat(std::ostream &ostr) const
{
	Operator_Extension::ShowStat(ostr);

	const char dirs[3] = {'x','y','z'};
	ostr << " Sheet\t\t\t: " << dirs[m_ny] << "-normal, line " << m_sheetX0[m_ny]
		 << (m_normalSignPositive ? ", domain on the positive side" : ", domain on the negative side") << endl;
	ostr << " Virtual PML\t\t: " << m_numCells << " cells of " << m_delta << " m, "
		 << m_numLines[0] << "x" << m_numLines[1] << "x" << m_numLines[2] << " local lines" << endl;
	ostr << " Grading function\t: \"" << m_GradFunc << "\"" << endl;
}
