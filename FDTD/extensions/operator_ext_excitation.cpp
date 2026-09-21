/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_excitation.h"
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "engine_ext_excitation.h"
#include "FDTD/excitation.h"
#include "ContinuousStructure.h"

#include "CSPrimCurve.h"
#include "CSPropExcitation.h"
#include "CSPrimBox.h"

Operator_Ext_Excitation::Operator_Ext_Excitation(Operator* op) : Operator_Extension(op)
{
	Init();
}

Operator_Ext_Excitation::~Operator_Ext_Excitation()
{
	Reset();
}

Operator_Extension* Operator_Ext_Excitation::Clone(Operator* op)
{
	Operator_Ext_Excitation* clone = new Operator_Ext_Excitation(op, this);
	return clone;
}

void Operator_Ext_Excitation::Init()
{
	Operator_Extension::Init();
	Volt_delay = 0;
	Volt_amp = 0;
	Volt_dir = 0;
	Volt_Count = 0;
	Curr_delay = 0;
	Curr_amp = 0;
	Curr_dir = 0;
	Curr_Count = 0;

	for (int n=0; n<3; ++n)
	{
		Volt_index[n] = 0;
		Curr_index[n] = 0;
		Volt_Count_Dir[n] = 0;
		Curr_Count_Dir[n] = 0;
	}
	m_Exc = 0;
}

void Operator_Ext_Excitation::Reset()
{
	Operator_Extension::Reset();
	delete[] Volt_delay;
	Volt_delay = 0;
	delete[] Volt_dir;
	Volt_dir = 0;
	delete[] Volt_amp;
	Volt_amp = 0;
	delete[] Curr_delay;
	Curr_delay = 0;
	delete[] Curr_dir;
	Curr_dir = 0;
	delete[] Curr_amp;
	Curr_amp = 0;

	Volt_Count = 0;
	Curr_Count = 0;

	for (int n=0; n<3; ++n)
	{
		delete[] Volt_index[n];
		Volt_index[n] = 0;
		delete[] Curr_index[n];
		Curr_index[n] = 0;

		Volt_Count_Dir[n] = 0;
		Curr_Count_Dir[n] = 0;
	}
}


Operator_Ext_Excitation::Operator_Ext_Excitation(Operator* op, Operator_Ext_Excitation* op_ext) : Operator_Extension(op, op_ext)
{
	Init();
}

bool Operator_Ext_Excitation::shiftCoordsForModeFile(double * const coord0, CSPrimitives * cPrim)
{
	CSPrimBox* primBox = cPrim->ToBox();
	if (primBox == NULL)
	{
		cerr << "Operator_Ext_Excitation::BuildExtension: Error obtaining box primitive" << endl;
		return false;
	}

	for (int dirIdx = 0 ; dirIdx < 3 ; dirIdx++)
	{
		double OS = primBox->GetCoord(int(dirIdx*2));
		coord0[dirIdx] -= OS;
	}

	return true;

}

bool Operator_Ext_Excitation::BuildExtension()
{
	m_Exc = m_Op->GetExcitationSignal();
	double dT = m_Op->GetTimestep();
	if (dT==0)
		return false;
	if (m_Exc==0)
		return false;

	Reset();
	ContinuousStructure* CSX = m_Op->GetGeometryCSX();

	unsigned int pos[3];
	double amp=0;

	vector<unsigned int> volt_vIndex[3];
	vector<FDTD_FLOAT> volt_vExcit;
	vector<unsigned int> volt_vDelay;
	vector<unsigned int> volt_vDir;
	double volt_coord[3];

	vector<unsigned int> curr_vIndex[3];
	vector<FDTD_FLOAT> curr_vExcit;
	vector<unsigned int> curr_vDelay;
	vector<unsigned int> curr_vDir;
	double curr_coord[3];

	vector<CSProperties*> vec_prop = CSX->GetPropertyByType(CSProperties::EXCITATION);

	if (vec_prop.size()==0)
	{
		cerr << "Operator::CalcFieldExcitation: Warning, no excitation properties found" << endl;
		return false;
	}

	CSPropExcitation* 	elec=NULL;
	CSProperties* 		prop=NULL;

	// Parse every mode file up front.  GetWeightedExcitation() parses lazily on
	// first use, which is fine serially but is a data race here: the cell scan
	// below is parallel over z-planes, so several threads would enter
	// ParseModeFile() on the same CSPropExcitation at once and corrupt the
	// heap while filling the same vectors ("double free or corruption").
	// Doing it here also keeps the per-cell path free of any locking.
	for (size_t p=0; p<vec_prop.size(); ++p)
	{
		CSPropExcitation* pe = vec_prop.at(p)->ToExcitation();
		if ((pe!=NULL) && pe->GetFieldSourceIsFile())
			if (!pe->ParseModeFile())
				cerr << "Operator_Ext_Excitation::BuildExtension: Warning, could not parse mode file '"
				     << pe->GetModeFileName() << "'" << endl;
	}

	unsigned int numLines[] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};
	// Full-volume scan: 2 GetPropertyByCoordPriority probes per cell per component.
	// Parallel over z-planes; each plane collects into its own buckets, concatenated
	// in z order afterwards -> reproduces the serial push_back ordering exactly.
	// Writes into the main operator (SetVV/SetVI) touch only the current cell.
	const int _nz = (int)numLines[2];
	std::vector< std::vector<FDTD_FLOAT> >   B_volt_vExcit(_nz), B_curr_vExcit(_nz);
	std::vector< std::vector<unsigned int> > B_volt_vDelay(_nz), B_curr_vDelay(_nz);
	std::vector< std::vector<unsigned int> > B_volt_vDir(_nz),   B_curr_vDir(_nz);
	std::vector< std::vector<unsigned int> > B_volt_vIndex0(_nz),B_volt_vIndex1(_nz),B_volt_vIndex2(_nz);
	std::vector< std::vector<unsigned int> > B_curr_vIndex0(_nz),B_curr_vIndex1(_nz),B_curr_vIndex2(_nz);
	// The primitives an excited cell was attributed to are marked used *after* the
	// loop: SetPrimitiveUsed() writes a shared object, and a mode-file shift that
	// fails cannot `return` out of an OpenMP structured block.  Both are collected
	// per z-plane and replayed serially below, which also keeps the order stable.
	std::vector< std::vector<CSPrimitives*> > B_used(_nz);
	bool shiftFailed = false;
#ifdef _OPENMP
	#pragma omp parallel for schedule(dynamic,1)
#endif
	for (int _z=0; _z<_nz; ++_z)
	{
		unsigned int pos[3];
		double volt_coord[3], curr_coord[3];
		CSPropExcitation* elec=NULL;
		CSProperties* prop=NULL;
		pos[2]=(unsigned int)_z;
		for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
		{
			vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(-1, pos[1], pos[2], CSProperties::EXCITATION);
			for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
			{
				//electric field excite
				for (int n=0; n<3; ++n)
				{
					if (m_Op->GetYeeCoords(n,pos,volt_coord,false)==false)
						continue;

					if (m_CC_R0_included && (n==2) && (pos[0]==0))
						volt_coord[1] = m_Op->GetDiscLine(1,0);

					if (m_CC_R0_included && (n==1) && (pos[0]==0))
						continue;

					// Also choose the highest priority primitive;
					CSPrimitives* 	highestPriorityPrim;
					CSProperties* 	prop = CSX->GetPropertyByCoordPriority(volt_coord, vPrims, true, &highestPriorityPrim);

					if (prop)
					{
						elec = prop->ToExcitation();
						if (elec==NULL)
							continue;
						if (!elec->GetEnabled())
							continue;

						if ((elec->GetActiveDir(n)) && ( (elec->GetExcitType()==0) || (elec->GetExcitType()==1) ))//&& (pos[n]<numLines[n]-1))
						{
							// If this is read from a file, the voltage coordinates need to be shifted
							// to the start point
							if (elec->GetFieldSourceIsFile()) // @suppress("Method cannot be resolved")
								if (!shiftCoordsForModeFile(volt_coord, highestPriorityPrim))
								{
									shiftFailed = true;
									continue;
								}

							amp = elec->GetWeightedExcitation(n,volt_coord)*m_Op->GetEdgeLength(n,pos); // delta[n]*gridDelta;

							if (amp!=0)
							{
								B_volt_vExcit[_z].push_back(amp);
								B_volt_vDelay[_z].push_back((unsigned int)(elec->GetDelay()/dT));
								B_volt_vDir[_z].push_back(n);
								B_volt_vIndex0[_z].push_back(pos[0]);
								B_volt_vIndex1[_z].push_back(pos[1]);
								B_volt_vIndex2[_z].push_back(pos[2]);

								// IFF it got this far, this primitive is used
								B_used[_z].push_back(highestPriorityPrim);
							}
							if (elec->GetExcitType()==1) //hard excite
							{
								m_Op->SetVV(n,pos[0],pos[1],pos[2], 0 );
								m_Op->SetVI(n,pos[0],pos[1],pos[2], 0 );
							}
						}
					}
				}

				//magnetic field excite
				for (int n=0; n<3; ++n)
				{
					if ((pos[0]>=numLines[0]-1) || (pos[1]>=numLines[1]-1) || (pos[2]>=numLines[2]-1))
						continue;  //skip the last H-Line which is outside the FDTD-domain
					if (m_Op->GetYeeCoords(n,pos,curr_coord,true)==false)
						continue;

					CSPrimitives*	highestPriorityPrim;
					CSProperties*	prop = CSX->GetPropertyByCoordPriority(curr_coord, vPrims, true, &highestPriorityPrim);
					if (prop)
					{
						elec = prop->ToExcitation();
						if (elec==NULL)
							continue;
						if (!elec->GetEnabled())
							continue;

						if ((elec->GetActiveDir(n)) && ( (elec->GetExcitType()==2) || (elec->GetExcitType()==3) ))
						{

							// If this is read from a file, the voltage coordinates need to be shifted
							// to the start point
							if (elec->GetFieldSourceIsFile())
								if (!shiftCoordsForModeFile(curr_coord, highestPriorityPrim))
								{
									shiftFailed = true;
									continue;
								}

							amp = elec->GetWeightedExcitation(n,curr_coord)*m_Op->GetEdgeLength(n,pos,true);// delta[n]*gridDelta;

							if (amp!=0)
							{
								B_curr_vExcit[_z].push_back(amp);
								B_curr_vDelay[_z].push_back((unsigned int)(elec->GetDelay()/dT));
								B_curr_vDir[_z].push_back(n);
								B_curr_vIndex0[_z].push_back(pos[0]);
								B_curr_vIndex1[_z].push_back(pos[1]);
								B_curr_vIndex2[_z].push_back(pos[2]);

								// IFF it got this far, this primitive is used
								B_used[_z].push_back(highestPriorityPrim);
							}
							if (elec->GetExcitType()==3) //hard excite
							{
								m_Op->SetII(n,pos[0],pos[1],pos[2], 0 );
								m_Op->SetIV(n,pos[0],pos[1],pos[2], 0 );
							}
						}
					}
				}

			}
		}
	}
	for (int _z=0; _z<_nz; ++_z)
	{
		volt_vExcit.insert(volt_vExcit.end(), B_volt_vExcit[_z].begin(), B_volt_vExcit[_z].end());
		volt_vDelay.insert(volt_vDelay.end(), B_volt_vDelay[_z].begin(), B_volt_vDelay[_z].end());
		volt_vDir.insert(  volt_vDir.end(),   B_volt_vDir[_z].begin(),   B_volt_vDir[_z].end());
		volt_vIndex[0].insert(volt_vIndex[0].end(), B_volt_vIndex0[_z].begin(), B_volt_vIndex0[_z].end());
		volt_vIndex[1].insert(volt_vIndex[1].end(), B_volt_vIndex1[_z].begin(), B_volt_vIndex1[_z].end());
		volt_vIndex[2].insert(volt_vIndex[2].end(), B_volt_vIndex2[_z].begin(), B_volt_vIndex2[_z].end());
		curr_vExcit.insert(curr_vExcit.end(), B_curr_vExcit[_z].begin(), B_curr_vExcit[_z].end());
		curr_vDelay.insert(curr_vDelay.end(), B_curr_vDelay[_z].begin(), B_curr_vDelay[_z].end());
		curr_vDir.insert(  curr_vDir.end(),   B_curr_vDir[_z].begin(),   B_curr_vDir[_z].end());
		curr_vIndex[0].insert(curr_vIndex[0].end(), B_curr_vIndex0[_z].begin(), B_curr_vIndex0[_z].end());
		curr_vIndex[1].insert(curr_vIndex[1].end(), B_curr_vIndex1[_z].begin(), B_curr_vIndex1[_z].end());
		curr_vIndex[2].insert(curr_vIndex[2].end(), B_curr_vIndex2[_z].begin(), B_curr_vIndex2[_z].end());
		for (size_t u=0; u<B_used[_z].size(); ++u)
			if (!B_used[_z][u]->GetPrimitiveUsed()) B_used[_z][u]->SetPrimitiveUsed(true);
	}
	if (shiftFailed)
		return false;

	//special treatment for primitives of type curve (treated as wires) see also Calc_PEC
	double p1[3];
	double p2[3];
	Grid_Path path;
	for (size_t p=0; p<vec_prop.size(); ++p)
	{
		prop = vec_prop.at(p);
		elec = prop->ToExcitation();
		for (size_t n=0; n<prop->GetQtyPrimitives(); ++n)
		{
			CSPrimitives* prim = prop->GetPrimitive(n);
			CSPrimCurve* curv = prim->ToCurve();
			if (curv)
			{
				for (size_t i=1; i<curv->GetNumberOfPoints(); ++i)
				{
					curv->GetPoint(i-1,p1,m_Op->m_MeshType);
					curv->GetPoint(i,p2,m_Op->m_MeshType);
					path = m_Op->FindPath(p1,p2);
					if (path.dir.size()>0)
						prim->SetPrimitiveUsed(true);
					for (size_t t=0; t<path.dir.size(); ++t)
					{
						n = path.dir.at(t);
						pos[0] = path.posPath[0].at(t);
						pos[1] = path.posPath[1].at(t);
						pos[2] = path.posPath[2].at(t);
						m_Op->GetYeeCoords(n,pos,volt_coord,false);
						if (elec!=NULL)
						{
							if ((elec->GetActiveDir(n)) && (pos[n]<numLines[n]-1) && ( (elec->GetExcitType()==0) || (elec->GetExcitType()==1) ))
							{
								amp = elec->GetWeightedExcitation(n,volt_coord)*m_Op->GetEdgeLength(n,pos);
								if (amp!=0)
								{
									volt_vExcit.push_back(amp);
									volt_vDelay.push_back((unsigned int)(elec->GetDelay()/dT));
									volt_vDir.push_back(n);
									volt_vIndex[0].push_back(pos[0]);
									volt_vIndex[1].push_back(pos[1]);
									volt_vIndex[2].push_back(pos[2]);
								}
								if (elec->GetExcitType()==1) //hard excite
								{
									m_Op->SetVV(n,pos[0],pos[1],pos[2], 0 );
									m_Op->SetVI(n,pos[0],pos[1],pos[2], 0 );
								}
							}
						}
					}
				}
			}
		}
	}

	// set voltage excitations
	setupVoltageExcitation( volt_vIndex, volt_vExcit, volt_vDelay, volt_vDir );

	// set current excitations
	setupCurrentExcitation( curr_vIndex, curr_vExcit, curr_vDelay, curr_vDir );

	return true;
}

void Operator_Ext_Excitation::setupVoltageExcitation( vector<unsigned int> const volt_vIndex[3], vector<FDTD_FLOAT> const& volt_vExcit,
		vector<unsigned int> const& volt_vDelay, vector<unsigned int> const& volt_vDir )
{
	Volt_Count = volt_vIndex[0].size();
	for (int n=0; n<3; n++)
	{
		Volt_Count_Dir[n]=0;
		delete[] Volt_index[n];
		Volt_index[n] = new unsigned int[Volt_Count];
	}
	delete[] Volt_delay;
	delete[] Volt_amp;
	delete[] Volt_dir;
	Volt_delay = new unsigned int[Volt_Count];
	Volt_amp = new FDTD_FLOAT[Volt_Count];
	Volt_dir = new unsigned short[Volt_Count];

//	cerr << "Excitation::setupVoltageExcitation(): Number of voltage excitation points: " << Volt_Count << endl;
//	if (Volt_Count==0)
//		cerr << "No E-Field/voltage excitation found!" << endl;
	for (int n=0; n<3; n++)
		for (unsigned int i=0; i<Volt_Count; i++)
			Volt_index[n][i] = volt_vIndex[n].at(i);
	for (unsigned int i=0; i<Volt_Count; i++)
	{
		Volt_delay[i] = volt_vDelay.at(i);
		Volt_amp[i]   = volt_vExcit.at(i);
		Volt_dir[i]   = volt_vDir.at(i);
		++Volt_Count_Dir[Volt_dir[i]];
	}
}

void Operator_Ext_Excitation::setupCurrentExcitation( vector<unsigned int> const curr_vIndex[3], vector<FDTD_FLOAT> const& curr_vExcit,
		vector<unsigned int> const& curr_vDelay, vector<unsigned int> const& curr_vDir )
{
	Curr_Count = curr_vIndex[0].size();
	for (int n=0; n<3; n++)
	{
		Curr_Count_Dir[n]=0;
		delete[] Curr_index[n];
		Curr_index[n] = new unsigned int[Curr_Count];
	}
	delete[] Curr_delay;
	delete[] Curr_amp;
	delete[] Curr_dir;
	Curr_delay = new unsigned int[Curr_Count];
	Curr_amp = new FDTD_FLOAT[Curr_Count];
	Curr_dir = new unsigned short[Curr_Count];

//	cerr << "Excitation::setupCurrentExcitation(): Number of current excitation points: " << Curr_Count << endl;
//	if (Curr_Count==0)
//		cerr << "No H-Field/current excitation found!" << endl;
	for (int n=0; n<3; ++n)
		for (unsigned int i=0; i<Curr_Count; i++)
			Curr_index[n][i] = curr_vIndex[n].at(i);
	for (unsigned int i=0; i<Curr_Count; i++)
	{
		Curr_delay[i] = curr_vDelay.at(i);
		Curr_amp[i]   = curr_vExcit.at(i);
		Curr_dir[i]   = curr_vDir.at(i);
		++Curr_Count_Dir[Curr_dir[i]];
	}

}

Engine_Extension* Operator_Ext_Excitation::CreateEngineExtention(Engine *engine)
{
	return new Engine_Ext_Excitation(this);
}

void Operator_Ext_Excitation::ShowStat(ostream &ostr)  const
{
	Operator_Extension::ShowStat(ostr);
	cout << "Voltage excitations\t: " << Volt_Count    << "\t (" << Volt_Count_Dir[0] << ", " << Volt_Count_Dir[1] << ", " << Volt_Count_Dir[2] << ")" << endl;
	cout << "Current excitations\t: " << Curr_Count << "\t (" << Curr_Count_Dir[0] << ", " << Curr_Count_Dir[1] << ", " << Curr_Count_Dir[2] << ")" << endl;
	cout << "Excitation Length (TS)\t: " << m_Exc->GetLength() << endl;
	cout << "Excitation Length (s)\t: " << m_Exc->GetLength()*m_Op->GetTimestep() << endl;
}

