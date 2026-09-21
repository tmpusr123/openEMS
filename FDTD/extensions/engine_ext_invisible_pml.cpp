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

#include "engine_ext_invisible_pml.h"
#include "operator_ext_invisible_pml.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"

Engine_Ext_InvisiblePML::Engine_Ext_InvisiblePML(Operator_Ext_InvisiblePML* op_ext) : Engine_Extension(op_ext)
{
	m_Op_PML = op_ext;

	// the sheet plane must be in place before the Mur ABC (and others) read it
	m_Priority = ENG_EXT_PRIO_INVISIBLE_PML;

	m_ny   = m_Op_PML->m_ny;
	m_nyP  = m_Op_PML->m_nyP;
	m_nyPP = m_Op_PML->m_nyPP;
	for (int n=0; n<3; ++n)
	{
		m_numLines[n] = m_Op_PML->m_numLines[n];
		m_offset[n] = m_Op_PML->m_sheetX0[n];
	}
	m_offset[m_ny] = 0;

	unsigned int N = m_Op_PML->m_numCells;
	bool posSide = m_Op_PML->m_normalSignPositive;
	unsigned int k0 = m_Op_PML->m_sheetX0[m_ny];

	m_lineInt   = m_Op_PML->InterfaceLine();
	m_lineGhost = m_Op_PML->GhostLine();
	m_mainInt   = k0;
	m_mainGhost = posSide ? k0 : k0-1;

	// tangential voltages: the sheet plane and the N-1 planes behind it (the outer PEC wall is never updated)
	m_voltRange[0][0] = 1;
	m_voltRange[0][1] = N;
	// normal voltages: the N edges behind the sheet
	m_voltRange[1][0] = posSide ? 0 : 1;
	m_voltRange[1][1] = posSide ? N-1 : N;
	// tangential currents: the N dual planes behind the sheet
	m_currRange[0][0] = posSide ? 0 : 1;
	m_currRange[0][1] = posSide ? N-1 : N;
	// normal currents: the sheet plane and the N-1 planes behind it
	m_currRange[1][0] = 1;
	m_currRange[1][1] = N;

	volt.Init("ipml_volt", m_numLines);
	volt_flux.Init("ipml_volt_flux", m_numLines);
	curr.Init("ipml_curr", m_numLines);
	curr_flux.Init("ipml_curr_flux", m_numLines);

	SetNumberOfThreads(1);
}

Engine_Ext_InvisiblePML::~Engine_Ext_InvisiblePML()
{
}

void Engine_Ext_InvisiblePML::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	m_numX = AssignJobs2Threads(m_numLines[m_nyP], m_NrThreads, false);
	m_start.resize(m_NrThreads,0);
	m_start.at(0)=0;
	for (size_t n=1; n<m_numX.size(); ++n)
		m_start.at(n) = m_start.at(n-1) + m_numX.at(n-1);
}

void Engine_Ext_InvisiblePML::UpdateVoltages(unsigned int start, unsigned int num)
{
	const Operator_Ext_InvisiblePML* op = m_Op_PML;
	unsigned int p[3], mP[3], mPP[3];
	FDTD_FLOAT f_help, flux;

	for (int n=0; n<3; ++n)
	{
		int nP  = (n+1)%3;
		int nPP = (n+2)%3;
		const unsigned int* range = m_voltRange[n==m_ny];
		for (p[m_nyP]=start; p[m_nyP]<start+num; ++p[m_nyP])
		{
			for (p[m_nyPP]=0; p[m_nyPP]<m_numLines[m_nyPP]; ++p[m_nyPP])
			{
				for (p[m_ny]=range[0]; p[m_ny]<=range[1]; ++p[m_ny])
				{
					// the same stencil (and lower-boundary shift) as Engine::UpdateVoltages
					for (int d=0; d<3; ++d)
						mP[d] = mPP[d] = p[d];
					mP[nP]   -= (p[nP]>0);
					mPP[nPP] -= (p[nPP]>0);

					// Engine_Ext_UPML::DoPreVoltageUpdates
					f_help = op->vv(n,p[0],p[1],p[2])   * volt(n,p[0],p[1],p[2])
						   - op->vvfo(n,p[0],p[1],p[2]) * volt_flux(n,p[0],p[1],p[2]);

					// Engine::UpdateVoltages, with the operator of Operator_Ext_UPML
					flux  = volt_flux(n,p[0],p[1],p[2]);
					flux *= op->vv_m(n,p[0],p[1],p[2]);
					flux += op->vi_m(n,p[0],p[1],p[2]) * (
								curr(nPP,p[0],p[1],p[2]) -
								curr(nPP,mP[0],mP[1],mP[2]) -
								curr(nP,p[0],p[1],p[2]) +
								curr(nP,mPP[0],mPP[1],mPP[2])
							);

					// Engine_Ext_UPML::DoPostVoltageUpdates
					volt_flux(n,p[0],p[1],p[2]) = flux;
					volt(n,p[0],p[1],p[2]) = f_help + op->vvfn(n,p[0],p[1],p[2]) * flux;
				}
			}
		}
	}
}

void Engine_Ext_InvisiblePML::UpdateCurrents(unsigned int start, unsigned int num)
{
	const Operator_Ext_InvisiblePML* op = m_Op_PML;
	unsigned int p[3], pP[3], pPP[3];
	FDTD_FLOAT f_help, flux;

	// as in the main engine, the currents on the last transverse lines are not updated
	unsigned int stopP = std::min(start+num, m_numLines[m_nyP]-1);

	for (int n=0; n<3; ++n)
	{
		int nP  = (n+1)%3;
		int nPP = (n+2)%3;
		const unsigned int* range = m_currRange[n==m_ny];
		for (p[m_nyP]=start; p[m_nyP]<stopP; ++p[m_nyP])
		{
			for (p[m_nyPP]=0; p[m_nyPP]<m_numLines[m_nyPP]-1; ++p[m_nyPP])
			{
				for (p[m_ny]=range[0]; p[m_ny]<=range[1]; ++p[m_ny])
				{
					for (int d=0; d<3; ++d)
						pP[d] = pPP[d] = p[d];
					++pP[nP];
					++pPP[nPP];

					// Engine_Ext_UPML::DoPreCurrentUpdates
					f_help = op->ii(n,p[0],p[1],p[2])   * curr(n,p[0],p[1],p[2])
						   - op->iifo(n,p[0],p[1],p[2]) * curr_flux(n,p[0],p[1],p[2]);

					// Engine::UpdateCurrents, with the operator of Operator_Ext_UPML
					flux  = curr_flux(n,p[0],p[1],p[2]);
					flux *= op->ii_m(n,p[0],p[1],p[2]);
					flux += op->iv_m(n,p[0],p[1],p[2]) * (
								volt(nPP,p[0],p[1],p[2]) -
								volt(nPP,pP[0],pP[1],pP[2]) -
								volt(nP,p[0],p[1],p[2]) +
								volt(nP,pPP[0],pPP[1],pPP[2])
							);

					// Engine_Ext_UPML::DoPostCurrentUpdates
					curr_flux(n,p[0],p[1],p[2]) = flux;
					curr(n,p[0],p[1],p[2]) = f_help + op->iifn(n,p[0],p[1],p[2]) * flux;
				}
			}
		}
	}
}

template <typename EngType>
void Engine_Ext_InvisiblePML::DoPreVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	// copy the main grid's tangential currents in front of the sheet into the ghost layer
	unsigned int loc[3], pos[3];
	loc[m_ny] = m_lineGhost;
	pos[m_ny] = m_mainGhost;
	for (loc[m_nyP]=m_start.at(threadID); loc[m_nyP]<m_start.at(threadID)+m_numX.at(threadID); ++loc[m_nyP])
	{
		pos[m_nyP] = loc[m_nyP] + m_offset[m_nyP];
		for (loc[m_nyPP]=0; loc[m_nyPP]<m_numLines[m_nyPP]; ++loc[m_nyPP])
		{
			pos[m_nyPP] = loc[m_nyPP] + m_offset[m_nyPP];
			curr(m_nyP ,loc[0],loc[1],loc[2]) = eng->EngType::GetCurr(m_nyP ,pos);
			curr(m_nyPP,loc[0],loc[1],loc[2]) = eng->EngType::GetCurr(m_nyPP,pos);
		}
	}
}

void Engine_Ext_InvisiblePML::DoPreVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_InvisiblePML::DoPostVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	UpdateVoltages(m_start.at(threadID), m_numX.at(threadID));

	// place the sheet plane into the main grid, where the PEC has zeroed it
	unsigned int loc[3], pos[3];
	loc[m_ny] = m_lineInt;
	pos[m_ny] = m_mainInt;
	for (loc[m_nyP]=m_start.at(threadID); loc[m_nyP]<m_start.at(threadID)+m_numX.at(threadID); ++loc[m_nyP])
	{
		pos[m_nyP] = loc[m_nyP] + m_offset[m_nyP];
		for (loc[m_nyPP]=0; loc[m_nyPP]<m_numLines[m_nyPP]; ++loc[m_nyPP])
		{
			pos[m_nyPP] = loc[m_nyPP] + m_offset[m_nyPP];
			eng->EngType::SetVolt(m_nyP ,pos, volt(m_nyP ,loc[0],loc[1],loc[2]));
			eng->EngType::SetVolt(m_nyPP,pos, volt(m_nyPP,loc[0],loc[1],loc[2]));
		}
	}
}

void Engine_Ext_InvisiblePML::DoPostVoltageUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_InvisiblePML::DoPreCurrentUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	// the main grid owns the final sheet-plane voltages (e.g. after other extensions)
	unsigned int loc[3], pos[3];
	loc[m_ny] = m_lineInt;
	pos[m_ny] = m_mainInt;
	for (loc[m_nyP]=m_start.at(threadID); loc[m_nyP]<m_start.at(threadID)+m_numX.at(threadID); ++loc[m_nyP])
	{
		pos[m_nyP] = loc[m_nyP] + m_offset[m_nyP];
		for (loc[m_nyPP]=0; loc[m_nyPP]<m_numLines[m_nyPP]; ++loc[m_nyPP])
		{
			pos[m_nyPP] = loc[m_nyPP] + m_offset[m_nyPP];
			volt(m_nyP ,loc[0],loc[1],loc[2]) = eng->EngType::GetVolt(m_nyP ,pos);
			volt(m_nyPP,loc[0],loc[1],loc[2]) = eng->EngType::GetVolt(m_nyPP,pos);
		}
	}
}

void Engine_Ext_InvisiblePML::DoPreCurrentUpdates(int threadID)
{
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_InvisiblePML::Apply2CurrentImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	UpdateCurrents(m_start.at(threadID), m_numX.at(threadID));

	// the normal current on the sheet plane is only read by the sheet-plane voltages,
	// but place it into the main grid anyway, so probes and dumps see it
	unsigned int loc[3], pos[3];
	unsigned int stopP = std::min(m_start.at(threadID)+m_numX.at(threadID), m_numLines[m_nyP]-1);
	loc[m_ny] = m_lineInt;
	pos[m_ny] = m_mainInt;
	for (loc[m_nyP]=m_start.at(threadID); loc[m_nyP]<stopP; ++loc[m_nyP])
	{
		pos[m_nyP] = loc[m_nyP] + m_offset[m_nyP];
		for (loc[m_nyPP]=0; loc[m_nyPP]<m_numLines[m_nyPP]-1; ++loc[m_nyPP])
		{
			pos[m_nyPP] = loc[m_nyPP] + m_offset[m_nyPP];
			eng->EngType::SetCurr(m_ny,pos, curr(m_ny,loc[0],loc[1],loc[2]));
		}
	}
}

void Engine_Ext_InvisiblePML::Apply2Current(int threadID)
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, threadID);
}
