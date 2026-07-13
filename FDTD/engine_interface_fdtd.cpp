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

#include "engine_interface_fdtd.h"

Engine_Interface_FDTD::Engine_Interface_FDTD(Operator* op) : Engine_Interface_Base(op)
{
	if (op==NULL)
	{
		cerr << "Engine_Interface_FDTD::Engine_Interface_FDTD: Error: Operator is not set! Exit!" << endl;
		exit(1);
	}
	m_Op = op;
	m_Eng = m_Op->GetEngine();
	if (m_Eng==NULL)
	{
		cerr << "Engine_Interface_FDTD::Engine_Interface_FDTD: Error: Engine is not set! Exit!" << endl;
		exit(1);
	}
}

Engine_Interface_FDTD::~Engine_Interface_FDTD()
{
}

double* Engine_Interface_FDTD::GetEField(const unsigned int* pos, double* out) const
{
	return GetRawInterpolatedField(pos, out, 0);
}

double* Engine_Interface_FDTD::GetJField(const unsigned int* pos, double* out) const
{
	return GetRawInterpolatedField(pos, out, 1);
}

double* Engine_Interface_FDTD::GetDField(const unsigned int* pos, double* out) const
{
	return GetRawInterpolatedField(pos, out, 3);
}

double* Engine_Interface_FDTD::GetRotHField(const unsigned int* pos, double* out) const
{
	return GetRawInterpolatedField(pos, out, 2);
}

double* Engine_Interface_FDTD::GetRawInterpolatedField(const unsigned int* pos, double* out, int type) const
{
	unsigned int iPos[] = {pos[0],pos[1],pos[2]};
	int nP,nPP;
	double delta;
	switch (m_InterpolType)
	{
	default:
	case NO_INTERPOLATION:
		for (int n=0; n<3; ++n)
			out[n] = GetRawField(n,pos,type);
		break;
	case NODE_INTERPOLATE:
		for (int n=0; n<3; ++n)
		{
			if (pos[n]==m_Op->GetNumberOfLines(n, true)-1)  // use only the "lower value" at the upper bound
			{
				--iPos[n];
				out[n] = (double)GetRawField(n,iPos,type);
				++iPos[n];
				continue;
			}
			delta = m_Op->GetEdgeLength(n,iPos);
			out[n] = GetRawField(n,iPos,type);
			if (delta==0)
			{
				out[n]=0;
				continue;
			}
			if (pos[n]==0) // use only the "upper value" at the lower bound
				continue;
			--iPos[n];
			double deltaDown = m_Op->GetEdgeLength(n,iPos);
			double deltaRel = delta / (delta+deltaDown);
			out[n] = out[n]*(1.0-deltaRel) + (double)GetRawField(n,iPos,type)*deltaRel;
			++iPos[n];
		}
		break;
	case CELL_INTERPOLATE:
		for (int n=0; n<3; ++n)
		{
			nP = (n+1)%3;
			nPP = (n+2)%3;
			if ((pos[0]==m_Op->GetNumberOfLines(0,true)-1) || (pos[1]==m_Op->GetNumberOfLines(1,true)-1) || (pos[2]==m_Op->GetNumberOfLines(2,true)-1))
			{
				out[n] = 0; //electric field outside the field domain is always zero
				continue;
			}
			out[n]=GetRawField(n,iPos,type);
			++iPos[nP];
			out[n]+=GetRawField(n,iPos,type);
			++iPos[nPP];
			out[n]+=GetRawField(n,iPos,type);
			--iPos[nP];
			out[n]+=GetRawField(n,iPos,type);
			--iPos[nPP];
			out[n]/=4;
		}
		break;
	}
	return out;
}

double* Engine_Interface_FDTD::GetHField(const unsigned int* pos, double* out) const
{
		return GetRawInterpolatedDualField(pos, out, 0);
}

double* Engine_Interface_FDTD::GetBField(const unsigned int* pos, double* out) const
{
		return GetRawInterpolatedDualField(pos, out, 1);
}

double Engine_Interface_FDTD::GetRawDualField(unsigned int n, const unsigned int* pos, int type) const
{
	double value = m_Eng->GetCurr(n,pos[0],pos[1],pos[2]);
	double delta = m_Op->GetEdgeLength(n,pos,true);
	if ((type==0) && (delta))
		return value/delta;
	if ((type==1) && (m_Op->m_mueR_ptr) && (delta))
	{
		ArrayLib::ArrayNIJK<float>& m_mueR = *m_Op->m_mueR_ptr;
		return value * m_mueR[n][pos[0]][pos[1]][pos[2]] / delta;
	}
	return 0.0;
}

double* Engine_Interface_FDTD::GetRawInterpolatedDualField(const unsigned int* pos, double* out, int type) const
{
	unsigned int iPos[] = {pos[0],pos[1],pos[2]};
	int nP,nPP;
	double delta;
	switch (m_InterpolType)
	{
	default:
	case NO_INTERPOLATION:
		out[0] = GetRawDualField(0, pos, type);
		out[1] = GetRawDualField(1, pos, type);
		out[2] = GetRawDualField(2, pos, type);
		break;
	case NODE_INTERPOLATE:
		for (int n=0; n<3; ++n)
		{
			nP = (n+1)%3;
			nPP = (n+2)%3;
			if ((pos[0]==m_Op->GetNumberOfLines(0,true)-1) || (pos[1]==m_Op->GetNumberOfLines(1,true)-1) || (pos[2]==m_Op->GetNumberOfLines(2,true)-1) || (pos[nP]==0) || (pos[nPP]==0))
			{
				out[n] = 0;
				continue;
			}
			out[n] = GetRawDualField(n, iPos, type);
			--iPos[nP];
			out[n]+= GetRawDualField(n, iPos, type);
			--iPos[nPP];
			out[n]+= GetRawDualField(n, iPos, type);
			++iPos[nP];
			out[n]+= GetRawDualField(n, iPos, type);
			++iPos[nPP];
			out[n]/=4;
		}
		break;
	case CELL_INTERPOLATE:
		for (int n=0; n<3; ++n)
		{
			delta = m_Op->GetEdgeLength(n,iPos,true);
			out[n] = GetRawDualField(n, iPos, type);
			if ((pos[n]>=m_Op->GetNumberOfLines(n,true)-1))
			{
				out[n] = 0; //magnetic field on the outer boundaries is always zero
				continue;
			}
			++iPos[n];
			double deltaUp = m_Op->GetEdgeLength(n,iPos,true);
			double deltaRel = delta / (delta+deltaUp);
			out[n] = out[n]*(1.0-deltaRel) + (double)GetRawDualField(n, iPos, type)*deltaRel;
			--iPos[n];
		}
		break;
	}

	return out;
}

double Engine_Interface_FDTD::CalcVoltageIntegral(const unsigned int* start, const unsigned int* stop) const
{
	if (((start[0]!=stop[0]) + (start[1]!=stop[1]) + (start[2]!=stop[2]))!=1)
	{
		cerr << "Engine_Interface_FDTD::CalcVoltageIntegral: Error, only a 1D/line integration is allowed" << endl;
		return 0;
	}
	//cerr << "CalcVoltageIntegral" << start[0] << ", " << start[1] << ", " << start[2] << " -> " << stop[0] << ", " << stop[1] << ", " << stop[2] << ", " << endl;
	double result=0;
	for (int n=0; n<3; ++n)
	{
		if (start[n]<stop[n])
		{
			unsigned int pos[3]={start[0],start[1],start[2]};
			for (; pos[n]<stop[n]; ++pos[n])
				result += m_Eng->GetVolt(n,pos[0],pos[1],pos[2]);

		}
		else
		{
			unsigned int pos[3]={stop[0],stop[1],stop[2]};
			for (; pos[n]<start[n]; ++pos[n])
				result -= m_Eng->GetVolt(n,pos[0],pos[1],pos[2]);
		}
	}
	return result;
}

//double Engine_Interface_FDTD::CalcVoltageIntegral(const unsigned int* start, const unsigned int* stop) const
//{
//	//cerr << "CalcVoltageIntegral" << start[0] << ", " << start[1] << ", " << start[2] << " -> " << stop[0] << ", " << stop[1] << ", " << stop[2] << ", " << endl;
//	double result=0;
//	//unsigned int pos[3]={min(start[0],stop[0]),min(start[1],stop[1]),min(start[2],stop[2])};
//	unsigned int pos[3]={start[0],start[1],start[2]};
//	for (int n=0; n<3; ++n)
//	{
//		if (start[n]<stop[n])
//		{
//			for (; pos[n]<stop[n]; ++pos[n])
//			{
//				//cerr << "at pos: " << n << ": " << pos[0] << ", " << pos[1] << ", " << pos[2] << endl;
//				result += m_Eng->GetVolt(n,pos[0],pos[1],pos[2]);
//			}
//		}
//		else
//		{
//			for (--pos[n]; pos[n]>=stop[n]; --pos[n])
//			{
//				//cerr << "at neg: " << n << ": " << pos[0] << ", " << pos[1] << ", " << pos[2] << endl;
//				result -= m_Eng->GetVolt(n,pos[0],pos[1],pos[2]);
//			}
//		}
//		pos[n] = stop[n];
//	}
//	return result;
//}

double Engine_Interface_FDTD::GetRawField(unsigned int n, const unsigned int* pos, int type) const
{
	double value = m_Eng->GetVolt(n,pos[0],pos[1],pos[2]);
	double delta = m_Op->GetEdgeLength(n,pos);
	if ((type==0) && (delta))
		return value/delta;
	if ((type==1) && (m_Op->m_kappa_ptr) && (delta)) {
		ArrayLib::ArrayNIJK<float>& kappa = *m_Op->m_kappa_ptr;
		return value * kappa[n][pos[0]][pos[1]][pos[2]] / delta;
	}
	if ((type==3) && (m_Op->m_epsR_ptr) && (delta)) {
		ArrayLib::ArrayNIJK<float>& epsR = *m_Op->m_epsR_ptr;
		return value * epsR[n][pos[0]][pos[1]][pos[2]] / delta;
	}
	if (type==2) //calc rot(H)
	{
		int nP = (n+1)%3;
		int nPP = (n+2)%3;
		unsigned int locPos[] = {pos[0],pos[1],pos[2]};
		double area = m_Op->GetEdgeArea(n,pos);
		value  = m_Eng->GetCurr(nPP,pos);
		value -= m_Eng->GetCurr(nP,pos);
		if (pos[nPP]>0)
		{
			--locPos[nPP];
			value += m_Eng->GetCurr(nP,locPos);
			++locPos[nPP];
		}
		if (pos[nP]>0)
		{
			--locPos[nP];
			value -= m_Eng->GetCurr(nPP,locPos);
		}
		return value/area;
	}

	return 0.0;
}

// ---------------------------------------------------------------------------
// Interpolation stencils for GPU-side field extraction.
//
// These mirror GetRawField / GetRawInterpolatedField / *DualField above term
// for term, but emit (index, coeff) pairs instead of reading live values. The
// weight math (GetEdgeLength/GetEdgeArea/kappa/epsR/mueR) is identical, so a
// gather over the emitted entries reproduces the direct path to fp32. Keep in
// lockstep with the functions above if they ever change.
// ---------------------------------------------------------------------------

// raw primary field, type 0:E 1:J 2:rotH 3:D  (E/J/D -> volt, rotH -> curr)
void Engine_Interface_FDTD::BuildRawFieldStencil(unsigned int n, const unsigned int* pos, int type,
                                                 std::vector<FieldStencilEntry>& e, bool& useCurr) const
{
	e.clear();
	if (type==2) // rot(H): curl of the surrounding current faces / cell area
	{
		useCurr = true;
		double area = m_Op->GetEdgeArea(n,pos);
		if (area==0) return;
		double inv = 1.0/area;
		int nP = (n+1)%3, nPP = (n+2)%3;
		unsigned int p[3] = {pos[0],pos[1],pos[2]};
		e.push_back({StencilIndex(nPP,p), (float)( inv)});
		e.push_back({StencilIndex(nP ,p), (float)(-inv)});
		if (pos[nPP]>0) { --p[nPP]; e.push_back({StencilIndex(nP ,p), (float)( inv)}); ++p[nPP]; }
		if (pos[nP] >0) { --p[nP];  e.push_back({StencilIndex(nPP,p), (float)(-inv)}); }
		return;
	}
	useCurr = false;
	double delta = m_Op->GetEdgeLength(n,pos);
	if (delta==0) return;
	if (type==0)
		e.push_back({StencilIndex(n,pos), (float)(1.0/delta)});
	else if (type==1 && m_Op->m_kappa_ptr)
	{
		ArrayLib::ArrayNIJK<float>& kappa = *m_Op->m_kappa_ptr;
		e.push_back({StencilIndex(n,pos), (float)(kappa[n][pos[0]][pos[1]][pos[2]]/delta)});
	}
	else if (type==3 && m_Op->m_epsR_ptr)
	{
		ArrayLib::ArrayNIJK<float>& epsR = *m_Op->m_epsR_ptr;
		e.push_back({StencilIndex(n,pos), (float)(epsR[n][pos[0]][pos[1]][pos[2]]/delta)});
	}
}

// raw dual field, type 0:H 1:B  (both -> curr)
void Engine_Interface_FDTD::BuildRawDualFieldStencil(unsigned int n, const unsigned int* pos, int type,
                                                     std::vector<FieldStencilEntry>& e) const
{
	e.clear();
	double delta = m_Op->GetEdgeLength(n,pos,true);
	if (delta==0) return;
	if (type==0)
		e.push_back({StencilIndex(n,pos), (float)(1.0/delta)});
	else if (type==1 && m_Op->m_mueR_ptr)
	{
		ArrayLib::ArrayNIJK<float>& mueR = *m_Op->m_mueR_ptr;
		e.push_back({StencilIndex(n,pos), (float)(mueR[n][pos[0]][pos[1]][pos[2]]/delta)});
	}
}

static inline void _appendScaled(std::vector<FieldStencilEntry>& dst,
                                 const std::vector<FieldStencilEntry>& src, double scale)
{
	for (size_t i=0;i<src.size();++i)
		dst.push_back({src[i].src, (float)(src[i].coeff*scale)});
}

// mirror GetRawInterpolatedField (type 0:E 1:J 2:rotH 3:D)
void Engine_Interface_FDTD::BuildInterpField(const unsigned int* pos, int type,
                                             std::vector<FieldStencilEntry> out[3]) const
{
	std::vector<FieldStencilEntry> raw;
	bool uc;
	unsigned int iPos[3] = {pos[0],pos[1],pos[2]};
	int nP,nPP;
	double delta;
	for (int n=0;n<3;++n) { out[n].clear(); iPos[0]=pos[0]; iPos[1]=pos[1]; iPos[2]=pos[2]; }
	switch (m_InterpolType)
	{
	default:
	case NO_INTERPOLATION:
		for (int n=0;n<3;++n) { BuildRawFieldStencil(n,pos,type,raw,uc); _appendScaled(out[n],raw,1.0); }
		break;
	case NODE_INTERPOLATE:
		for (int n=0;n<3;++n)
		{
			iPos[0]=pos[0]; iPos[1]=pos[1]; iPos[2]=pos[2];
			if (pos[n]==m_Op->GetNumberOfLines(n,true)-1)
			{
				--iPos[n];
				BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,1.0);
				continue;
			}
			delta = m_Op->GetEdgeLength(n,iPos);
			if (delta==0) continue;                 // out[n]=0
			if (pos[n]==0)
			{
				BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,1.0);
				continue;
			}
			--iPos[n];
			double deltaDown = m_Op->GetEdgeLength(n,iPos);
			double deltaRel = delta/(delta+deltaDown);
			// out[n] = raw(pos)*(1-deltaRel) + raw(pos-e_n)*deltaRel
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,deltaRel);
			++iPos[n];
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,1.0-deltaRel);
		}
		break;
	case CELL_INTERPOLATE:
		for (int n=0;n<3;++n)
		{
			nP=(n+1)%3; nPP=(n+2)%3;
			if ((pos[0]==m_Op->GetNumberOfLines(0,true)-1) || (pos[1]==m_Op->GetNumberOfLines(1,true)-1) || (pos[2]==m_Op->GetNumberOfLines(2,true)-1))
				continue;                            // out[n]=0
			iPos[0]=pos[0]; iPos[1]=pos[1]; iPos[2]=pos[2];
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,0.25);
			++iPos[nP];
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,0.25);
			++iPos[nPP];
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,0.25);
			--iPos[nP];
			BuildRawFieldStencil(n,iPos,type,raw,uc); _appendScaled(out[n],raw,0.25);
		}
		break;
	}
}

// mirror GetRawInterpolatedDualField (type 0:H 1:B)
void Engine_Interface_FDTD::BuildInterpDualField(const unsigned int* pos, int type,
                                                 std::vector<FieldStencilEntry> out[3]) const
{
	std::vector<FieldStencilEntry> raw;
	unsigned int iPos[3];
	int nP,nPP;
	double delta;
	for (int n=0;n<3;++n) out[n].clear();
	switch (m_InterpolType)
	{
	default:
	case NO_INTERPOLATION:
		for (int n=0;n<3;++n) { BuildRawDualFieldStencil(n,pos,type,raw); _appendScaled(out[n],raw,1.0); }
		break;
	case NODE_INTERPOLATE:
		for (int n=0;n<3;++n)
		{
			nP=(n+1)%3; nPP=(n+2)%3;
			if ((pos[0]==m_Op->GetNumberOfLines(0,true)-1) || (pos[1]==m_Op->GetNumberOfLines(1,true)-1) || (pos[2]==m_Op->GetNumberOfLines(2,true)-1) || (pos[nP]==0) || (pos[nPP]==0))
				continue;                            // out[n]=0
			iPos[0]=pos[0]; iPos[1]=pos[1]; iPos[2]=pos[2];
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,0.25);
			--iPos[nP];
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,0.25);
			--iPos[nPP];
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,0.25);
			++iPos[nP];
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,0.25);
		}
		break;
	case CELL_INTERPOLATE:
		for (int n=0;n<3;++n)
		{
			iPos[0]=pos[0]; iPos[1]=pos[1]; iPos[2]=pos[2];
			delta = m_Op->GetEdgeLength(n,iPos,true);
			if (pos[n]>=m_Op->GetNumberOfLines(n,true)-1)
				continue;                            // out[n]=0
			++iPos[n];
			double deltaUp = m_Op->GetEdgeLength(n,iPos,true);
			double deltaRel = delta/(delta+deltaUp);
			--iPos[n];
			// out[n] = rawDual(pos)*(1-deltaRel) + rawDual(pos+e_n)*deltaRel
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,1.0-deltaRel);
			++iPos[n];
			BuildRawDualFieldStencil(n,iPos,type,raw); _appendScaled(out[n],raw,deltaRel);
		}
		break;
	}
}

bool Engine_Interface_FDTD::BuildFieldStencil(const unsigned int* pos, int dumpType,
                                              std::vector<FieldStencilEntry> out[3], bool& useCurr) const
{
	switch (dumpType)
	{
	case 0: BuildInterpField(pos,0,out); useCurr=false; return true;  // E
	case 2: BuildInterpField(pos,1,out); useCurr=false; return true;  // J (kappa*E)
	case 4: BuildInterpField(pos,3,out); useCurr=false; return true;  // D (eps*E)
	case 3: BuildInterpField(pos,2,out); useCurr=true;  return true;  // rotH (curl of curr)
	case 1: BuildInterpDualField(pos,0,out); useCurr=true; return true; // H
	case 5: BuildInterpDualField(pos,1,out); useCurr=true; return true; // B (mue*H)
	default: return false;
	}
}

// Evaluate the stencil against the live engine values and compare to the direct
// Get*Field path at nSamples random positions. Returns worst |diff|/|peak|.
double Engine_Interface_FDTD::VerifyFieldStencil(int dumpType, int nSamples) const
{
	unsigned int N[3] = {m_Op->GetNumberOfLines(0,true), m_Op->GetNumberOfLines(1,true), m_Op->GetNumberOfLines(2,true)};
	std::vector<FieldStencilEntry> st[3];
	bool useCurr;
	double worst = 0.0, peak = 1e-30;
	unsigned int seed = 12345u;
	for (int s=0;s<nSamples;++s)
	{
		unsigned int pos[3];
		for (int a=0;a<3;++a) { seed = seed*1664525u+1013904223u; pos[a] = seed % N[a]; }
		double ref[3];
		switch (dumpType)
		{
		case 0: GetEField(pos,ref); break;
		case 1: GetHField(pos,ref); break;
		case 2: GetJField(pos,ref); break;
		case 3: GetRotHField(pos,ref); break;
		case 4: GetDField(pos,ref); break;
		case 5: GetBField(pos,ref); break;
		default: return -1;
		}
		if (!BuildFieldStencil(pos,dumpType,st,useCurr)) return -1;
		for (int n=0;n<3;++n)
		{
			double v = 0.0;
			for (size_t i=0;i<st[n].size();++i)
			{
				unsigned int idx = st[n][i].src;
				unsigned int comp = idx%3, flat = idx/3;
				unsigned int z = flat % N[2], y = (flat/N[2]) % N[1], x = flat/(N[2]*N[1]);
				double fv = useCurr ? m_Eng->GetCurr(comp,x,y,z) : m_Eng->GetVolt(comp,x,y,z);
				v += (double)st[n][i].coeff * fv;
			}
			peak = std::max(peak, std::fabs(ref[n]));
			worst = std::max(worst, std::fabs(v-ref[n]));
		}
	}
	return worst/peak;
}

double Engine_Interface_FDTD::CalcFastEnergy() const
{
	double E_energy=0.0;
	double H_energy=0.0;

	unsigned int pos[3];
	if (m_Eng->GetType()==Engine::BASIC)
	{
		for (pos[0]=0; pos[0]<m_Op->GetNumberOfLines(0)-1; ++pos[0])
		{
			for (pos[1]=0; pos[1]<m_Op->GetNumberOfLines(1)-1; ++pos[1])
			{
				for (pos[2]=0; pos[2]<m_Op->GetNumberOfLines(2)-1; ++pos[2])
				{
					E_energy+=m_Eng->Engine::GetVolt(0,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(0,pos[0],pos[1],pos[2]);
					E_energy+=m_Eng->Engine::GetVolt(1,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(1,pos[0],pos[1],pos[2]);
					E_energy+=m_Eng->Engine::GetVolt(2,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(2,pos[0],pos[1],pos[2]);

					H_energy+=m_Eng->Engine::GetCurr(0,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(0,pos[0],pos[1],pos[2]);
					H_energy+=m_Eng->Engine::GetCurr(1,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(1,pos[0],pos[1],pos[2]);
					H_energy+=m_Eng->Engine::GetCurr(2,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(2,pos[0],pos[1],pos[2]);
				}
			}
		}
	}
	else
	{
		for (pos[0]=0; pos[0]<m_Op->GetNumberOfLines(0)-1; ++pos[0])
		{
			for (pos[1]=0; pos[1]<m_Op->GetNumberOfLines(1)-1; ++pos[1])
			{
				for (pos[2]=0; pos[2]<m_Op->GetNumberOfLines(2)-1; ++pos[2])
				{
					E_energy+=m_Eng->GetVolt(0,pos[0],pos[1],pos[2]) * m_Eng->GetVolt(0,pos[0],pos[1],pos[2]);
					E_energy+=m_Eng->GetVolt(1,pos[0],pos[1],pos[2]) * m_Eng->GetVolt(1,pos[0],pos[1],pos[2]);
					E_energy+=m_Eng->GetVolt(2,pos[0],pos[1],pos[2]) * m_Eng->GetVolt(2,pos[0],pos[1],pos[2]);

					H_energy+=m_Eng->GetCurr(0,pos[0],pos[1],pos[2]) * m_Eng->GetCurr(0,pos[0],pos[1],pos[2]);
					H_energy+=m_Eng->GetCurr(1,pos[0],pos[1],pos[2]) * m_Eng->GetCurr(1,pos[0],pos[1],pos[2]);
					H_energy+=m_Eng->GetCurr(2,pos[0],pos[1],pos[2]) * m_Eng->GetCurr(2,pos[0],pos[1],pos[2]);
				}
			}
		}
	}
	return __EPS0__*E_energy + __MUE0__*H_energy;
}
