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

#ifndef ENGINE_INTERFACE_FDTD_H
#define ENGINE_INTERFACE_FDTD_H

#include <cmath>
#include <vector>

#include "Common/engine_interface_base.h"
#include "operator.h"
#include "engine.h"

//! One term of a field-interpolation stencil: coeff * <volt|curr>[src].
//
// Every interpolated dump field is a fixed linear combination of the engine's
// raw edge/face values, with time-invariant weights (edge lengths, cell areas,
// eps/kappa/mue). BuildFieldStencil() emits that combination so it can be
// evaluated on the GPU as a simple gather -- reading the device field arrays
// directly, no host readback and no material arrays uploaded. `src` is the flat
// index into the engine's volt/curr array: (x*NY*NZ + y*NZ + z)*3 + comp.
struct FieldStencilEntry
{
	unsigned int src;
	float        coeff;
};

class Engine_Interface_FDTD : public Engine_Interface_Base
{
public:
	Engine_Interface_FDTD(Operator* op);
	virtual ~Engine_Interface_FDTD();

	//! Set the FDTD operator
	virtual void SetFDTDOperator(Operator* op) {SetOperator(op); m_Op=op;}
	//! Set the FDTD engine
	virtual void SetFDTDEngine(Engine* eng) {m_Eng=eng;}

	//! Get the FDTD engine in case direct access is needed. Direct access is not recommended!
	const Engine* GetFDTDEngine() const {return m_Eng;}
	//! Get the FDTD operator in case direct access is needed. Direct access is not recommended!
	const Operator* GetFDTDOperator() const {return m_Op;}

	virtual double* GetEField(const unsigned int* pos, double* out) const;
	virtual double* GetHField(const unsigned int* pos, double* out) const;
	virtual double* GetJField(const unsigned int* pos, double* out) const;
	virtual double* GetRotHField(const unsigned int* pos, double* out) const;
	virtual double* GetDField(const unsigned int* pos, double* out) const;
	virtual double* GetBField(const unsigned int* pos, double* out) const;

	virtual double CalcVoltageIntegral(const unsigned int* start, const unsigned int* stop) const;

	virtual double GetTime(bool dualTime=false) const {return ((double)m_Eng->GetNumberOfTimesteps() + (double)dualTime*0.5)*m_Op->GetTimestep();};
	virtual unsigned int GetNumberOfTimesteps() const {return m_Eng->GetNumberOfTimesteps();}

	virtual double CalcFastEnergy() const;

	//! Build the interpolation stencil at output position \p pos for a
	//! ProcessFields dump type (0:E 1:H 2:J 3:rotH 4:D 5:B). Emits one entry
	//! list per output component (each summing to that component's value) and
	//! sets \p useCurr (true -> entries index the curr array, else volt). The
	//! coefficients reproduce GetEField/GetHField/... exactly, so a GPU gather
	//! over these entries matches the host CalcField to fp32. Returns false if
	//! the dump type is unknown (caller must fall back to the host path).
	bool BuildFieldStencil(const unsigned int* pos, int dumpType,
	                       std::vector<FieldStencilEntry> out[3], bool& useCurr) const;

	//! Self-check: for \p nSamples random positions, verify the stencil
	//! evaluates (against the live engine values) to the same vector as the
	//! direct GetEField/... path. Returns the worst |difference|/|peak|.
	double VerifyFieldStencil(int dumpType, int nSamples) const;

protected:
	Operator* m_Op;
	Engine* m_Eng;

	//! Flat index into the engine volt/curr array for (comp n, position pos).
	inline unsigned int StencilIndex(unsigned int n, const unsigned int* pos) const
	{
		return (pos[0]*m_Op->GetNumberOfLines(1)*m_Op->GetNumberOfLines(2)
		        + pos[1]*m_Op->GetNumberOfLines(2) + pos[2])*3 + n;
	}
	//! Emit the raw (un-interpolated) primary-field stencil for component n.
	//! type: 0:E 1:J 2:rotH 3:D. Sets useCurr (rotH -> curr, else volt).
	void BuildRawFieldStencil(unsigned int n, const unsigned int* pos, int type,
	                          std::vector<FieldStencilEntry>& e, bool& useCurr) const;
	//! Emit the raw dual-field stencil for component n. type: 0:H 1:B (curr).
	void BuildRawDualFieldStencil(unsigned int n, const unsigned int* pos, int type,
	                              std::vector<FieldStencilEntry>& e) const;
	//! Interpolated primary/dual field stencils (mirror Get*InterpolatedField).
	void BuildInterpField(const unsigned int* pos, int type,
	                      std::vector<FieldStencilEntry> out[3]) const;
	void BuildInterpDualField(const unsigned int* pos, int type,
	                          std::vector<FieldStencilEntry> out[3]) const;

	//! Internal method to get an interpolated field of a given type. (0: E, 1: J, 2: rotH, 3: D)
	virtual double* GetRawInterpolatedField(const unsigned int* pos, double* out, int type) const;
	//! Internal method to get a raw field of a given type. (0: E, 1: J, 2: rotH, 3: D)
	virtual double GetRawField(unsigned int n, const unsigned int* pos, int type) const;

	//! Internal method to get an interpolated dual field of a given type. (0: H, 1: B)
	virtual double* GetRawInterpolatedDualField(const unsigned int* pos, double* out, int type) const;
	//! Internal method to get a raw dual field of a given type. (0: H, 1: B)
	virtual double GetRawDualField(unsigned int n, const unsigned int* pos, int type) const;
};

#endif // ENGINE_INTERFACE_FDTD_H
