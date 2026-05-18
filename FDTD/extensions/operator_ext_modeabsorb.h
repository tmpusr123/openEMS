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

#ifndef OPERATOR_EXT_MODEABSORB_H
#define OPERATOR_EXT_MODEABSORB_H

#include <vector>

#include "FDTD/operator.h"
#include "operator_extension.h"

class CSPropModeAbsorb;
class CSPrimitives;

// ---------------------------------------------------------------------------
// File-based modal absorber. Drop-in twin of Operator_Ext_WaveguideAbsorber
// (matched-modal-source per Alimenti-Roselli-Mezzanotte 2000) but sourcing the
// mode shape from the E / H CSV files attached to the CSPropModeAbsorb
// property instead of from a CSPropExcitation's weight functions. Used by the
// CST-style "Tier 1" port-plane sheet stamped via CSX::AddModeAbsorb in
// compute/fdtd/ports.py and by the auto-wired one openEMS WaveguidePort places
// at the domain boundary.
// ---------------------------------------------------------------------------

class Operator_Ext_ModeAbsorb : public Operator_Extension
{
	friend class Engine_Ext_ModeAbsorb;
public:
	Operator_Ext_ModeAbsorb(Operator* op);
	virtual ~Operator_Ext_ModeAbsorb();

	// SetupModeAbsorbers() in openems.cpp constructs the operator extension,
	// then calls this with the property + primitive picked out of the CSX.
	bool SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop);

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();

	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const { return std::string("Mode Absorber Extension"); }

	virtual void ShowStat(std::ostream &ostr) const;

	// Engine-side accessor for underlying Operator (needed by Luo-Chen
	// branch to read dt and edge length).  Forwards inherited m_Op.
	Operator* GetOperator() const { return m_Op; }

protected:
	Operator_Ext_ModeAbsorb(Operator* op, Operator_Ext_ModeAbsorb* op_ext);
	void Initialize();

	CSPropModeAbsorb* m_prop;
	CSPrimitives*     m_prim;

	int m_ny, m_nyP, m_nyPP;   // propagation and transverse directions
	int m_dir;                 // +1 or -1 (propagation sign)

	// Sheet plane grid index (the user box is zero-thickness in m_ny)
	unsigned int m_srcLine;

	// Absorber plane grid indices (single-layer matched-modal sheet)
	unsigned int m_E_abs_line;
	unsigned int m_H_abs_line;

	// Cross-section bounds
	unsigned int m_start[3], m_stop[3];

	// Cross-section dimensions for E and H mode distributions
	unsigned int m_numLines_E[2];
	unsigned int m_numLines_H[2];

	double m_Zw;    // wave impedance (from mode solver, supplied by user)

	// Pre-computed weights (same pattern as Operator_Ext_WaveguideAbsorber)
	double** m_E_OverlapW[2];   // [comp][posP][posPP]  for E projection
	double** m_E_SubtractW[2];  //                       for E write-back

	double** m_H_OverlapW[2];   // dual-mesh H weights
	double** m_H_SubtractW[2];

	// State-space port operator (CST CBBPortOperatorStateSpace equivalent).
	// When m_UseStateSpace=true the engine uses these pre-computed CN-advance
	// matrices instead of the V/I matched-modal-source kernel.  See
	// CSPropModeAbsorb.h for the math.
	bool                m_UseStateSpace;
	unsigned int        m_SS_n;          // state-vector size
	std::vector<double> m_SS_P;          // n×n row-major: (I−0.5·dt·A)⁻¹·(I+0.5·dt·A)
	std::vector<double> m_SS_Q;          // n×1:           (I−0.5·dt·A)⁻¹·dt·B
	std::vector<double> m_SS_C;          // 1×n
	double              m_SS_D;
};

#endif // OPERATOR_EXT_MODEABSORB_H
