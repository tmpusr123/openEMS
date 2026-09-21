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

#ifndef OPERATOR_EXT_LOSSYMETAL_H
#define OPERATOR_EXT_LOSSYMETAL_H

#include "FDTD/operator.h"
#include "operator_ext_lorentzmaterial.h"

/*!
  FDTD extension for a lossy metal (3D conductor volume) using Surface
  Impedance Boundary Condition (SIBC).

  The interior edges are handled as PEC (by CalcPEC via the METAL flag).
  This extension detects surface edges and applies SIBC to model
  frequency-dependent conductor loss.

  Phase 1: Constant surface resistance R_s = sqrt(omega_c * mu_0 / (2*sigma))
  Phase 2 (future): ADE-based frequency-dependent SIBC with rational approx.
  */
class Operator_Ext_LossyMetal : public Operator_Ext_LorentzMaterial
{
public:
	Operator_Ext_LossyMetal(Operator* op, double f_max);

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();

	virtual bool IsCylinderCoordsSave(bool closedAlpha, bool R0_included) const {UNUSED(closedAlpha); UNUSED(R0_included); return true;}
	virtual bool IsCylindricalMultiGridSave(bool child) const {UNUSED(child); return true;}
	virtual bool IsMPISave() const {return true;}

	virtual std::string GetExtensionName() const
	{
		return std::string("Lossy Metal (SIBC) Extension");
	}

protected:
	//! Copy constructor
	Operator_Ext_LossyMetal(Operator* op, Operator_Ext_LossyMetal* op_ext);
	double m_f_max;

	//! Check if an edge is on the surface of a LossyMetal primitive.
	//! Returns the number of outside faces (0 = interior, >0 = surface).
	//! surface_normal_dir is set to the perpendicular direction with an outside neighbor.
	//! surface_width returns the effective surface width for SIBC conductance.
	int GetSurfaceInfo(int n, unsigned int pos[3], CSPrimitives* prim,
	                   double &surface_width);
};

#endif // OPERATOR_EXT_LOSSYMETAL_H
