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

/*
 * Invisible PML: a local, sheet-shaped termination for guided-wave ports.
 *
 * The sheet lies on the face of a PEC block (or a PEC domain boundary). An
 * N-cell UPML is extruded behind it, on the PEC side, but that PML does not
 * exist in the main mesh: its fields and coefficients live only in this
 * extension. Each timestep the extension updates the virtual PML and writes the
 * tangential E on the sheet plane (which the PEC would otherwise zero) back into
 * the main field domain.
 *
 * The update equations and grading are those of Operator_Ext_UPML. The
 * material of the virtual PML is the cross-section of the first cell in front
 * of the sheet, extruded; so a line or strip that runs into the sheet runs on
 * into the PML, exactly as a signal layer runs into a boundary PML.
 *
 * Local index along the sheet normal ny, for a virtual PML of N cells:
 *   domain on the positive side (NormalSignPositive):
 *     l = 0 is the outer PEC wall, l = N is the sheet plane; the main grid's
 *     tangential H at the sheet is copied into the ghost layer l = N (dual).
 *   domain on the negative side:
 *     l = 1 is the sheet plane, l = N+1 is the outer PEC wall; the main grid's
 *     tangential H in front of the sheet is copied into the ghost layer l = 0.
 */

#ifndef OPERATOR_EXT_INVISIBLE_PML_H
#define OPERATOR_EXT_INVISIBLE_PML_H

#include "FDTD/operator.h"
#include "operator_extension.h"
#include "tools/arraylib/array_nijk.h"

#include "CSPropAbsorbingBC.h"

class FunctionParser;

class Operator_Ext_InvisiblePML : public Operator_Extension
{
	friend class Engine_Ext_InvisiblePML;
public:
	Operator_Ext_InvisiblePML(Operator* op);
	virtual ~Operator_Ext_InvisiblePML();

	//! Take the sheet position, normal and PML depth from the absorbing BC property
	virtual bool SetInitParams(CSPrimitives* prim, CSPropAbsorbingBC* abc_prop);

	//! Set the grading function, same syntax and variables as Operator_Ext_UPML::SetGradingFunction
	virtual bool SetGradingFunction(std::string func);

	virtual bool BuildExtension();

	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const
	{
		return std::string("Invisible PML sheet");
	}

	virtual void ShowStat(std::ostream &ostr) const;

protected:
	//! Local index along ny of the sheet plane and of the ghost (dual) layer
	unsigned int InterfaceLine() const {return m_normalSignPositive ? m_numCells : 1;}
	unsigned int GhostLine() const     {return m_normalSignPositive ? m_numCells : 0;}

	//! Distance from the sheet, in cells, of a node plane / of the half plane l+1/2
	double NodeDepth(unsigned int l) const {return m_normalSignPositive ? double(m_numCells) - l : double(l) - 1.0;}
	double HalfDepth(unsigned int l) const {return m_normalSignPositive ? double(m_numCells) - l - 0.5 : double(l) - 0.5;}

	//! Grading conductivity at a depth (in cells) into the virtual PML
	double GradingKappa(double depthCells) const;

	int				m_ny, m_nyP, m_nyPP;
	bool			m_normalSignPositive;
	unsigned int	m_numCells;			// PML depth in cells
	unsigned int	m_sheetX0[3];		// sheet bounding box in the main mesh
	unsigned int	m_sheetX1[3];
	unsigned int	m_numLines[3];		// virtual domain size, local indices
	double			m_delta;			// cell size along ny, in meter

	std::string		m_GradFunc;
	FunctionParser*	m_GradingFunction;

	// Coefficients, same meaning as in Operator_Ext_UPML, where the "main" pair
	// (vv/vi, ii/iv) is what Operator_Ext_UPML writes into the main operator.
	ArrayLib::ArrayNIJK<FDTD_FLOAT> vv_m;	// main: new flux from old flux
	ArrayLib::ArrayNIJK<FDTD_FLOAT> vi_m;	// main: new flux from curl H
	ArrayLib::ArrayNIJK<FDTD_FLOAT> vv;		// new voltage from old voltage
	ArrayLib::ArrayNIJK<FDTD_FLOAT> vvfo;	// new voltage from old voltage flux
	ArrayLib::ArrayNIJK<FDTD_FLOAT> vvfn;	// new voltage from new voltage flux
	ArrayLib::ArrayNIJK<FDTD_FLOAT> ii_m;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> iv_m;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> ii;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> iifo;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> iifn;
};

#endif // OPERATOR_EXT_INVISIBLE_PML_H
