/*
*	Copyright (C) 2023-2025 Gadi Lahav (gadi@rfwithcare.com), Thorsten Liebig (Thorsten.Liebig@gmx.de)
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
 * The localized boundary conditions currently implemented are denoted:
 * - First order Mur BC, single phased velocity
 * - First order Mur BC with super absorption, single phase velocity.
 * These were chosen after experimentation with the various boundary
 * conditions suggested in [1]. The amount of post processing and
 * absorption performance pointed towards the Single PV 1st order Mur.
 * Since this method also requires an extra mesh cell in the doublet grid,
 * the option to use the regular 1st order Mur was left available.
 *
 * References:
 * [1] Betz, Vaughn Timothy, and R. Mittra. "Absorbing boundary conditions for the finite-difference time-domain analysis of guided-wave structures." Coordinated Science Laboratory Report no. UILU-ENG-93-2243 (1993).‏
 */


#ifndef OPERATOR_EXT_ABSORBING_BC_H
#define OPERATOR_EXT_ABSORBING_BC_H

#include <vector>

#include "FDTD/operator.h"
#include "operator_extension.h"
#include "tools/arraylib/array_ij.h"
#include "tools/arraylib/array_ijk.h"

#include "CSPropAbsorbingBC.h"

class Operator_Ext_Absorbing_BC : public Operator_Extension
{
	friend class Engine_Ext_Absorbing_BC;
public:

	// This should be a replica of the CSXCAD property, but can also be something
	// else, later
	enum ABCtype
	{
		UNDEFINED	= 0,
		MUR_1ST 	= 1,	// Mur's BC, 1st order
		MUR_1ST_SA 	= 2,	// Mur's BC, 1st order, with Super Absorption
		CPML		= 3		// Convolution PML strip behind the sheet (CFS, kappa=1 simplification)
	};

	Operator_Ext_Absorbing_BC(Operator* op);
	~Operator_Ext_Absorbing_BC();

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();

	virtual Engine_Extension* CreateEngineExtention();

	//virtual bool IsMPISave() const {return true;}

	virtual std::string GetExtensionName() const
	{
		return std::string("Local absorbing boundary condition sheet");
	}

	virtual void ShowStat(std::ostream &ostr) const;

	//! Initialize all parameters, so the extension can be built later
	virtual bool SetInitParams(CSPrimitives* prim, CSPropAbsorbingBC* abc_prop);

protected:

	Operator_Ext_Absorbing_BC(Operator* op, Operator_Ext_Absorbing_BC* op_ext);
	void Initialize();

	unsigned int			m_numCells;		// Number of cells in each primitive

	// Storage for the directions. ny is the normal direction to the sheet
	int				m_ny,
					m_nyP,
					m_nyPP;

	// Storage for sheet bounding box start and stop.
	unsigned int	m_sheetX0[3];
	unsigned int	m_sheetX1[3];

	unsigned int 	m_numLines[2];

	bool 			m_normalSignPositive;


	ABCtype			m_ABCtype;

	double			m_phaseVelocity;

	// Coefficients, to be initialized on-demand.
	ArrayLib::ArrayIJ<FDTD_FLOAT>	m_K1_nyP;
	ArrayLib::ArrayIJ<FDTD_FLOAT>	m_K1_nyPP;
	ArrayLib::ArrayIJ<FDTD_FLOAT> 	m_K2_nyP;
	ArrayLib::ArrayIJ<FDTD_FLOAT>	m_K2_nyPP;

	// ---- CPML members ----
	// User-supplied parameters
	unsigned int        m_pmlDepth;          // PML strip thickness (cells along m_ny)
	double              m_pmlSigmaMax;       // 0 = auto from R(0)=1e-6
	double              m_pmlAlphaMax;       // 0 = auto
	double              m_pmlKappaMax;       // coordinate stretch peak (>= 1)
	unsigned int        m_pmlProfileOrder;   // polynomial order p

	// Per-depth-cell decay factor exp(-(sigma+alpha)*dt/eps0). 1D, size = m_pmlDepth.
	std::vector<double> m_pml_b_z;

	// Pre-baked per-cell convolution coefficients used by the engine. They fold
	// the signed engine sign, c_z[k] = sigma/(sigma+alpha)*(b-1), and the local
	// vi (or iv) coefficient at that cell into a single multiplier so the engine
	// hook is just:  psi[k] = b_z[k]*psi + cv[i,j,k]*diff;  V += psi.
	// Indexed as (i_along_nyP, j_along_nyPP, k_depth).
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_cv_nyP;    // for V[m_nyP] update — Psi term
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_cv_nyPP;   // for V[m_nyPP] update — Psi term
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_ci_nyP;    // for I[m_nyP] update — Psi term
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_ci_nyPP;   // for I[m_nyPP] update — Psi term

	// kappa-stretch correction: sign * vi * (1/kappa - 1). Applied per-step
	// on the m_ny-axis curl difference so the engine's full-curl update is
	// effectively divided by kappa only on the m_ny axis.
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_kapV_nyP;
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_kapV_nyPP;
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_kapI_nyP;
	ArrayLib::ArrayIJK<FDTD_FLOAT> m_pml_kapI_nyPP;

	// Sign of the inward step from the sheet into the PML strip (+1 or -1).
	// PML extends opposite to where the wave lives.
	int                 m_pmlStepSign;
};

#endif // OPERATOR_EXT_ABSORBING_BC_H
