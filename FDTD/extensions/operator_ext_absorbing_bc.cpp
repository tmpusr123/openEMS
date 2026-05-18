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

#include "operator_ext_absorbing_bc.h"
#include "engine_ext_absorbing_bc.h"

#include "tools/array_ops.h"
#include "tools/constants.h"

#include "CSPrimBox.h"

#include <cmath>
#include <algorithm>
#include <limits>

using std::cerr;
using std::endl;

namespace
{
	// Default CFS-CPML parameters used when the user leaves them at 0.
	// 4-cell depth target with R(0) ~ 1e-6 normal-incidence reflection.
	constexpr double  CPML_DEFAULT_R0      = 1e-6;
	constexpr double  CPML_DEFAULT_ALPHA_MAX = 0.05;   // [S/m] gives alpha/eps0 ~ 5.6 GHz
}

Operator_Ext_Absorbing_BC::Operator_Ext_Absorbing_BC(Operator* op) : Operator_Extension(op)
{
	Initialize();
}

Operator_Ext_Absorbing_BC::~Operator_Ext_Absorbing_BC()
{
}

Operator_Ext_Absorbing_BC::Operator_Ext_Absorbing_BC(Operator* op, Operator_Ext_Absorbing_BC* op_ext) : Operator_Extension(op, op_ext)
{
	Initialize();
}

Operator_Extension* Operator_Ext_Absorbing_BC::Clone(Operator* op)
{
	return new Operator_Ext_Absorbing_BC(op, this);
}

void Operator_Ext_Absorbing_BC::Initialize()
{
	m_numCells = 0;

	m_ny = m_nyP = m_nyPP = -1;

	for (unsigned int dirIdx = 0 ; dirIdx < 3 ; m_sheetX0[dirIdx++] = 0);
	for (unsigned int dirIdx = 0 ; dirIdx < 3 ; m_sheetX1[dirIdx++] = 0);

	m_normalSignPositive = true;

	m_ABCtype = ABCtype::UNDEFINED;

	m_phaseVelocity = 0.0;

	m_pmlDepth         = 4;
	m_pmlSigmaMax      = 0.0;
	m_pmlAlphaMax      = 0.0;
	// Bérenger 2002 ("Application of the CFS PML to the Absorption of
	// Evanescent Waves in Waveguides") assumes κ=1 throughout the
	// derivation of α_opt = nπcε₀/a.  κ>1 over-stretches the coordinate
	// and damps the propagating component before it can be drained by σ,
	// hurting the evanescent absorption.  Match the paper.
	// Override via CPMLKappaMax envelope (or future setter) if you want
	// to revisit the Wang-2022 / Roden-Gedney recommendation of κ=4–5.
	m_pmlKappaMax      = 1.0;
	m_pmlProfileOrder  = 3;
	m_pmlStepSign      = +1;
}

bool Operator_Ext_Absorbing_BC::SetInitParams(CSPrimitives* prim, CSPropAbsorbingBC* abc_prop)
{
	CSPrimBox* cSheet = dynamic_cast<CSPrimBox*>(prim);
	// If this just so happen to not be a sheet, ignore this
	if (!cSheet)
	{
		cerr << "Operator_Ext_Absorbing_BC::SetInitParams(): Warning: Absorbing sheet validation failed, skipping. "
									<< " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	// Check that this is actually a sheet
	// Get box start and stop positions


	// snap to the native coordinate system
	int Snap_Dimension =
		m_Op->SnapBox2Mesh(
				cSheet->GetStartCoord()->GetCoords(m_Op->m_MeshType),	// Start Coord
				cSheet->GetStopCoord()->GetCoords(m_Op->m_MeshType),	// Stop Coord
				m_sheetX0,	// Start Index
				m_sheetX1,	// Stop Index
				false,		// Dual H-mesh
				true);		// Full mesh?

	// Verify that snapped dimension is correct
	if (Snap_Dimension <= 0)
	{
		if (Snap_Dimension >= -1)
			cerr << "Operator_Ext_Absorbing_BC::SetInitParams(): Warning: Absorbing sheet snapping failed! Dimension is: " << Snap_Dimension << " skipping. "
					<< " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		// Snap_Dimension == -2 means outside the simulation domain --> no special warning, but box probably marked as unused!
		return false;
	}

	// check that one of the dimensions is 1, and only one
	unsigned int sheetCheck = 0;
	m_ny = -1;
	for (int dimIdx = 0 ; dimIdx < 3 ; dimIdx++)
	{
		unsigned int Ncells = m_sheetX1[dimIdx] - m_sheetX0[dimIdx] + 1;

		// Count the number of dimensions that are single-cell length (start == stop)
		sheetCheck += Ncells == 1;

		// Update the normal direction
		m_ny = (Ncells == 1) ? dimIdx : m_ny;
	}

	if (sheetCheck != 1)
	{
		cerr 	<< "Operator_Ext_Absorbing_BC::SetInitParams(): Warning: Absorbing sheet is not a sheet! Skipping. "
				<< " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;
		return false;
	}

	m_phaseVelocity = abc_prop->GetPhaseVelocity();
	// Currently not handling arbitrary phase velocity
	if (m_phaseVelocity == 0.0)
	{
		cerr 	<< "Operator_Ext_Absorbing_BC::SetInitParams(): Warning: Absorbing sheet Currently does not support per-material velocity. Setting to C0 "
				<< " ID: " << prim->GetID() << " @ Property: " << abc_prop->GetName() << endl;

		m_phaseVelocity = __C0__;
	}

	// Copy all of the relevant data, so BuildExtension can work
	m_ABCtype = (ABCtype)(abc_prop->GetAbsorbingBoundaryType());
	m_normalSignPositive = abc_prop->GetNormalSignPositive();

	if (m_ABCtype == ABCtype::CPML)
	{
		m_pmlDepth        = std::max<unsigned int>(1, abc_prop->GetCPMLDepth());
		m_pmlSigmaMax     = abc_prop->GetCPMLSigmaMax();
		m_pmlAlphaMax     = abc_prop->GetCPMLAlphaMax();
		m_pmlProfileOrder = std::max<unsigned int>(1, abc_prop->GetCPMLProfileOrder());

		// CPML-strip geometry convention (2026-05-13): the user's "sheet"
		// sits at the INNER edge of the strip and the strip extends
		// OUTWARD toward the simulation boundary, terminating at the PEC
		// backstop.  m_normalSignPositive describes the outward normal of
		// the absorbing face, so the strip grows in that direction.
		//
		// This matches the σ profile baked below: k=0 at the sheet
		// (zeta≈0 → σ≈0, α≈α_max), k=D-1 at the outer cell (zeta≈1 →
		// σ≈σ_max, α≈0).  A wave entering the strip sees σ ramp up from 0
		// at the sheet to σ_max at the PEC, which is canonical
		// Roden-Gedney 2000 / Bérenger 2002.
		//
		// Prior to this fix the sign was inverted (NSP? -1 : +1), causing
		// the strip cells to be assigned INTO the simulation interior with
		// σ_max landing on the cell the wave hits first — a ~60-80 dB
		// reflection penalty vs Bérenger's predicted bound.
		m_pmlStepSign = m_normalSignPositive ? +1 : -1;
	}

	prim->SetPrimitiveUsed(true);

	return true;
}

bool Operator_Ext_Absorbing_BC::BuildExtension()
{
	// ---- CPML branch ----
	// Build the per-cell convolution coefficients for the PML strip behind the
	// sheet. Geometry: the user's box is a 1-cell-thick sheet at m_sheetX0[m_ny]
	// (== m_sheetX1[m_ny]). The PML occupies cells
	//     m_sheetX0[m_ny] + k * m_pmlStepSign,    k = 0 .. m_pmlDepth - 1
	// k=0 is the inner edge (lossless, alpha-only), k=D-1 is the outer edge
	// (sigma at peak).
	if (m_ABCtype == ABCtype::CPML)
	{
		const double dT = m_Op->GetTimestep();

		// First, validate that the PML strip fits in the domain along m_ny.
		const int sheet_pos = (int)m_sheetX0[m_ny];
		const int outer_pos = sheet_pos + m_pmlStepSign * (int)(m_pmlDepth - 1);
		const int n_lines   = (int)m_Op->GetNumberOfLines(m_ny, true);
		if (outer_pos < 0 || outer_pos >= n_lines)
		{
			cerr << "Operator_Ext_Absorbing_BC::BuildExtension(): CPML strip of depth "
			     << m_pmlDepth << " does not fit in the domain along axis "
			     << m_ny << " starting from sheet position " << sheet_pos
			     << " (outer_pos=" << outer_pos << ", n_lines=" << n_lines << ")."
			     << " Move the sheet inward or shrink CPMLDepth." << endl;
			return false;
		}

		// Number of in-plane cells (transverse to m_ny). Mirrors the Mur path.
		unsigned int Ncells[3];
		for (unsigned int dimIdx = 0; dimIdx < 3; ++dimIdx)
			Ncells[dimIdx] = m_sheetX1[dimIdx] - m_sheetX0[dimIdx] + 1;

		m_nyP  = (m_ny + 1) % 3;
		m_nyPP = (m_ny + 2) % 3;
		m_numLines[0] = Ncells[m_nyP];
		m_numLines[1] = Ncells[m_nyPP];
		m_numCells    = Ncells[0] * Ncells[1] * Ncells[2] * m_pmlDepth;

		// Auto-pick sigma_max from R(0) target if user left it at 0.
		// Standard recipe: sigma_max = -(p+1) ln(R0) / (2 eta d) where d is the
		// physical PML thickness in metres.
		unsigned int pos_at_sheet[3] = {0, 0, 0};
		pos_at_sheet[m_ny]   = m_sheetX0[m_ny];
		pos_at_sheet[m_nyP]  = m_sheetX0[m_nyP];
		pos_at_sheet[m_nyPP] = m_sheetX0[m_nyPP];
		// Operator::GetEdgeLength already returns the edge length in metres
		// (it does GetDiscDelta(n,pos,dual) * gridDelta internally). Do NOT
		// multiply by GetGridDelta() again — that double-converts and yields
		// a 1000× too-small thickness, making the auto sigma_max 1000× too
		// big which produces ~10⁻¹³⁵ b_z values at the outer edge (effectively
		// a hard wall, with the impedance mismatch reflecting back into the
		// interior — the root cause of the late-time blow-up).
		const double delta_m = std::fabs(m_Op->GetEdgeLength(m_ny, pos_at_sheet));
		const double pml_thickness_m = m_pmlDepth * delta_m;
		const double eta_bg = __Z0__ * std::sqrt(m_Op->GetBackgroundMueR()
		                                       / m_Op->GetBackgroundEpsR());

		double sigma_max = m_pmlSigmaMax;
		if (sigma_max <= 0.0)
			sigma_max = -(double)(m_pmlProfileOrder + 1)
			            * std::log(CPML_DEFAULT_R0)
			            / (2.0 * eta_bg * pml_thickness_m);

		double alpha_max = m_pmlAlphaMax;
		if (alpha_max <= 0.0)
			alpha_max = CPML_DEFAULT_ALPHA_MAX;

		const double kappa_max = std::max(1.0, m_pmlKappaMax);

		// [CPML-INIT-PARAMS] log the resolved parameters before per-k profile.
		std::cerr << "[CPML-INIT-PARAMS] axis=" << m_ny
		          << " sheet=(" << m_sheetX0[0] << "," << m_sheetX0[1] << "," << m_sheetX0[2] << ")"
		          << " stepSign=" << m_pmlStepSign
		          << " depth=" << m_pmlDepth
		          << " p=" << m_pmlProfileOrder
		          << " sigma_max=" << sigma_max
		          << " alpha_max=" << alpha_max
		          << " kappa_max=" << kappa_max
		          << " dT=" << dT
		          << " eta_bg=" << eta_bg
		          << " thickness_m=" << pml_thickness_m
		          << std::endl;

		m_pml_b_z.assign(m_pmlDepth, 0.0);
		std::vector<double> c_z(m_pmlDepth, 0.0);
		std::vector<double> kappa_z(m_pmlDepth, 1.0);
		std::vector<double> sigma_z_log(m_pmlDepth, 0.0);
		std::vector<double> alpha_z_log(m_pmlDepth, 0.0);
		std::vector<double> tau_log(m_pmlDepth, 0.0);

		const double D = (double)m_pmlDepth;
		const double p = (double)m_pmlProfileOrder;

		for (unsigned int k = 0; k < m_pmlDepth; ++k)
		{
			// Cell-centred normalised depth. k=0 nearest the sheet (zeta small),
			// k=D-1 deepest (zeta near 1).
			const double zeta = ((double)k + 0.5) / D;
			const double sigma_z = sigma_max * std::pow(zeta, p);
			// Alpha peaks at the inner edge so DC drains as soon as the wave
			// enters the PML. Linear taper is conventional (p_alpha = 1).
			const double alpha_z = alpha_max * (1.0 - zeta);
			const double kap     = 1.0 + (kappa_max - 1.0) * std::pow(zeta, p);
			const double tau     = (sigma_z / kap + alpha_z) * dT / __EPS0__;
			const double b       = std::exp(-tau);
			const double denom   = sigma_z * kap + kap * kap * alpha_z;
			kappa_z[k]   = kap;
			m_pml_b_z[k] = b;
			c_z[k]       = (denom > 0.0) ? (sigma_z / denom) * (b - 1.0) : 0.0;
			sigma_z_log[k] = sigma_z;
			alpha_z_log[k] = alpha_z;
			tau_log[k]   = tau;
		}

		// [CPML-INIT-PROF] per-k profile: zeta, sigma, alpha, kappa, tau, b, c.
		// Tab-separated for easy CSV parsing.
		std::cerr << "[CPML-INIT-PROF] k\tzeta\tsigma_z\talpha_z\tkappa_z\ttau\tb_z\tc_z\t1/(1-b)\tc/(1-b)" << std::endl;
		for (unsigned int k = 0; k < m_pmlDepth; ++k)
		{
			const double zeta = ((double)k + 0.5) / D;
			const double one_over_one_minus_b = (m_pml_b_z[k] < 1.0)
			    ? 1.0 / (1.0 - m_pml_b_z[k]) : std::numeric_limits<double>::infinity();
			std::cerr << "[CPML-INIT-PROF] " << k
			          << "\t" << zeta
			          << "\t" << sigma_z_log[k]
			          << "\t" << alpha_z_log[k]
			          << "\t" << kappa_z[k]
			          << "\t" << tau_log[k]
			          << "\t" << m_pml_b_z[k]
			          << "\t" << c_z[k]
			          << "\t" << one_over_one_minus_b
			          << "\t" << c_z[k] * one_over_one_minus_b
			          << std::endl;
		}

		// Bake the engine's m_ny-axis curl signs and the local vi/iv coefficient
		// into per-cell multipliers so the engine update is sign-free.
		// Sign convention derived from Engine::UpdateVoltages/UpdateCurrents:
		//   V[m_nyP ] gets m_ny term with sign -1, "other" current = m_nyPP
		//   V[m_nyPP] gets m_ny term with sign +1, "other" current = m_nyP
		//   I[m_nyP ] gets m_ny term with sign +1, "other" voltage = m_nyPP
		//   I[m_nyPP] gets m_ny term with sign -1, "other" voltage = m_nyP
		unsigned int psi_extent[3] = {m_numLines[0], m_numLines[1], m_pmlDepth};
		m_pml_cv_nyP .Init("pml_cv_nyP" , psi_extent);
		m_pml_cv_nyPP.Init("pml_cv_nyPP", psi_extent);
		m_pml_ci_nyP .Init("pml_ci_nyP" , psi_extent);
		m_pml_ci_nyPP.Init("pml_ci_nyPP", psi_extent);
		m_pml_kapV_nyP .Init("pml_kapV_nyP" , psi_extent);
		m_pml_kapV_nyPP.Init("pml_kapV_nyPP", psi_extent);
		m_pml_kapI_nyP .Init("pml_kapI_nyP" , psi_extent);
		m_pml_kapI_nyPP.Init("pml_kapI_nyPP", psi_extent);

		unsigned int pos[3] = {0, 0, 0};
		for (unsigned int i = 0; i < m_numLines[0]; ++i)
		{
			pos[m_nyP] = m_sheetX0[m_nyP] + i;
			for (unsigned int j = 0; j < m_numLines[1]; ++j)
			{
				pos[m_nyPP] = m_sheetX0[m_nyPP] + j;
				for (unsigned int k = 0; k < m_pmlDepth; ++k)
				{
					pos[m_ny] = (unsigned int)((int)m_sheetX0[m_ny]
					                           + m_pmlStepSign * (int)k);
					const FDTD_FLOAT vi_P  = m_Op->GetVI(m_nyP , pos[0], pos[1], pos[2]);
					const FDTD_FLOAT vi_PP = m_Op->GetVI(m_nyPP, pos[0], pos[1], pos[2]);
					const FDTD_FLOAT iv_P  = m_Op->GetIV(m_nyP , pos[0], pos[1], pos[2]);
					const FDTD_FLOAT iv_PP = m_Op->GetIV(m_nyPP, pos[0], pos[1], pos[2]);

					// Diagnostic 2026-05-08: log vi/iv at strip boundary cells
					// (k=0 inner, k=D-1 outer) AND at adjacent non-strip cells
					// at TWO transverse positions: corner (i=0,j=0) and
					// interior (i=mid, j=mid) to distinguish PEC-corner
					// effects from bulk-cross-section discontinuity.
					const unsigned int i_mid = m_numLines[0] / 2;
					const unsigned int j_mid = m_numLines[1] / 2;
					const bool is_corner = (i == 0 && j == 0);
					const bool is_mid    = (i == i_mid && j == j_mid);
					if ((is_corner || is_mid) && (k == 0 || k == m_pmlDepth - 1))
					{
						const char* loc_name = is_corner ? "CORNER" : "MID";
						unsigned int pos_adj[3] = {pos[0], pos[1], pos[2]};
						int adj_offset = (k == 0) ? -m_pmlStepSign : +m_pmlStepSign;
						int adj_ny = (int)pos[m_ny] + adj_offset;
						pos_adj[m_ny] = (unsigned int)adj_ny;
						const FDTD_FLOAT vi_P_adj  = m_Op->GetVI(m_nyP , pos_adj[0], pos_adj[1], pos_adj[2]);
						const FDTD_FLOAT vi_PP_adj = m_Op->GetVI(m_nyPP, pos_adj[0], pos_adj[1], pos_adj[2]);
						const FDTD_FLOAT iv_P_adj  = m_Op->GetIV(m_nyP , pos_adj[0], pos_adj[1], pos_adj[2]);
						const FDTD_FLOAT iv_PP_adj = m_Op->GetIV(m_nyPP, pos_adj[0], pos_adj[1], pos_adj[2]);
						const char* edge_name = (k == 0) ? "INNER" : "OUTER";
						std::cerr << "[CPML-DIAG] " << edge_name << "/" << loc_name
						          << " strip cell (i,j,z)=(" << i << "," << j << "," << pos[m_ny] << ")"
						          << " vi_P=" << vi_P << " vi_PP=" << vi_PP
						          << " iv_P=" << iv_P << " iv_PP=" << iv_PP << std::endl;
						std::cerr << "[CPML-DIAG] " << edge_name << "/" << loc_name
						          << " adj   cell (i,j,z)=(" << i << "," << j << "," << pos_adj[m_ny] << ")"
						          << " vi_P=" << vi_P_adj << " vi_PP=" << vi_PP_adj
						          << " iv_P=" << iv_P_adj << " iv_PP=" << iv_PP_adj << std::endl;
						const FDTD_FLOAT rel_vi_P  = (vi_P  != 0) ? std::fabs((vi_P_adj - vi_P) / vi_P) : 0;
						const FDTD_FLOAT rel_vi_PP = (vi_PP != 0) ? std::fabs((vi_PP_adj - vi_PP) / vi_PP) : 0;
						std::cerr << "[CPML-DIAG] " << edge_name << "/" << loc_name
						          << " rel-jump |Δvi/vi|: P=" << rel_vi_P << " PP=" << rel_vi_PP << std::endl;
					}

					// Per cell, c_z and the kappa-stretch correction are baked
					// together. The engine ran V += vi*sign*diff_M for the full
					// (un-stretched) curl. We need vi*sign*((1/kappa)*diff_M +
					// Psi_unsigned). Decompose:
					//   total CPML correction = vi*sign*(1/kappa - 1)*diff_M
					//                         + vi*sign*Psi_unsigned
					// Pre-bake (sign * vi * (1/kappa - 1)) into m_pml_kap_*
					// and (sign * vi * c_z) into m_pml_cv_* / m_pml_ci_*.
					const double inv_kap_minus_one = (kappa_z[k] != 0.0)
					    ? (1.0 / kappa_z[k] - 1.0) : 0.0;

					// 2026-05-08 STEPSIGN-AWARE CV/CI experiment REVERTED — also
					// failed (1.094/step still on active port 1).  Empirical
					// evidence so far suggests the asymmetry hypothesis is
					// wrong; both ports' CPML are equally broken, port 2 just
					// stays bounded because no source reaches there.
					m_pml_cv_nyP (i, j, k) = (FDTD_FLOAT)(-1.0 * c_z[k] * vi_P );
					m_pml_cv_nyPP(i, j, k) = (FDTD_FLOAT)(+1.0 * c_z[k] * vi_PP);
					m_pml_ci_nyP (i, j, k) = (FDTD_FLOAT)(+1.0 * c_z[k] * iv_P );
					m_pml_ci_nyPP(i, j, k) = (FDTD_FLOAT)(-1.0 * c_z[k] * iv_PP);

					m_pml_kapV_nyP (i, j, k) = (FDTD_FLOAT)(-1.0 * inv_kap_minus_one * vi_P );
					m_pml_kapV_nyPP(i, j, k) = (FDTD_FLOAT)(+1.0 * inv_kap_minus_one * vi_PP);
					m_pml_kapI_nyP (i, j, k) = (FDTD_FLOAT)(+1.0 * inv_kap_minus_one * iv_P );
					m_pml_kapI_nyPP(i, j, k) = (FDTD_FLOAT)(-1.0 * inv_kap_minus_one * iv_PP);
				}
			}
		}

		// [CPML-INIT-GEOM] final summary: which cells the PML occupies
		// and the per-edge baked coefficients at corner+mid spot cells.
		const unsigned int i_mid_log = m_numLines[0] / 2;
		const unsigned int j_mid_log = m_numLines[1] / 2;
		std::cerr << "[CPML-INIT-GEOM] axis=" << m_ny
		          << " sheet[m_ny]=" << m_sheetX0[m_ny]
		          << " stepSign=" << m_pmlStepSign
		          << " inner_pos=" << m_sheetX0[m_ny]
		          << " outer_pos=" << ((int)m_sheetX0[m_ny] + m_pmlStepSign * (int)(m_pmlDepth - 1))
		          << " depth=" << m_pmlDepth
		          << " transverse=(" << m_numLines[0] << "x" << m_numLines[1] << ")"
		          << std::endl;
		// Per-edge spot-check of the baked coefficients (corner cell, mid cell).
		for (unsigned int k_log = 0; k_log < m_pmlDepth; ++k_log)
		{
			std::cerr << "[CPML-INIT-COEFS] k=" << k_log
			          << " | corner cv_P=" << m_pml_cv_nyP(0, 0, k_log)
			          << " cv_PP=" << m_pml_cv_nyPP(0, 0, k_log)
			          << " ci_P=" << m_pml_ci_nyP(0, 0, k_log)
			          << " ci_PP=" << m_pml_ci_nyPP(0, 0, k_log)
			          << " kapV_P=" << m_pml_kapV_nyP(0, 0, k_log)
			          << " kapV_PP=" << m_pml_kapV_nyPP(0, 0, k_log)
			          << " | mid cv_P=" << m_pml_cv_nyP(i_mid_log, j_mid_log, k_log)
			          << " cv_PP=" << m_pml_cv_nyPP(i_mid_log, j_mid_log, k_log)
			          << " kapV_P=" << m_pml_kapV_nyP(i_mid_log, j_mid_log, k_log)
			          << std::endl;
		}

		return true;
	}

	double dT	= m_Op->GetTimestep();

	unsigned int	pos[] = {0,0,0};
	double			coord[] = {0.0,0.0,0.0};

	double 			delta;

	unsigned int	Ncells[] = {0,0,0},
					totCells = 1;

	// Count the number of cells in each direction
	for (unsigned int dimIdx = 0 ; dimIdx < 3 ; dimIdx++)
	{
		Ncells[dimIdx] = m_sheetX1[dimIdx] - m_sheetX0[dimIdx] + 1;
		totCells *= Ncells[dimIdx];
	}

	m_numCells = totCells;

	// Update remaining directions
	m_nyP 	= (m_ny + 1) % 3;
	m_nyPP 	= (m_ny + 2) % 3;

	// The position starts and stops in the same place.
	pos[m_ny] = m_sheetX0[m_ny];

	// The position is considered in the middle of the mesh cell.
	delta = fabs(m_Op->GetEdgeLength(m_ny,pos));

	// Initialize coefficients for this engine extension
	FDTD_FLOAT	vt_nyP  = m_phaseVelocity*dT,
				vt_nyPP = m_phaseVelocity*dT;

	// Initialize number of lines to be used in the arrayIJ
	m_numLines[0] = Ncells[m_nyP];
	m_numLines[1] = Ncells[m_nyPP];

	// Initialize containers. If there are more BCs in the future, this needs to be updated with the respective conditions.
	m_K1_nyP.Init("K1_Coeff_nyP", m_numLines);
	m_K1_nyPP.Init("K1_Coeff_nyPP", m_numLines);

	if (m_ABCtype == ABCtype::MUR_1ST_SA)
	{
		m_K2_nyP.Init("K2_Coeff_nyP", m_numLines);
		m_K2_nyPP.Init("K2_Coeff_nyPP", m_numLines);
	}

	// Prepare containers for per-material assignment
	coord[m_ny] = m_Op->GetDiscLine(m_ny,pos[m_ny]);
	// Check if this is the first or last cell
	if (m_sheetX0[m_ny] == 0)
		coord[m_ny] +=  delta/2 / m_Op->GetGridDelta();
	if (m_sheetX0[m_ny] == (m_Op->GetNumberOfLines(m_ny,true)-1))
		coord[m_ny] += -delta/2 / m_Op->GetGridDelta();

	// Initialize array coefficients
	unsigned int	arrI = 0,
					arrJ;

	for (pos[m_nyP] = m_sheetX0[m_nyP] ; pos[m_nyP] <= m_sheetX1[m_nyP] ; ++pos[m_nyP])
	{
		coord[m_nyP] = m_Op->GetDiscLine(m_nyP,pos[m_nyP]);
		arrJ = 0;
		for (pos[m_nyPP] = m_sheetX0[m_nyPP] ; pos[m_nyPP] <= m_sheetX1[m_nyPP] ; ++pos[m_nyPP])
		{
			coord[m_nyPP] = m_Op->GetDiscLine(m_nyPP,pos[m_nyPP]);

			if (m_phaseVelocity == 0.0)
			{
				double eps,mue;
				CSProperties** prop = m_Op->GetGeometryCSX()->GetPropertiesByCoordsPriority(coord, CSProperties::MATERIAL, false);
				if(prop != NULL)
				{
					cerr << "Operator_Ext_Absorbing_BC::BuildExtension(): Warning: This shouldn't happen...";
					/*CSPropMaterial* mat = (*prop)->ToMaterial();

					eps = mat->GetEpsilonWeighted(int(m_dir[1]),coord);
					mue = mat->GetMueWeighted(int(m_dir[1]),coord);
					vt_nyP = __C0__ * dT / sqrt(eps*mue);

					eps = mat->GetEpsilonWeighted(int(m_dir[2]),coord);
					mue = mat->GetMueWeighted(int(m_dir[2]),coord);
					vt_nyPP = __C0__ * dT / sqrt(eps*mue);*/
				}
				else
				{
					eps = m_Op->GetBackgroundEpsR();
					mue = m_Op->GetBackgroundMueR();

					vt_nyP  = __C0__ * dT / sqrt(eps*mue);
					vt_nyPP = __C0__ * dT / sqrt(eps*mue);
				}

			}

			// If more boundary types of boundary conditions are added in the future,
			// this needs to be in a condition, as well.
			m_K1_nyP(arrI,arrJ)  = (vt_nyP  - delta) / (vt_nyP  + delta);
			m_K1_nyPP(arrI,arrJ) = (vt_nyPP - delta) / (vt_nyPP + delta);

			if (m_ABCtype == ABCtype::MUR_1ST_SA)
			{
				m_K2_nyP(arrI,arrJ)  = vt_nyP /delta;
				m_K2_nyPP(arrI,arrJ) = vt_nyPP/delta;
			}

			arrJ++;
		}
		arrI++;
	}

	return true;
}

Engine_Extension* Operator_Ext_Absorbing_BC::CreateEngineExtention()
{
	Engine_Ext_Absorbing_BC* eng_ext = new Engine_Ext_Absorbing_BC(this);
	return eng_ext;
}

void Operator_Ext_Absorbing_BC::ShowStat(std::ostream &ostr) const
{
	Operator_Extension::ShowStat(ostr);

	ostr << " Total cells: " << m_numCells << endl;
}









