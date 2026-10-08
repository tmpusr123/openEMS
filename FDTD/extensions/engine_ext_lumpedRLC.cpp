/*
*	Additional
*	Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
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

#include "engine_ext_lumpedRLC.h"
#include "operator_ext_lumpedRLC.h"

#include "FDTD/engine_sse.h"

Engine_Ext_LumpedRLC::Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC* op_ext_RLC) : Engine_Extension(op_ext_RLC)
{
	// Local pointer of the operator.
	m_Op_Ext_RLC = op_ext_RLC;

	v_Vdn		= new FDTD_FLOAT*[3];
	v_Jn		= new FDTD_FLOAT*[3];
	v_Il		= NULL;
	v_q			= NULL;

	// No additional allocations are required if there are no actual lumped elements.
	if (!(m_Op_Ext_RLC->RLC_count))
		return;

	// Initialize ADE containers for currents and voltages
	v_Il 		= new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];
	v_q 		= new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

	for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
	{
		v_Il[posIdx] 	= 0.0;
		v_q[posIdx] 	= 0.0;
	}

	for (unsigned int k = 0 ; k < 3 ; k++)
	{
		v_Vdn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];
		v_Jn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

		for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
		{
			v_Jn[k][posIdx] = 0.0;
			v_Vdn[k][posIdx] = 0.0;;
		}
	}

}

Engine_Ext_LumpedRLC::~Engine_Ext_LumpedRLC()
{
	// Only delete if values were allocated in the first place
	if (m_Op_Ext_RLC->RLC_count)
	{
		delete[] v_Il;
		delete[] v_q;

		for (unsigned int k = 0 ; k < 3 ; k++)
		{
			delete[] v_Vdn[k];
			delete[] v_Jn[k];
		}
	}

	delete[] v_Vdn;
	delete[] v_Jn;

	v_Il	= NULL;
	v_q		= NULL;

	v_Vdn	= NULL;
	v_Jn	= NULL;

#if WITH_CUDA
	if (d_cells) cudaFree(d_cells);
	if (d_Il)    cudaFree(d_Il);
	if (d_Vd1)   cudaFree(d_Vd1);
	if (d_Vd2)   cudaFree(d_Vd2);
	if (d_J1)    cudaFree(d_J1);
	if (d_J2)    cudaFree(d_J2);
	for (size_t g = 0; g < mg_cells.size(); ++g)
	{
		cudaSetDevice(mg_dev[g]);
		if (mg_cells[g]) cudaFree(mg_cells[g]);
		if (mg_Il[g])    cudaFree(mg_Il[g]);
		if (mg_Vd1[g])   cudaFree(mg_Vd1[g]);
		if (mg_Vd2[g])   cudaFree(mg_Vd2[g]);
		if (mg_J1[g])    cudaFree(mg_J1[g]);
		if (mg_J2[g])    cudaFree(mg_J2[g]);
	}
#endif

	m_Op_Ext_RLC = NULL;


}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdates()
{
#if WITH_CUDA
	if (m_Eng->GetType() == Engine::CUDA) {
		if (m_Op_Ext_RLC->RLC_count)
			DoPreVoltageUpdatesCuda(static_cast<Engine_cuda*>(m_Eng));
		return;
	}
#endif
	unsigned int **pos = m_Op_Ext_RLC->v_RLC_pos;
	int *dir = m_Op_Ext_RLC->v_RLC_dir;

	// Iterate Vd containers
	FDTD_FLOAT	*v_temp;
	v_temp = v_Vdn[2];
	v_Vdn[2] = v_Vdn[1];
	v_Vdn[1] = v_Vdn[0];
	v_Vdn[0] = v_temp;

	// In pre-process, only update the parallel inductor current:
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
		v_Il[pIdx] += (m_Op_Ext_RLC->v_RLC_i2v[pIdx])*(m_Op_Ext_RLC->v_RLC_ilv[pIdx])*v_Vdn[1][pIdx];

	return;
}

template <typename EngType>
void Engine_Ext_LumpedRLC::Apply2VoltagesImpl(EngType* eng)
{
	unsigned int **pos = m_Op_Ext_RLC->v_RLC_pos;
	int *dir = m_Op_Ext_RLC->v_RLC_dir;

	// Iterate J containers
	FDTD_FLOAT	*v_temp;
	v_temp = v_Jn[2];
	v_Jn[2] = v_Jn[1];
	v_Jn[1] = v_Jn[0];
	v_Jn[0] = v_temp;


	// Read engine calculated node voltage
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
		v_Vdn[0][pIdx] = eng->EngType::GetVolt(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx]);

	// Post process: trapezoidal state-space series update (luikore e5aa1ec).
	// Vd[n-1] = v_Vdn[1], J[n-1] = v_Jn[1], q[n-1] = v_q. Parallel elements
	// have all series coefficients zero and vvd = 1: Vd = Vraw - Il, J = 0.
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
	{
		FDTD_FLOAT A = m_Op_Ext_RLC->v_RLC_dJdV[pIdx];
		FDTD_FLOAT B = m_Op_Ext_RLC->v_RLC_aV[pIdx]*v_Vdn[1][pIdx]
		             + m_Op_Ext_RLC->v_RLC_aQ[pIdx]*v_q[pIdx]
		             + m_Op_Ext_RLC->v_RLC_aJ[pIdx]*v_Jn[1][pIdx];
		FDTD_FLOAT vcd = m_Op_Ext_RLC->v_RLC_vcd[pIdx];

		// Solve the implicit node coupling
		//   Vd[n] = Vraw - (dT/2Cd)*(J[n] + J[n-1]),  J[n] = A*Vd[n] + B
		v_Vdn[0][pIdx] = m_Op_Ext_RLC->v_RLC_vvd[pIdx]
		               * (v_Vdn[0][pIdx] - v_Il[pIdx] - vcd*(B + v_Jn[1][pIdx]));

		// Updated element current
		v_Jn[0][pIdx] = A*v_Vdn[0][pIdx] + B;

		// Trapezoidal charge integration
		v_q[pIdx] = v_q[pIdx] + m_Op_Ext_RLC->m_dT_half*(v_Jn[0][pIdx] + v_Jn[1][pIdx]);
	}


	// Update node voltage
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
		eng->EngType::SetVolt(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx],v_Vdn[0][pIdx]);
}

void Engine_Ext_LumpedRLC::Apply2Voltages()
{
#if WITH_CUDA
	if (m_Eng->GetType() == Engine::CUDA) {
		if (m_Op_Ext_RLC->RLC_count)
			Apply2VoltagesCuda(static_cast<Engine_cuda*>(m_Eng));
		return;
	}
#endif
	ENG_DISPATCH(Apply2VoltagesImpl);
}
