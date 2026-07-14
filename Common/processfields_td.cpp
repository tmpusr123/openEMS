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

#include "processfields_td.h"
#include "Common/operator_base.h"
#include "FDTD/engine_interface_fdtd.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <cstdlib>

using namespace std;

// OPENEMS_PROF=1: cumulative wall time split between field interpolation
// (CalcField) and file write, summed over all TD dumps; printed at exit.
namespace {
	double g_td_calc = 0.0, g_td_write = 0.0;
	struct TDProfDump { ~TDProfDump() {
		if (getenv("OPENEMS_PROF"))
			fprintf(stderr, "[PROF] TD dumps: interpolation(CalcField) %.2fs | file write %.2fs\n",
			        g_td_calc, g_td_write);
	}} g_td_prof;
	static inline double _now() { timeval t; gettimeofday(&t,NULL); return t.tv_sec+1e-6*t.tv_usec; }
}

ProcessFieldsTD::ProcessFieldsTD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	pad_length = 8;
}

ProcessFieldsTD::~ProcessFieldsTD()
{
}

void ProcessFieldsTD::InitProcess()
{
	if (Enabled==false) return;
	
	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
	{
		m_Vtk_Dump_File->SetHeader(string("openEMS TD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());
		if (getenv("OPENEMS_VTK_NOCOMPRESS")) m_Vtk_Dump_File->SetCompress(false);
	}

	if (m_HDF5_Dump_File)
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/TD");
}

int ProcessFieldsTD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	string filename = m_filename;

	EnsureGpuGather();

	// Fast path: the engine already interpolated this dump on-device into a
	// contiguous buffer laid out exactly as the HDF5 writer stores it, so write
	// it straight out -- no jagged CalcField array, no repack. (HDF5 only; VTK
	// and the t=0 zero frame fall through to the generic path below.)
	if (m_fileType==HDF5_FILETYPE && GpuActive())
	{
		const float* buf = GpuFreshBuffer();
		if (buf)
		{
			double _tw = _now();
			stringstream ss;
			ss << std::setw( pad_length ) << std::setfill( '0' ) << m_Eng_Interface->GetNumberOfTimesteps();
			size_t n_size[4] = {3, numLines[2], numLines[1], numLines[0]};
			bool ok = m_HDF5_Dump_File->WriteData(ss.str(), buf, 4, n_size);
			float time[1] = {(float)m_Eng_Interface->GetTime(m_dualTime)};
			ok &= m_HDF5_Dump_File->WriteAtrribute("/FieldData/TD/"+ss.str(),"time",time,1);
			g_td_write += _now() - _tw;
			if (!ok) { SetEnable(false); cerr << "ProcessFieldsTD::Process: can't dump to file... disabled! " << endl; }
			return GetNextInterval();
		}
	}

	// OPENEMS_STENCIL_CHECK=1: once, on excited fields, verify the GPU-gather
	// interpolation stencil matches the direct interpolation path.
	if (getenv("OPENEMS_STENCIL_CHECK") && m_Eng_Interface->GetNumberOfTimesteps()>0)
	{
		static bool done=false;
		if (!done) {
			done=true;
			Engine_Interface_FDTD* ei = dynamic_cast<Engine_Interface_FDTD*>(m_Eng_Interface);
			if (ei) {
				const char* nm[6]={"E","H","J","rotH","D","B"};
				for (int dt=0; dt<6; ++dt)
					fprintf(stderr, "[STENCIL] %-4s dumpType=%d worst rel-err vs direct = %.3e\n",
					        nm[dt], dt, ei->VerifyFieldStencil(dt, 4000));
			}
		}
	}

	double _t0 = _now();
	float**** field = CalcField();
	double _t1 = _now();
	g_td_calc += _t1 - _t0;
	bool success = true;

	if (m_fileType==VTK_FILETYPE)
	{
		m_Vtk_Dump_File->SetTimestep(m_Eng_Interface->GetNumberOfTimesteps());
		m_Vtk_Dump_File->ClearAllFields();
		m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),field);
		success &= m_Vtk_Dump_File->Write();
	}
	else if (m_fileType==HDF5_FILETYPE)
	{
		stringstream ss;
		ss << std::setw( pad_length ) << std::setfill( '0' ) << m_Eng_Interface->GetNumberOfTimesteps();
		size_t datasize[]={numLines[0],numLines[1],numLines[2]};
		success &= m_HDF5_Dump_File->WriteVectorField(ss.str(), field, datasize);
		float time[1] = {(float)m_Eng_Interface->GetTime(m_dualTime)};
		success &= m_HDF5_Dump_File->WriteAtrribute("/FieldData/TD/"+ss.str(),"time",time,1);
	}
	else
	{
		success = false;
		cerr << "ProcessFieldsTD::Process: unknown File-Type" << endl;
	}

	g_td_write += _now() - _t1;

	Delete_N_3DArray<FDTD_FLOAT>(field,numLines);

	if (success==false)
	{
		SetEnable(false);
		cerr << "ProcessFieldsTD::Process: can't dump to file... disabled! " << endl;
	}

	return GetNextInterval();
}
