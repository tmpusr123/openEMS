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

#include <iomanip>
#include "tools/global.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include "processfields.h"
#include "FDTD/engine_interface_fdtd.h"
#include "field_gather_backend.h"
#include <cstdlib>

ProcessFields::ProcessFields(Engine_Interface_Base* eng_if) : Processing(eng_if)
{
	m_DumpType = E_FIELD_DUMP;
	// vtk-file is default
	m_fileType = VTK_FILETYPE;
	m_SampleType = NONE;
	m_Vtk_Dump_File = NULL;
	m_HDF5_Dump_File = NULL;
	SetPrecision(6);
	m_dualTime = false;

	// dump box should be always inside the snapped lines
	m_SnapMethod = 1;

	m_gpu_backend = NULL;
	m_gpu_dump_id = -1;
	m_gpu_hostbuf = NULL;
	m_gpu_tried = false;

	for (int n=0; n<3; ++n)
	{
		numLines[n]=0;
		posLines[n]=NULL;
		discLines[n]=NULL;
		subSample[n]=1;
		optResolution[n]=0;
	}
}

ProcessFields::~ProcessFields()
{
	delete m_Vtk_Dump_File;
	m_Vtk_Dump_File = NULL;
	for (int n=0; n<3; ++n)
	{
		delete[] posLines[n];
		posLines[n]=NULL;
		delete[] discLines[n];
		discLines[n]=NULL;
	}
}

string ProcessFields::GetFieldNameByType(DumpType type)
{
	switch (type)
	{
	case E_FIELD_DUMP:
		return "E-Field";
	case H_FIELD_DUMP:
		return "H-Field";
	case J_FIELD_DUMP:
		return "J-Field";
	case ROTH_FIELD_DUMP:
		return "RotH-Field";
	case D_FIELD_DUMP:
		return "D-Field";
	case B_FIELD_DUMP:
		return "B-Field";
	case SAR_LOCAL_DUMP:
		return "SAR-local";
	case SAR_1G_DUMP:
		return "SAR_1g";
	case SAR_10G_DUMP:
		return "SAR_10g";
	case SAR_RAW_DATA:
		return "SAR_raw_data";
	}
	return "unknown field";
}

bool ProcessFields::NeedConductivity() const
{
	switch (m_DumpType)
	{
	case J_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

bool ProcessFields::NeedPermittivity() const
{
	switch (m_DumpType)
	{
	case D_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

bool ProcessFields::NeedPermeability() const
{
	switch (m_DumpType)
	{
	case B_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

void ProcessFields::InitProcess()
{
	if (Enabled==false) return;

	CalcMeshPos();

	if (m_fileType==VTK_FILETYPE)
	{
		delete m_Vtk_Dump_File;
		m_Vtk_Dump_File = new VTK_File_Writer(m_filename,(int)m_Mesh_Type);

		#ifdef OUTPUT_IN_DRAWINGUNITS
		double discScaling = 1;
		#else
		double discScaling = Op->GetGridDelta();
		#endif
		m_Vtk_Dump_File->SetMeshLines(discLines,numLines,discScaling);
		m_Vtk_Dump_File->SetNativeDump(g_settings.NativeFieldDumps());
	}
	if (m_fileType==HDF5_FILETYPE)
	{
		delete m_HDF5_Dump_File;
		m_HDF5_Dump_File = new HDF5_File_Writer(m_filename+".h5");

		#ifdef OUTPUT_IN_DRAWINGUNITS
		double discScaling = 1;
		#else
		double discScaling = Op->GetGridDelta();
		#endif
		m_HDF5_Dump_File->WriteRectMesh(numLines,discLines,(int)m_Mesh_Type,discScaling);

		m_HDF5_Dump_File->WriteAtrribute("/","openEMS_HDF5_version",0.2);
	}
}

void ProcessFields::SetDumpMode(Engine_Interface_Base::InterpolationType mode)
{
	m_Eng_Interface->SetInterpolationType(mode);
	if (mode==Engine_Interface_Base::CELL_INTERPOLATE)
		m_dualMesh=true;
	else if (mode==Engine_Interface_Base::NODE_INTERPOLATE)
		m_dualMesh=false;
	//else keep the preset/user defined case
}

void ProcessFields::DefineStartStopCoord(double* dstart, double* dstop)
{
	Processing::DefineStartStopCoord(dstart,dstop);

	// normalize order of start and stop
	for (int n=0; n<3; ++n)
	{
		if (start[n]>stop[n])
		{
			unsigned int help = start[n];
			start[n]=stop[n];
			stop[n]=help;
		}
	}
}

double ProcessFields::CalcTotalEnergyEstimate() const
{
	return m_Eng_Interface->CalcFastEnergy();
}

void ProcessFields::SetSubSampling(unsigned int subSampleRate, int dir)
{
	if (dir>2) return;
	if (dir<0)
	{
		subSample[0]=subSampleRate;
		subSample[1]=subSampleRate;
		subSample[2]=subSampleRate;
	}
	else subSample[dir]=subSampleRate;
	m_SampleType = SUBSAMPLE;
}

void ProcessFields::SetOptResolution(double optRes, int dir)
{
	if (dir>2) return;
	if (dir<0)
	{
		optResolution[0]=optRes;
		optResolution[1]=optRes;
		optResolution[2]=optRes;
	}
	else optResolution[dir]=optRes;
	m_SampleType = OPT_RESOLUTION;
}

void ProcessFields::CalcMeshPos()
{
	if ((m_SampleType==SUBSAMPLE) || (m_SampleType==NONE))
	{
		vector<unsigned int> tmp_pos;

		for (int n=0; n<3; ++n)
		{
			// construct new discLines
			tmp_pos.clear();
			for (unsigned int i=start[n]; i<=stop[n]; i+=subSample[n])
				tmp_pos.push_back(i);

			numLines[n] = tmp_pos.size();
			delete[] discLines[n];
			discLines[n] = new double[numLines[n]];
			delete[] posLines[n];
			posLines[n] = new unsigned int[numLines[n]];
			for (unsigned int i=0; i<numLines[n]; ++i)
			{
				posLines[n][i] = tmp_pos.at(i);
				discLines[n][i] = Op->GetDiscLine(n,tmp_pos.at(i),m_dualMesh);
			}
		}
	}
	if ((m_SampleType==OPT_RESOLUTION))
	{
		vector<unsigned int> tmp_pos;
		double oldPos=0;
		for (int n=0; n<3; ++n)
		{
			// construct new discLines
			tmp_pos.clear();
			tmp_pos.push_back(start[n]);
			oldPos=Op->GetDiscLine(n,start[n],m_dualMesh);
			if (stop[n]==0)
				tmp_pos.push_back(stop[n]);
			else
				for (unsigned int i=start[n]+1; i<=stop[n]-1; ++i)
				{
					if ( (Op->GetDiscLine(n,i+1,m_dualMesh)-oldPos) >= optResolution[n])
					{
						tmp_pos.push_back(i);
						oldPos=Op->GetDiscLine(n,i,m_dualMesh);
					}
				}
			if (start[n]!=stop[n])
				tmp_pos.push_back(stop[n]);
			numLines[n] = tmp_pos.size();
			delete[] discLines[n];
			discLines[n] = new double[numLines[n]];
			delete[] posLines[n];
			posLines[n] = new unsigned int[numLines[n]];
			for (unsigned int i=0; i<numLines[n]; ++i)
			{
				posLines[n][i] = tmp_pos.at(i);
				discLines[n][i] = Op->GetDiscLine(n,tmp_pos.at(i),m_dualMesh);
			}
		}
	}
}

// Build the interpolation stencil over the dump box and register it with the
// engine's GPU gather backend. Falls back silently (m_gpu_dump_id stays -1) if
// the engine is not a single-GPU CUDA engine, the dump type is unsupported, the
// user disabled it (OPENEMS_GPU_DUMP=0), or the stencil exceeds the VRAM budget.
void ProcessFields::SetupGpuGather()
{
	m_gpu_tried = true;
	const char* env = getenv("OPENEMS_GPU_DUMP");
	if (env && env[0]=='0') return;

	Engine_Interface_FDTD* ei = dynamic_cast<Engine_Interface_FDTD*>(m_Eng_Interface);
	if (!ei) return;
	FieldGatherBackend* be = dynamic_cast<FieldGatherBackend*>(const_cast<Engine*>(ei->GetFDTDEngine()));
	if (!be) return;
	if (m_DumpType>B_FIELD_DUMP) return;   // only the 6 vector field dumps

	size_t NI=numLines[0], NJ=numLines[1], NK=numLines[2];
	size_t nOut = 3*NI*NJ*NK;
	if (nOut==0) return;

	// Build each point's 3-component stencil once (BuildFieldStencil returns all
	// 3 at once) and bucket entries per component. The output buffer is laid out
	// exactly as HDF5_File_Writer::WriteVectorField stores it -- component order
	// {3, NK, NJ, NI} with i fastest, so output o = ((n*NK + k)*NJ + j)*NI + i --
	// so the gather buffer can be written straight to HDF5 with no repack.
	// Iterate spatially in (k, j, i) to match; concatenate per-component at end.
	size_t nSpatial = NI*NJ*NK;
	std::vector<unsigned int> csrc[3];            // per-component source indices
	std::vector<float>        ccoeff[3];          // per-component coefficients
	std::vector<unsigned int> ccount[3];          // entries per output point
	for (int n=0;n<3;++n) ccount[n].reserve(nSpatial);
	std::vector<FieldStencilEntry> st[3];
	bool useCurr=false;
	for (unsigned int k=0; k<NK; ++k)
		for (unsigned int j=0; j<NJ; ++j)
			for (unsigned int i=0; i<NI; ++i)
			{
				unsigned int pos[3] = {posLines[0][i], posLines[1][j], posLines[2][k]};
				if (!ei->BuildFieldStencil(pos, m_DumpType, st, useCurr)) return;
				for (int n=0;n<3;++n)
				{
					for (size_t e=0;e<st[n].size();++e) { csrc[n].push_back(st[n][e].src); ccoeff[n].push_back(st[n][e].coeff); }
					ccount[n].push_back((unsigned int)st[n].size());
				}
			}

	std::vector<unsigned int> offsets; offsets.reserve(nOut+1);
	std::vector<unsigned int> src;
	std::vector<float> coeff;
	offsets.push_back(0);
	size_t run = 0;
	for (int n=0;n<3;++n)   // concatenate in component order to match CSR output layout
	{
		src.insert(src.end(), csrc[n].begin(), csrc[n].end());
		coeff.insert(coeff.end(), ccoeff[n].begin(), ccoeff[n].end());
		for (size_t p=0;p<ccount[n].size();++p) { run += ccount[n][p]; offsets.push_back((unsigned int)run); }
	}

	int id = be->RegisterFieldGather(offsets, src, coeff, useCurr, nOut, &m_gpu_hostbuf);
	if (id>=0) { m_gpu_backend = be; m_gpu_dump_id = id; }
	if (getenv("OPENEMS_PROF") && id>=0)
		fprintf(stderr, "[PROF] GPU dump stencil built: %zu outputs, %zu entries (%.0f MB device)\n",
		        nOut, src.size(), (src.size()*8.0 + nOut*4.0)/1048576.0);
}

const float* ProcessFields::GpuFreshBuffer() const
{
	if (m_gpu_dump_id<0) return NULL;
	if (m_gpu_backend->GetGatherTS(m_gpu_dump_id)==(long)m_Eng_Interface->GetNumberOfTimesteps())
		return m_gpu_hostbuf;
	return NULL;
}

FDTD_FLOAT**** ProcessFields::CalcField()
{
	//create array
	FDTD_FLOAT**** field = Create_N_3DArray<FDTD_FLOAT>(numLines);

	switch (m_DumpType)
	{
	case E_FIELD_DUMP: case H_FIELD_DUMP: case J_FIELD_DUMP:
	case ROTH_FIELD_DUMP: case D_FIELD_DUMP: case B_FIELD_DUMP:
		break;
	default:
		cerr << "ProcessFields::CalcField(): Error, unknown dump type..." << endl;
		return field;
	}

	if (!m_gpu_tried) SetupGpuGather();

	// GPU path: the engine gathered the interpolated fields straight from the
	// device at the last chunk finish. Use that buffer when it holds the
	// current timestep (it always does at a dump boundary, since dumps align to
	// chunk boundaries). At t=0 the buffer is not yet filled and the fields are
	// zero, so the freshly-zeroed 'field' array is already correct.
	if (m_gpu_dump_id>=0)
	{
		unsigned int nowTS = m_Eng_Interface->GetNumberOfTimesteps();
		if (m_gpu_backend->GetGatherTS(m_gpu_dump_id)==(long)nowTS)
		{
			// buffer layout is HDF5 order: o = ((n*NK + k)*NJ + j)*NI + i
			int NI=(int)numLines[0], NJ=(int)numLines[1], NK=(int)numLines[2];
			const float* buf = m_gpu_hostbuf;
#ifdef _OPENMP
			#pragma omp parallel for collapse(2) schedule(static)
#endif
			for (int n=0; n<3; ++n)
				for (int k=0; k<NK; ++k)
					for (int j=0; j<NJ; ++j)
					{
						size_t base = ((size_t)(n*NK+k)*NJ + j)*NI;
						for (int i=0; i<NI; ++i)
							field[n][i][j][k] = buf[base+i];
					}
		}
		else if (nowTS!=0)
		{
			cerr << "ProcessFields: GPU dump buffer stale at TS " << nowTS
			     << " (have " << m_gpu_backend->GetGatherTS(m_gpu_dump_id)
			     << ") -- results for this frame are zero; please report" << endl;
		}
		return field;
	}

	// The dump grid points are independent and the engine interface only READS
	// the (paused) engine/operator state here, so extract them in parallel.
	// This loop is the dominant cost of field monitors on fast engines (one
	// interpolating virtual call per point, millions of points per sample).
#ifdef _OPENMP
	#pragma omp parallel for collapse(2) schedule(static)
#endif
	for (int i=0; i<(int)numLines[0]; ++i)
	{
		for (int j=0; j<(int)numLines[1]; ++j)
		{
			unsigned int pos[3];
			double out[3];
			pos[0]=posLines[0][i];
			pos[1]=posLines[1][j];
			for (unsigned int k=0; k<numLines[2]; ++k)
			{
				pos[2]=posLines[2][k];
				switch (m_DumpType)
				{
				case E_FIELD_DUMP:    m_Eng_Interface->GetEField(pos,out);    break;
				case H_FIELD_DUMP:    m_Eng_Interface->GetHField(pos,out);    break;
				case J_FIELD_DUMP:    m_Eng_Interface->GetJField(pos,out);    break;
				case ROTH_FIELD_DUMP: m_Eng_Interface->GetRotHField(pos,out); break;
				case D_FIELD_DUMP:    m_Eng_Interface->GetDField(pos,out);    break;
				case B_FIELD_DUMP:    m_Eng_Interface->GetBField(pos,out);    break;
				default: out[0]=out[1]=out[2]=0; break;
				}
				field[0][i][j][k] = out[0];
				field[1][i][j][k] = out[1];
				field[2][i][j][k] = out[2];
			}
		}
	}
	return field;
}

