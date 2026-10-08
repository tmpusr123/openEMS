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

#include "processfields_fd.h"
#include "Common/operator_base.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <typeinfo>

using namespace std;

ProcessFieldsFD::ProcessFieldsFD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	m_device_dft_tried = false;
	m_device_dft = false;
	m_FD_SampleCount_ts0 = 0;
}

ProcessFieldsFD::~ProcessFieldsFD()
{
	for (size_t n = 0; n<m_FD_Fields.size(); ++n)
	{
		Delete_N_3DArray(m_FD_Fields.at(n),numLines);
	}
	m_FD_Fields.clear();
}

void ProcessFieldsFD::InitProcess()
{
	if (Enabled==false) return;

	if (m_FD_Samples.size()==0)
	{
		cerr << "ProcessFieldsFD::InitProcess: No frequencies found... skipping this dump!" << endl;
		Enabled=false;
		return;
	}

	//setup the hdf5 file
	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
	{
		m_Vtk_Dump_File->SetHeader(string("openEMS FD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());
		// FD dumps write their whole batch of phase/abs/arg frames at PostProcess
		// (once, at end of run) rather than per-timestep, but for many frequencies
		// that batch of compressed VTK writes is still a real shutdown cost, so
		// honor the same escape hatch as the TD path.
		if (getenv("OPENEMS_VTK_NOCOMPRESS")) m_Vtk_Dump_File->SetCompress(false);
	}

	if (m_HDF5_Dump_File)
	{
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/FD");
		m_HDF5_Dump_File->WriteAtrribute("/FieldData/FD","frequency",m_FD_Samples);
	}

	//create data structures...
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		std::complex<float>**** field_fd = Create_N_3DArray<std::complex<float> >(numLines);
		m_FD_Fields.push_back(field_fd);
	}
}

std::complex<float> ProcessFieldsFD::DFTFactor(size_t n, double T) const
{
	std::complex<float> exp_jwt_2_dt = std::exp( (std::complex<float>)(-2.0 * _I * M_PI * m_FD_Samples.at(n) * T) );
	exp_jwt_2_dt *= 2; // *2 for single-sided spectrum
	exp_jwt_2_dt *= Op->GetTimestep() * m_FD_Interval; // multiply with timestep-interval
	return exp_jwt_2_dt;
}

// The exact sampling decision of Process(), without its side effects. Only
// used in device-DFT mode, which requires m_ProcessSteps to be empty -- the one
// part of CheckTimestep() that has state.
bool ProcessFieldsFD::DeviceDFTFactors(unsigned int ts, std::vector<std::complex<float> >& factors) const
{
	if (Enabled==false) return false;
	if (ts<startTS || ts>stopTS) return false;
	if ((m_FD_Interval==0) || (ts%m_FD_Interval!=0)) return false;
	// The engine asks when it holds timestep ts, which is also what
	// GetTime() reports at that moment (the same value Process() uses).
	double T = m_Eng_Interface->GetTime(m_dualTime);
	factors.resize(m_FD_Samples.size());
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		factors[n] = DFTFactor(n, T);
	return true;
}

// Move the running DFT onto the device: the engine already gathers this dump's
// interpolated samples there, so instead of copying every sample to the host
// and summing it here, it keeps the frequency sums in device memory and this
// class reads them once at the end. Same factors, same per-cell summation
// order. Falls back to the host DFT whenever anything is unsupported.
void ProcessFieldsFD::SetupDeviceDFT()
{
	m_device_dft_tried = true;
	const char* env = getenv("OPENEMS_GPU_DFT");
	if (env && env[0]=='0') return;
	if (typeid(*this)!=typeid(ProcessFieldsFD)) return;   // SAR & co. keep their own Process()
	if (m_ProcessSteps.size()>0) return;                  // stateful CheckTimestep
	if (m_FD_Interval==0 || m_FD_Samples.size()==0) return;
	if (m_Eng_Interface->GetNumberOfTimesteps()!=0) return; // must see every chunk
	EnsureGpuGather();
	if (!GpuActive()) return;
	if (m_gpu_backend->EnableFieldDFT(m_gpu_dump_id, m_FD_Samples.size(), this))
	{
		m_device_dft = true;
		if (getenv("OPENEMS_PROF"))
			fprintf(stderr, "[PROF] FD dump '%s': running DFT on the device (%zu frequencies)\n", m_filename.c_str(), m_FD_Samples.size());
	}
}

int ProcessFieldsFD::Process()
{
	if (Enabled==false) return -1;
	if (!m_device_dft_tried) SetupDeviceDFT();
	if (CheckTimestep()==false) return GetNextInterval();

	if ((m_FD_Interval==0) || (m_Eng_Interface->GetNumberOfTimesteps()%m_FD_Interval!=0))
		return GetNextInterval();

	if (m_device_dft)
	{
		// The engine has already added this sample on the device (at the
		// chunk finish for this timestep). A sample at ts 0 sees all-zero
		// fields; the device never gets one, which adds nothing either way.
		if (m_Eng_Interface->GetNumberOfTimesteps()==0)
			++m_FD_SampleCount_ts0;
		++m_FD_SampleCount;
		return GetNextInterval();
	}

	FDTD_FLOAT**** field_td = CalcField();
	std::complex<float>**** field_fd = NULL;

	double T = m_Eng_Interface->GetTime(m_dualTime);
	const int NI = (int)numLines[0];
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		std::complex<float> exp_jwt_2_dt = DFTFactor(n, T);
		field_fd = m_FD_Fields.at(n);
		// Running-DFT accumulation is the dominant per-sample FD cost once the
		// interpolation (CalcField) is GPU/OpenMP-accelerated. Parallelize over
		// x-planes: each cell owns its running sum and the writes are disjoint,
		// so the result is bit-identical to the serial loop -- the per-cell
		// temporal accumulation order is unchanged (Process() is still invoked
		// once per sample, in timestep order).
#ifdef _OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (int x=0; x<NI; ++x)
		{
			for (unsigned int y=0; y<numLines[1]; ++y)
			{
				for (unsigned int z=0; z<numLines[2]; ++z)
				{
					field_fd[0][x][y][z] += field_td[0][x][y][z] * exp_jwt_2_dt;
					field_fd[1][x][y][z] += field_td[1][x][y][z] * exp_jwt_2_dt;
					field_fd[2][x][y][z] += field_td[2][x][y][z] * exp_jwt_2_dt;
				}
			}
		}
	}
	Delete_N_3DArray<FDTD_FLOAT>(field_td,numLines);
	++m_FD_SampleCount;
	return GetNextInterval();
}

// Copy the device sums into m_FD_Fields (gather order {3,NK,NJ,NI}, i fastest).
void ProcessFieldsFD::FetchDeviceDFT()
{
	const int NI=(int)numLines[0], NJ=(int)numLines[1], NK=(int)numLines[2];
	std::vector<std::complex<float> > buf((size_t)3*NI*NJ*NK);
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		long cnt = m_gpu_backend->ReadFieldDFT(m_gpu_dump_id, n, buf.data());
		if (cnt<0)
		{
			cerr << "ProcessFieldsFD: reading the device DFT of '" << m_filename << "' failed -- this dump is zero" << endl;
			return;
		}
		if (n==0 && cnt!=(long)(m_FD_SampleCount-m_FD_SampleCount_ts0))
			cerr << "ProcessFieldsFD: device DFT of '" << m_filename << "' holds " << cnt << " samples, expected "
			     << (m_FD_SampleCount-m_FD_SampleCount_ts0) << " -- please report" << endl;
		std::complex<float>**** field_fd = m_FD_Fields.at(n);
#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static)
#endif
		for (int c=0; c<3; ++c)
			for (int k=0; k<NK; ++k)
				for (int j=0; j<NJ; ++j)
				{
					size_t base = ((size_t)(c*NK+k)*NJ + j)*NI;
					for (int i=0; i<NI; ++i)
						field_fd[c][i][j][k] = buf[base+i];
				}
	}
}

void ProcessFieldsFD::PostProcess()
{
	if (m_device_dft)
		FetchDeviceDFT();
	DumpFDData();
}

void ProcessFieldsFD::DumpFDData()
{
	if (m_fileType==VTK_FILETYPE)
	{
		// Every frame (21 phases, magnitude, phase angle) is an independent
		// compressed file of the whole box, and writing them one after another
		// was the dominant cost of a field-monitor run (minutes for a few
		// monitors of ~1M points). Write the frames of a frequency concurrently,
		// one writer per frame; each file's content is unchanged.
		// OPENEMS_FD_VTK_PHASES_ONLY=1 skips the _abs/_arg frames for callers
		// that only use the phase series.
		const int Nr_Ph = 21;
		const char* po = getenv("OPENEMS_FD_VTK_PHASES_ONLY");
		const int nFrames = Nr_Ph + ((po && po[0]=='1') ? 0 : 2);
		const std::string fieldName = GetFieldNameByType(m_DumpType);
		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			std::complex<float>**** field_fd = m_FD_Fields.at(n);
			int failed = 0;
#ifdef _OPENMP
			#pragma omp parallel for schedule(dynamic,1) reduction(+:failed)
#endif
			for (int fr=0; fr<nFrames; ++fr)
			{
				unsigned int pos[3];
				FDTD_FLOAT**** field = Create_N_3DArray<float>(numLines);
				stringstream ss;
				if (fr<Nr_Ph)
				{
					//dump multiple phase to vtk-files
					double angle = 2.0 * M_PI * fr / Nr_Ph;
					std::complex<float> exp_jwt = std::exp( (std::complex<float>)( _I * angle) );
					for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
						for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
							for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
							{
								field[0][pos[0]][pos[1]][pos[2]] = real(field_fd[0][pos[0]][pos[1]][pos[2]] * exp_jwt);
								field[1][pos[0]][pos[1]][pos[2]] = real(field_fd[1][pos[0]][pos[1]][pos[2]] * exp_jwt);
								field[2][pos[0]][pos[1]][pos[2]] = real(field_fd[2][pos[0]][pos[1]][pos[2]] * exp_jwt);
							}
					ss << m_filename << fixed << "_f=" << m_FD_Samples.at(n) << "_p=" << std::setw( 3 ) << std::setfill( '0' ) <<(int)(angle * 180 / M_PI);
				}
				else if (fr==Nr_Ph)
				{
					//dump magnitude to vtk-files
					for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
						for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
							for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
							{
								field[0][pos[0]][pos[1]][pos[2]] = abs(field_fd[0][pos[0]][pos[1]][pos[2]]);
								field[1][pos[0]][pos[1]][pos[2]] = abs(field_fd[1][pos[0]][pos[1]][pos[2]]);
								field[2][pos[0]][pos[1]][pos[2]] = abs(field_fd[2][pos[0]][pos[1]][pos[2]]);
							}
					ss << m_filename << fixed << "_f=" << m_FD_Samples.at(n) << "_abs";
				}
				else
				{
					//dump phase to vtk-files
					for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
						for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
							for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
							{
								field[0][pos[0]][pos[1]][pos[2]] = arg(field_fd[0][pos[0]][pos[1]][pos[2]]);
								field[1][pos[0]][pos[1]][pos[2]] = arg(field_fd[1][pos[0]][pos[1]][pos[2]]);
								field[2][pos[0]][pos[1]][pos[2]] = arg(field_fd[2][pos[0]][pos[1]][pos[2]]);
							}
					ss << m_filename << fixed << "_f=" << m_FD_Samples.at(n) << "_arg";
				}
				VTK_File_Writer* w = m_Vtk_Dump_File->CloneEmpty();
				w->SetFilename(ss.str());
				w->AddVectorField(fieldName,field);
				if (w->Write()==false)
					++failed;
				delete w;
				Delete_N_3DArray(field,numLines);
			}
			if (failed)
				cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
		}
		return;
	}

	if (m_fileType==HDF5_FILETYPE)
	{
		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			stringstream ss;
			ss << "f" << n;
			size_t datasize[]={numLines[0],numLines[1],numLines[2]};
			if (m_HDF5_Dump_File->WriteVectorField(ss.str(), m_FD_Fields.at(n), datasize)==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;

			//legacy support, use /FieldData/FD frequency-Attribute in the future
			float freq[1] = {(float)m_FD_Samples.at(n)};
			if (m_HDF5_Dump_File->WriteAtrribute("/FieldData/FD/"+ss.str()+"_real","frequency",freq,1)==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
			if (m_HDF5_Dump_File->WriteAtrribute("/FieldData/FD/"+ss.str()+"_imag","frequency",freq,1)==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
		}
		return;
	}

	cerr << "ProcessFieldsFD::Process: unknown File-Type" << endl;
}
