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

#ifndef PROCESSFIELDS_FD_H
#define PROCESSFIELDS_FD_H

#include "processfields.h"

#include "field_gather_backend.h"

class ProcessFieldsFD : public ProcessFields, public FieldDFTClient
{
public:
	ProcessFieldsFD(Engine_Interface_Base* eng_if);
	virtual ~ProcessFieldsFD();

	virtual std::string GetProcessingName() const {return "frequency domain field dump";}

	virtual void InitProcess();

	virtual int Process();
	virtual void PostProcess();

	virtual bool DeviceDFTFactors(unsigned int ts, std::vector<std::complex<float> >& factors) const;

protected:
	virtual void DumpFDData();

	//! weight of the sample taken at time T, per frequency (shared by the host
	//! and the device DFT so both use bit-identical factors)
	std::complex<float> DFTFactor(size_t n, double T) const;

	// --- device-resident running DFT (CUDA engine) -------------------------
	void SetupDeviceDFT();
	void FetchDeviceDFT();
	bool m_device_dft_tried;
	bool m_device_dft;          //!< sums live on the device (gather id m_gpu_dump_id)
	unsigned int m_FD_SampleCount_ts0; //!< samples taken at ts==0 (zero fields, never sent to the device)

	//! frequency domain field storage
	std::vector<std::complex<float>****> m_FD_Fields;
};

#endif // PROCESSFIELDS_FD_H
