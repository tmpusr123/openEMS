/*
*	GPU field-dump gather backend (CUDA engines implement this).
*
*	A field dump's interpolated output is a fixed sparse linear map of the
*	engine's raw volt/curr values (see Engine_Interface_FDTD::BuildFieldStencil).
*	ProcessFields builds that CSR stencil on the host and hands it to the engine
*	through this interface; the engine evaluates it with a gather kernel reading
*	the device field arrays directly -- no full readback, no material upload.
*
*	This header is CUDA-free so it can be included from Common/ code.
*/

#ifndef FIELD_GATHER_BACKEND_H
#define FIELD_GATHER_BACKEND_H

#include <vector>
#include <complex>
#include <cstddef>

//! Supplies the running-DFT weights of a frequency-domain dump to the engine.
//  The engine calls DeviceDFTFactors() whenever the device holds the fields of
//  timestep ts (every chunk finish). It returns true -- and fills one weight
//  per frequency -- exactly when the dump's Process() will take a sample at
//  ts. It must not have side effects: it is asked once per chunk, the dump's
//  Process() runs later (possibly while the next chunk already computes).
class FieldDFTClient
{
public:
	virtual ~FieldDFTClient() {}
	virtual bool DeviceDFTFactors(unsigned int ts, std::vector<std::complex<float> >& factors) const = 0;
};

class FieldGatherBackend
{
public:
	virtual ~FieldGatherBackend() {}

	//! Register a field-dump gather.
	//  CSR: offsets has nOut+1 entries; output o sums src[]/coeff[] over
	//  [offsets[o], offsets[o+1]).  nOut = 3*NI*NJ*NK.  useCurr selects the
	//  curr array (else volt).  The engine allocates a (pinned) host buffer of
	//  nOut floats, returns it via *hostOut, and refreshes it on every chunk
	//  finish where the device holds newly-computed fields (tagged by GetGatherTS).
	//  Returns a dump id (>=0), or -1 if the engine declines (e.g. multi-GPU,
	//  or the stencil exceeds the device-memory budget) -- caller then uses the
	//  host interpolation path.
	virtual int RegisterFieldGather(const std::vector<unsigned int>& offsets,
	                                const std::vector<unsigned int>& src,
	                                const std::vector<float>& coeff,
	                                bool useCurr, size_t nOut, float** hostOut) = 0;

	//! Timestep count whose fields the dump id's host buffer currently holds
	//! (-1 if not yet filled). Compare to GetNumberOfTimesteps() for freshness.
	virtual long GetGatherTS(int id) const = 0;

	//! Keep a running DFT of gather id on the device instead of handing the
	//! raw samples to the host. nFreq complex sums of nOut values are kept in
	//! device memory and advanced at every chunk finish for which the client
	//! reports a sample. Returns false if unsupported or out of memory; the
	//! caller then keeps the host DFT. Must be called before the first chunk.
	virtual bool EnableFieldDFT(int id, size_t nFreq, const FieldDFTClient* client) {(void)id;(void)nFreq;(void)client;return false;}

	//! Copy the device sums of frequency freq of gather id into host (nOut
	//! values, gather output order). Returns the number of samples the device
	//! accumulated, or -1 on error.
	virtual long ReadFieldDFT(int id, size_t freq, std::complex<float>* host) {(void)id;(void)freq;(void)host;return -1;}
};

#endif // FIELD_GATHER_BACKEND_H
