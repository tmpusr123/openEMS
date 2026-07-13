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
#include <cstddef>

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
};

#endif // FIELD_GATHER_BACKEND_H
