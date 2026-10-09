#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

#include <vector>
#include <utility>

#include "engine_cuda.h"
#include "operator.h"

//! CUDA FDTD-operator
class Operator_CUDA : public Operator
{
	friend class Engine_cuda;

public:
	static Operator_CUDA* New(unsigned int cuda_device_number = 0);

	virtual void setCUDAdevice(unsigned int cuda_device_number);

	virtual Engine* CreateEngine();

	//! Multi-GPU slab starts (size nslabs+1, last = number of x lines), or empty
	//! for the engine's own balanced split.
	const std::vector<int>& SlabStarts() const {return m_slab_starts;}

protected:
	unsigned int m_cuda_device_number;
	std::vector<int> m_slab_starts;

	//! Balanced x cuts for nslabs, moved out of every [lo,hi] span that must stay
	//! on one slab; false if that leaves a slab thinner than minPlanes.
	static bool ComputeSlabStarts(int nx, int nslabs, int minPlanes,
		const std::vector<std::pair<int,int> >& keep, std::vector<int>& starts);

};

#endif // OPERATOR_CUDA_H
