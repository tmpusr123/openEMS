#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

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

protected:
	unsigned int m_cuda_device_number;

};

#endif // OPERATOR_CUDA_H
