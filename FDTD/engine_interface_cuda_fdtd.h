/*

*	Copyright (C) 2025 Tommy Gu (radiotommy@gmail.com)
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

#ifndef ENGINE_INTERFACE_CUDA_FDTD_H
#define ENGINE_INTERFACE_CUDA_FDTD_H

#include "engine_interface_fdtd.h"
#include "operator_cuda.h"
#include "engine_cuda.h"

class Engine_Interface_CUDA_FDTD : public Engine_Interface_FDTD
{
public:
	Engine_Interface_CUDA_FDTD(Operator_CUDA* op);
	virtual ~Engine_Interface_CUDA_FDTD();

	virtual double CalcFastEnergy() const;

protected:
	Operator_CUDA* m_Op_CUDA;
	Engine_cuda* m_Eng_CUDA;
};

#endif // ENGINE_INTERFACE_SSE_FDTD_H
