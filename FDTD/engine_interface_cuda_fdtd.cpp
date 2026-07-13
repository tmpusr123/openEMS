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

#include "engine_interface_cuda_fdtd.h"

Engine_Interface_CUDA_FDTD::Engine_Interface_CUDA_FDTD(Operator_CUDA* op) : Engine_Interface_FDTD(op)
{
	m_Op_CUDA= op;
	m_Eng_CUDA = dynamic_cast<Engine_cuda*>(m_Op_CUDA->GetEngine());
	if (m_Eng_CUDA==NULL)
	{
		cerr << "Engine_Interface_SSE_FDTD::Engine_Interface_SSE_FDTD: Error: SSE-Engine is not set! Exit!" << endl;
		exit(1);
	}
}

Engine_Interface_CUDA_FDTD::~Engine_Interface_CUDA_FDTD()
{
	m_Op_CUDA = NULL;
	m_Eng_CUDA = NULL;
}

double Engine_Interface_CUDA_FDTD::CalcFastEnergy() const
{
	return m_Eng_CUDA->CalcFastEnergy();
}
