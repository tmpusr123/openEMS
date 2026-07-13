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


#include "engine_ext_united_upml.h"


Engine_Ext_United_UPML::Engine_Ext_United_UPML(std::vector<Operator_Ext_UPML *> *op_ext_upml_list) : Engine_Extension(NULL)
{
    m_Op_UPML_List = op_ext_upml_list;

}

Engine_Ext_United_UPML::~Engine_Ext_United_UPML()
{
    for (auto vflux : m_volt_fluxes) {
        delete vflux;
    }
    for (auto cflow : m_curr_fluxes) {
        delete cflow;
    }
#if WITH_CUDA
    if (m_d_coeff_table) cudaFree(m_d_coeff_table);
    for (auto idx : m_d_coeff_index)
        if (idx) cudaFree(idx);
    for (auto &a : mg_allocs) {
        cudaSetDevice(a.first);
        cudaFree(a.second);
    }
#endif
}


