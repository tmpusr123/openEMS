#include "engine_ext_modeabsorb.h"
#include "operator_ext_modeabsorb.h"

Engine_Ext_ModeAbsorb::Engine_Ext_ModeAbsorb(Operator_Ext_ModeAbsorb* op_ext) : Engine_Extension(op_ext)
{
	m_Op_Ext = op_ext;
}

Engine_Ext_ModeAbsorb::~Engine_Ext_ModeAbsorb()
{
}
