#include "operator_ext_modeabsorb.h"
#include "engine_ext_modeabsorb.h"
#include "CSPropModeAbsorb.h"

Operator_Ext_ModeAbsorb::Operator_Ext_ModeAbsorb(Operator* op) : Operator_Extension(op)
{
	m_prop = NULL;
	m_prim = NULL;
}

Operator_Ext_ModeAbsorb::Operator_Ext_ModeAbsorb(Operator* op, Operator_Ext_ModeAbsorb* op_ext) : Operator_Extension(op, op_ext)
{
	m_prop = op_ext->m_prop;
	m_prim = op_ext->m_prim;
}

Operator_Ext_ModeAbsorb::~Operator_Ext_ModeAbsorb()
{
}

bool Operator_Ext_ModeAbsorb::SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop)
{
	m_prim = prim;
	m_prop = prop;
	return (prim != NULL && prop != NULL);
}

Operator_Extension* Operator_Ext_ModeAbsorb::Clone(Operator* op)
{
	return new Operator_Ext_ModeAbsorb(op, this);
}

bool Operator_Ext_ModeAbsorb::BuildExtension()
{
	// Stub: not yet implemented
	return true;
}

Engine_Extension* Operator_Ext_ModeAbsorb::CreateEngineExtention()
{
	return new Engine_Ext_ModeAbsorb(this);
}
