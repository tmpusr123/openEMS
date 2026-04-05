#include "operator_ext_modalfdtd.h"
#include "engine_ext_modalfdtd.h"
#include "CSPropModeAbsorb.h"

Operator_Ext_ModalFDTD::Operator_Ext_ModalFDTD(Operator* op) : Operator_Extension(op)
{
	m_prop = NULL;
	m_prim = NULL;
}

Operator_Ext_ModalFDTD::Operator_Ext_ModalFDTD(Operator* op, Operator_Ext_ModalFDTD* op_ext) : Operator_Extension(op, op_ext)
{
	m_prop = op_ext->m_prop;
	m_prim = op_ext->m_prim;
}

Operator_Ext_ModalFDTD::~Operator_Ext_ModalFDTD()
{
}

bool Operator_Ext_ModalFDTD::SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop)
{
	m_prim = prim;
	m_prop = prop;
	return (prim != NULL && prop != NULL);
}

Operator_Extension* Operator_Ext_ModalFDTD::Clone(Operator* op)
{
	return new Operator_Ext_ModalFDTD(op, this);
}

bool Operator_Ext_ModalFDTD::BuildExtension()
{
	// Stub: not yet implemented
	return true;
}

Engine_Extension* Operator_Ext_ModalFDTD::CreateEngineExtention()
{
	return new Engine_Ext_ModalFDTD(this);
}
