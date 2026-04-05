#include "engine_ext_modalfdtd.h"
#include "operator_ext_modalfdtd.h"

Engine_Ext_ModalFDTD::Engine_Ext_ModalFDTD(Operator_Ext_ModalFDTD* op_ext) : Engine_Extension(op_ext)
{
	m_Op_Ext = op_ext;
}

Engine_Ext_ModalFDTD::~Engine_Ext_ModalFDTD()
{
}
