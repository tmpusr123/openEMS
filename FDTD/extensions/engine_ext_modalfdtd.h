#ifndef ENGINE_EXT_MODALFDTD_H
#define ENGINE_EXT_MODALFDTD_H

#include "engine_extension.h"

class Operator_Ext_ModalFDTD;

class Engine_Ext_ModalFDTD : public Engine_Extension
{
public:
	Engine_Ext_ModalFDTD(Operator_Ext_ModalFDTD* op_ext);
	virtual ~Engine_Ext_ModalFDTD();

	virtual void DoPreVoltageUpdates() {}
	virtual void DoPostVoltageUpdates() {}
	virtual void DoPreCurrentUpdates() {}
	virtual void DoPostCurrentUpdates() {}

protected:
	Operator_Ext_ModalFDTD* m_Op_Ext;
};

#endif // ENGINE_EXT_MODALFDTD_H
