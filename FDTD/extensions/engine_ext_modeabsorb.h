#ifndef ENGINE_EXT_MODEABSORB_H
#define ENGINE_EXT_MODEABSORB_H

#include "engine_extension.h"

class Operator_Ext_ModeAbsorb;

class Engine_Ext_ModeAbsorb : public Engine_Extension
{
public:
	Engine_Ext_ModeAbsorb(Operator_Ext_ModeAbsorb* op_ext);
	virtual ~Engine_Ext_ModeAbsorb();

	virtual void DoPreVoltageUpdates() {}
	virtual void DoPostVoltageUpdates() {}
	virtual void DoPreCurrentUpdates() {}
	virtual void DoPostCurrentUpdates() {}

protected:
	Operator_Ext_ModeAbsorb* m_Op_Ext;
};

#endif // ENGINE_EXT_MODEABSORB_H
