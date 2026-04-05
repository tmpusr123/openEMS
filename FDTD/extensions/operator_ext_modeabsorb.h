#ifndef OPERATOR_EXT_MODEABSORB_H
#define OPERATOR_EXT_MODEABSORB_H

#include "FDTD/operator.h"
#include "operator_extension.h"

class CSPropModeAbsorb;
class CSPrimitives;

class Operator_Ext_ModeAbsorb : public Operator_Extension
{
	friend class Engine_Ext_ModeAbsorb;
public:
	Operator_Ext_ModeAbsorb(Operator* op);
	virtual ~Operator_Ext_ModeAbsorb();

	bool SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop);

	virtual Operator_Extension* Clone(Operator* op);
	virtual bool BuildExtension();
	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const { return std::string("Mode Absorber Extension"); }

protected:
	Operator_Ext_ModeAbsorb(Operator* op, Operator_Ext_ModeAbsorb* op_ext);
	CSPropModeAbsorb* m_prop;
	CSPrimitives* m_prim;
};

#endif // OPERATOR_EXT_MODEABSORB_H
