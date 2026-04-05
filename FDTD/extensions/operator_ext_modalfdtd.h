#ifndef OPERATOR_EXT_MODALFDTD_H
#define OPERATOR_EXT_MODALFDTD_H

#include "FDTD/operator.h"
#include "operator_extension.h"

class CSPropModeAbsorb;
class CSPrimitives;

class Operator_Ext_ModalFDTD : public Operator_Extension
{
	friend class Engine_Ext_ModalFDTD;
public:
	Operator_Ext_ModalFDTD(Operator* op);
	virtual ~Operator_Ext_ModalFDTD();

	bool SetInitParams(CSPrimitives* prim, CSPropModeAbsorb* prop);

	virtual Operator_Extension* Clone(Operator* op);
	virtual bool BuildExtension();
	virtual Engine_Extension* CreateEngineExtention();

	virtual std::string GetExtensionName() const { return std::string("Modal FDTD Extension"); }

protected:
	Operator_Ext_ModalFDTD(Operator* op, Operator_Ext_ModalFDTD* op_ext);
	CSPropModeAbsorb* m_prop;
	CSPrimitives* m_prim;
};

#endif // OPERATOR_EXT_MODALFDTD_H
