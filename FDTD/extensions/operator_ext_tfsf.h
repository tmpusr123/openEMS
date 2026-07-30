/*
*	Copyright (C) 2012 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef OPERATOR_EXT_TFSF_H
#define OPERATOR_EXT_TFSF_H

#include "operator_extension.h"
#include "FDTD/operator.h"
#include "tools/constants.h"

class Excitation;
class CSPropExcitation;

class Operator_Ext_TFSF : public Operator_Extension
{
	friend class Engine_Ext_TFSF;
public:
	//! Create a TFSF extension. If \a pw_prop is given, this instance is bound
	//! to that single plane-wave (exc_type 10) excitation property; multiple
	//! bound instances superpose additively (e.g. two quadrature-delayed
	//! components forming a circular polarization). With pw_prop==NULL the
	//! legacy behavior applies: scan for the first plane-wave excitation.
	Operator_Ext_TFSF(Operator* op, CSPropExcitation* pw_prop=NULL);
	~Operator_Ext_TFSF();

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();

	virtual Engine_Extension* CreateEngineExtention(Engine *engine);

	virtual bool IsCylinderCoordsSave(bool closedAlpha, bool R0_included) const {UNUSED(closedAlpha); UNUSED(R0_included); return false;}
	virtual bool IsCylindricalMultiGridSave(bool child) const {UNUSED(child); return false;}

	// FIXME, this extension is not save to use with MPI
	virtual bool IsMPISave() const {return false;}

	virtual string GetExtensionName() const {return string("Total-Field/Scattered-Field Extension");}

	virtual void ShowStat(ostream &ostr) const;

	virtual void Init();
	virtual void Reset();

protected:
	Excitation* m_Exc;

	//! plane-wave excitation property this instance is bound to (NULL = legacy scan)
	CSPropExcitation* m_PW_Prop;

	//! signal delay of this source in timesteps (from the property's delay),
	//! baked into every per-cell injection delay
	double m_SignalDelayTS;

	bool m_IncLow[3];
	bool m_ActiveDir[3][2]; // m_ActiveDir[direction][low/high]
	unsigned int m_Start[3];
	unsigned int m_Stop[3];
	unsigned int m_numLines[3];

	double m_PropDir[3];
	double m_E_Amp[3];
	double m_H_Amp[3];

	double m_Frequency;
	double m_PhVel;

	unsigned int m_maxDelay;

	// array setup [direction][low/high][component][ <mesh_position> ]
	unsigned int* m_VoltDelay[3][2][2];
	FDTD_FLOAT* m_VoltDelayDelta[3][2][2];
	FDTD_FLOAT* m_VoltAmp[3][2][2];

	unsigned int* m_CurrDelay[3][2][2];
	FDTD_FLOAT* m_CurrDelayDelta[3][2][2];
	FDTD_FLOAT* m_CurrAmp[3][2][2];

};

#endif // OPERATOR_EXT_TFSF_H
