#include "DCT.h"

CDct::CDct(int datalen):
	m_iDataLen(datalen)
{
	m_pCosTab = new float[(datalen) * datalen];
	float pi_factor = (float)( asin(1.0)*2.0/(float)m_iDataLen);
	float mfnorm = (float)sqrt(2.0f/(float)m_iDataLen);
	for (int i = 0; i != m_iDataLen; i++)
	{
		for (int j = 0; j != m_iDataLen; j++)
		{
				float t = (float)i*pi_factor;
					m_pCosTab[i*(m_iDataLen)+j] = (float)cos(t*(j-0.5f))*mfnorm;
		}
	}
}
void CDct::DoDct(float* inData, float* outData, int len){
	float* pCosTable = m_pCosTab + m_iDataLen;//È¥µôµÚ0Î¬
	for (int i = 0; i != len; i++)
	{
		outData[i]=0.0f;
		for (int j = 1; j != m_iDataLen; j++)
		{
			outData[i] += inData[j]*pCosTable[j];
		}
		pCosTable = pCosTable + m_iDataLen;
	}
}

CDct::~CDct()
{
	delete[]m_pCosTab;
}