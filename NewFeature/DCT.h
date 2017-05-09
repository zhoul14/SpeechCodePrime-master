#include "iostream"
#include "math.h"
#include <vector>
using namespace std;


class CDct
{
	float* m_pCosTab;
	int m_iDataLen;
public:
	CDct(int datalen);
	~CDct();
	void DoDct(float* inData, float* outData, int len);
private:

};
