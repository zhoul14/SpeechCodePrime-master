#include "CFFT.h"
class EF:public CFFTanalyser{
private:
	long	FFT_LEN,FrameLen, SubBandNum, CepstrumNum;
	float	*hamWin;			//hamming window coefficients
	float	*FFTFrame;
public:
	EF(
		float sampleF,
		long	DefFFTLen=512,		//默认的FFT长度为512点
		long	DefFrameLen=320	//默认的帧长为320点
		);
	~EF();
	float DoEF(short* inData, float* outVect);




};