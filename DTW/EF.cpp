/* MFCC: power spectrum --> magnitude spectrum --> cepstral coefficient */
//#include <iostream.h>
#include <math.h>
#include "EF.h"
#include "windows.h"
#define PREE        0.98f
#define MELFLOOR    1.0f
#define BEGINIDX 2
EF::EF(float sampleF, long DefFFTlen, long DefFrameLen):CFFTanalyser(DefFFTlen){

	int i;
	float a;
	FrameLen	= DefFrameLen;

	FFT_LEN = GetFFTAnalyseLen();
	hamWin = new float[FrameLen];						//hamming 窗系数
	FFTFrame = new float[FFT_LEN];
	a =(float)( asin(1.0)*4/(FrameLen-1) );
	for(i=0;i<FrameLen;i++)
		hamWin[i] = 0.54f - 0.46f * (float)cos(a*i);

}
EF::~EF(){
	delete []hamWin;
	delete []FFTFrame;
}

float EF::DoEF(short* inData, float* outVect){
	int i;
	float	FrameEnergy,ek,temp1,temp2,FFrameEnergy = 0.0f;

	FrameEnergy = 0.0f;
	for(i=0;i<FrameLen;i++)
	{
		FFTFrame[i] = (float)inData[i];
		FrameEnergy += (FFTFrame[i]*FFTFrame[i]);
	}
	for(i=FrameLen-1;i>0;i--)	//计算语音信号差分并进行加窗
	{
		FFTFrame[i] -= FFTFrame[i-1]*PREE;
		FFTFrame[i] *= hamWin[i];
	}
	FFTFrame[0] *= 1.0f-PREE;
	FFTFrame[0] *= hamWin[0];
	for(i=FrameLen;i<FFT_LEN;i++) FFTFrame[i]=0.0f;
	DoRealFFT(FFTFrame);
	for( i=0; i<FFT_LEN/2; i++ )		//the component at zero frequence is discarded （是否要扔掉）
	{
		temp1 = FFTFrame[2*i]; temp2 = FFTFrame[2*i+1];
		FFrameEnergy +=  sqrt(temp1*temp1+temp2*temp2);// 计算第k点频率处的能量
	}
	for (i = BEGINIDX; i < FFT_LEN/4 ; i++)
	{
		temp1 = FFTFrame[2*i]; temp2 = FFTFrame[2*i+1];
		ek = temp1*temp1+temp2*temp2;	// 计算第k点频率处的能量
		outVect[i - BEGINIDX] = sqrt(ek)/FFrameEnergy/2; //计算是否加平方改进
	}
	return(FFrameEnergy*2);			//返回当前帧的能量
 
}