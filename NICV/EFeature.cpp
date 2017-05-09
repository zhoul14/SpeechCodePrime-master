#include<windows.h>
#include "EFeature.h"
#include "EF.h"

EF ef(DEFAULT_SAMPLE_RATE,
	  DEFAULT_FFT_LEN,
	  DEFAULT_FRAME_LEN
	  );

bool getEFeature(short *SpeecBuffer,float *FeatureBuffer,float* EnergyBuff,long sampleNum, long FDim){
	long	i,FrameCounter,FrameNo;
	float	FrameEnergy,MaxEn,AveEn;
	long FrameNum = (sampleNum - FRAME_LEN + FRAME_STEP) / FRAME_STEP;

	//µÚ 0 ~ 13 Î¬´æ·Åµ¹Æ×£¬µÚ 14 ~ 27 Î¬´æ·ÅÒ»½×µ¹Æ×
	for( FrameNo = 0 ; FrameNo < FrameNum; FrameNo++ )
	{
		FrameEnergy = ef.DoEF(&SpeecBuffer[FrameNo * FRAME_STEP], FeatureBuffer + FrameNo * FDim);
		EnergyBuff[FrameNo] = FrameEnergy;
	}
	return true;
}