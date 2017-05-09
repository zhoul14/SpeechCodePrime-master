#include"GetNewFeature.h"
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "windows.h"
#include "common.h"
#include "CCodeBook.h"
#include "MFCC.h"
#include "SpeechFea.h"
#include"Fk.h"
#include "stdio.h"

 FEATURE *TelephoneMFCC(short *SpeechBuf,int SpeechSampleNum,int *FeaFrameNum);
 BOOL	getAdapMfcc(short *SpeecBuffer,float *FeatureBuffer,int FrameNum);
// FEATURE *TelephoneMFCC(short *SpeechBuf,int SpeechSampleNum,int *FeaFrameNum);
// BOOL	getAdapMfcc(short *SpeecBuffer,float *FeatureBuffer,int FrameNum);
GetNewFeature::GetNewFeature(short *sample,int sampleNum_in, bool d,int DefFFTLen,int DefFrameLen):CFFTanalyser(DefFFTLen)
{
	this->samples=sample;
	this->sampleNum=sampleNum_in;
	this->FFT_Len=DefFFTLen;
	this->FrameLen=DefFrameLen;
	this->dct = new CDct(MULTIFRAME_LEN);

	hamWin = new float[FrameLen];		
	FFTFrame = new float[FFT_Len];
	feature_tri=new float[N_DIM];
	FrameNum = (sampleNum - FRAME_LEN + FRAME_STEP) / FRAME_STEP;
	int i;
	float a =(float)( asin(1.0)*4/(FrameLen-1) );
	d48 = false;
	if (!d)
	{
		d42 = false;
	}

	for(i=0;i<FrameLen;i++)
		hamWin[i] = 0.54f - 0.46f * (float)cos(a*i);
	featureBuf = (float(*)[DIM+N_DIM])malloc(FrameNum * (DIM+N_DIM )* sizeof(float));
	featureMBuf = (float(*)[DIM+MIN_DIM])malloc(FrameNum * (DIM+MIN_DIM )* sizeof(float));

	if (featureBuf == NULL ) {
		printf("cannot malloc memory for FeatureBuf\n");
		exit(-1);
	}

	MakeMFCC();
	//MakeTeleMFCC();
	if (0)
	{
		MakeNewFeature(d48);
	}



}

GetNewFeature::~GetNewFeature()
{
	delete dct;
	delete []samples;
	delete []featureBuf;
	delete []featureMBuf;
	delete []hamWin;
	delete []FFTFrame;
	delete []feature_tri;
}

void GetNewFeature::MakeMFCC(){
	auto featureBuf_temp = (float(*)[DIM])malloc(FrameNum * DIM * sizeof(float));
	auto featureBuf_min = (float(*)[DIM+MIN_DIM])malloc(FrameNum * (DIM+MIN_DIM) * sizeof(float));

	if (featureBuf_temp == NULL ) {
		printf("cannot malloc memory for FeatureBuf_temp\n");
		exit(-1);
	}

	get20dBEnergyGeometryAveragMfcc_5ms(samples, featureBuf_temp, FrameNum);

	/*if(d48)
	{
		Make45in48(featureBuf_temp);
	}
	if(d42)
	{
		Make45in42(featureBuf_temp);
	}	*/
	mergeDctFt((float*)featureBuf_temp, FrameNum);
	free(featureBuf_temp);
	free(featureBuf_min);
}

void GetNewFeature::mergeDctFt(float *pTmp, int framenum)
{
	float* tmp = new float[(framenum) * 14 * DCT_LEN];//DCT特征，一共 14 * DCT_LEN 
	MakeMFDct(pTmp, tmp, framenum);
	int startIdx = (MULTIFRAME_LEN - 1)/2;

	for (int i = 0; i != framenum; i++)
	{
		int beginOff = 14;

		for (int j = 0; j != beginOff; j++)
		{
			featureBuf[i][j] = pTmp[j + i * (DIM)];
		}

		/*for (int j = 14 ; j != 14 * DCT_LEN;j++)
		{
			featureBuf[i][j] = tmp[(i - startIdx) * 14 * DCT_LEN + j - 14];
			if(featureBuf[i][j] == pTmp[j + i * DIM])
				printf("fucking same thing !!!\n");
		}*/
		/*for (int j = 14 ; j != 14 * DCT_LEN; j++)
		{
			featureBuf[i][j] = pTmp[j + i * DIM];
		}*/
		for (int j = beginOff; j != 14 * 2 + beginOff;j++)
		{
			featureBuf[i][j] = tmp[(i) * 14 * DCT_LEN + j - beginOff];
			if(featureBuf[i][j] == pTmp[j + i * DIM])
				printf("same thing !!!\n");
		}

		for (int j = 14 * 2 + beginOff; j != DIM+N_DIM; j++)
		{
			featureBuf[i][j] = pTmp[j - N_DIM + i * (DIM)];
		}
		
	}
	delete[] tmp;
}
void GetNewFeature::MakeMFDct(float *rawData, float* outData,int framenum, int framelen){
	float * tmp = NULL;
	float *inData = new float[framelen];
	int offset = 0;
	int startIdx = (MULTIFRAME_LEN - 1)/2;

	for (int i = 0; i != startIdx; i ++)
	{
		tmp = rawData;
		for (int j = 0; j != 14; j++)
		{
			for (int k =0; k != startIdx + i; k++)
			{
				inData[k] = *(tmp + k * DIM + j);
			}
			CDct c(startIdx + i);
			c.DoDct(inData, outData + (offset++) * DCT_LEN ,DCT_LEN);
		}
	}
	for(int i = startIdx; i != framenum - startIdx; i++){
		tmp = rawData + DIM * (i - startIdx);
		for(int j = 0; j != 14; j++){
			for (int k = 0; k != MULTIFRAME_LEN; k++)
			{
				inData[k] = *(tmp + k * DIM + j);
			}
			dct->DoDct(inData,outData + (offset++) * DCT_LEN, DCT_LEN);
		}
	}
	for (int i = framenum - startIdx; i != framenum; i ++)
	{
		tmp = rawData + DIM * (i - startIdx);
		for (int j = 0; j != 14; j++)
		{
			for (int k =0; k != framenum - i + startIdx; k++)
			{
				inData[k] = *(tmp + k * DIM + j);
			}
			CDct c(startIdx + framenum - i);
			c.DoDct(inData, outData + (offset++) * DCT_LEN ,DCT_LEN);
		}
	}
	delete []inData;
}


void GetNewFeature::MakeTeleMFCC()
{
	auto featureBuf_temp = (float(*)[DIM])malloc(FrameNum * DIM * sizeof(float));
	float* p_fsamples = new float[sampleNum];
	for (int i = 0; i != sampleNum; i++){
		p_fsamples[i] = samples[i];
	}
	getAdapMfcc(samples, (float*)featureBuf_temp, FrameNum);

	int i,j;
	for(i=0;i<FrameNum;i++)for(j=0;j<DIM;j++)
	featureBuf[i][j]=featureBuf_temp[i][j];

	delete []p_fsamples;
	free(featureBuf_temp);

}
void GetNewFeature::MakeNewFeature(bool fDim48){
	int i,j;
	
	for(i=0;i<FrameNum;i++){
		CalFeat(i);

		//for(j=0;j<N_DIM;j++)featureBuf[i][DIM+j]=feature_tri[j];
		for(j=0;j<N_DIM;j++)featureBuf[i][DIM+j]=feature_tri[j];

		//if (fDim48)
		//{
		//	continue;
		//}	
		/*for(j=N_DIM/2;j<N_DIM;j++)
		if(i==0)
		featureBuf[i][DIM+j]=0;
		else
		featureBuf[i][DIM+j]=featureBuf[i][DIM+j-N_DIM/2]-featureBuf[i-1][DIM+j-N_DIM/2];*/
	}
}


void GetNewFeature::Make45in48(float(*featureBuf_temp)[DIM]){
	int i,j;
	for(i=0;i<FrameNum;i++)for(j=0;j<DIM;j++)
	featureBuf[i][j]=featureBuf_temp[i][j];
}
void GetNewFeature::Make45in42(float(*feature_temp)[DIM]){
	int i,j;
	for(i=0;i<FrameNum;i++)for(j=0;j<DIM + MIN_DIM;j++)
		featureMBuf[i][j]=feature_temp[i][j];
}

void GetNewFeature::CalFeat(int idx){

#if 0
	int i;short *dataBegin=&samples[idx*FRAME_STEP];
	for(i=0;i<FrameLen;i++)
	{
		FFTFrame[i] = (float)(dataBegin[i]);
	}
	for( i= FrameLen-1; i > 0; i-- )	//计算语音信号差分并进行加窗
	{
		FFTFrame[i] -= FFTFrame[i-1]*PREE;
		FFTFrame[i] *= hamWin[i];
	}
	FFTFrame[0] *= (1.0f-PREE);
	FFTFrame[0] *= hamWin[0];
	for(i=FrameLen;i<FFT_Len;i++) FFTFrame[i]=0.0f;

	DoRealFFT(FFTFrame);

	int Wide_1 = SubbandBoundary / ( DEFAULT_SAMPLE_RATE / 512 );

	int Wide_2 = SubbandBoundary / ( DEFAULT_SAMPLE_RATE / 512) * 3;

	int Wide_3 = SubbandBoundary /  ( DEFAULT_SAMPLE_RATE / 512) / 2;

	float out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

	float energy_2500 = 0.0f, energy_6000 = 0.0f, energy = 0.0f;

	//能量归一化
	/*for ( i = 3; i < 4 * Wide_1; i++)
	energy_6000 += FFTFrame[i] * FFTFrame[i];
	energy_2500 = energy_6000;*/
	//for ( i; i < 9 * Wide_1; i++)
		//energy_6000 += FFTFrame[i] * FFTFrame[i];

	for(i=2;i<128;i++)energy+=FFTFrame[i]*FFTFrame[i];

	for (int i = 0; i < N_DIM; i++)
		feature_tri[i] = 0.0f;

	//for(int j=0;j<N_DIM;j++){
	int WideList[N_DIM] = { Wide_1, Wide_1*2.1, Wide_1*3.3, Wide_1*4.6, Wide_1 * 6.0};// Wide_1*4.2, Wide_1*5.4 };
	//int WideList[N_DIM] = { Wide_1, Wide_1*2, Wide_1*3}; 
	for (int j = 0; j < N_DIM; j++){

		//if (j)
		//{		
		//	WideList[j]+=WideList[j-1];
		//}
		for( int ii = 2; ii < Wide_1*(j+1); ii++){
			
			feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
		}
		if (j==3)
		{
			for (int ii=Wide_1*(j+1);ii < Wide_1*(j+1)+8;ii++)
			{
				feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
			}			
		}
		feature_tri[j]=log(feature_tri[j]/energy);//log

	}
	//for ( int j = N_dim_1; j < N_DIM; j++)
	//{
	//	for ( int ii = Wide_1 * (N_dim_1 + 1); ii < j * Wide_2 ; ii++)
	//	{
	//		feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
	//	}
	//	feature_tri[j]=log(feature_tri[j]/energy_7500);//log
	//}
	
#endif
}
float *GetNewFeature::GetFeatureBuff(){


	if (d42)
	{
		return GetFeatureMBuff();
	}
	float *features=(float*)featureBuf;

	float* fd = new float[FrameNum * (DIM+N_DIM)];
	
	for (int j = 0; j < FrameNum * (DIM+N_DIM); j++) {
	
		fd[j] = features[j];

	}

	return fd;
}
float *GetNewFeature::GetFeatureMBuff(){

	float *features=(float*)featureMBuf;

	float* fd = new float[FrameNum * (DIM+MIN_DIM)];

	for (int j = 0; j < FrameNum * (DIM+MIN_DIM); j++) {

		fd[j] = features[j];

	}

	return fd;
}


auto pCMfcc = CMFCC(DEFAULT_SAMPLE_RATE,
					DEFAULT_FFT_LEN,
					DEFAULT_FRAME_LEN,
					DEFAULT_SUB_BAND_NUM,
					DEFAULT_CEP_COEF_NUM);
float	aa[5] = {-0.75f, -0.375f, 0.0f, 0.375f, 0.75f};


FEATURE		g_FeatureBuf[500];	//5S @16KHz


BOOL getAdapMfcc(short *SpeecBuffer,float *FeatureBuffer,int FrameNum)
{
	float	MaxEn,enrg,FrameEnergy;
	long	j,FrameCounter;
	long	FrameNo;

	//第 0 ~ 13 维存放倒谱，第 14 ~ 27 维存放一阶倒谱

	for(FrameNo = 0 ; FrameNo < FrameNum; FrameNo++)
	{
		//第 0 ~ 13 维存放倒谱
		FrameEnergy = pCMfcc.DoSSMFCC( &SpeecBuffer[FrameNo * DEFAULT_FRAME_STEP], &FeatureBuffer[FrameNo * DIM] );	
		FeatureBuffer[FrameNo*DIM+3*D] = (float)( 10*log10(FrameEnergy) );
	}

	//计算头两帧和最后两帧的一阶差分倒谱
	/* margin cep 1 */
	for( j = 0; j < D; j++ )
	{
		FeatureBuffer[D+j]=(float)(2.0*(aa[1]*FeatureBuffer[j]+aa[3]*FeatureBuffer[DIM+j]));			//第0帧
		FeatureBuffer[DIM+D+j]=(float)(2.0*(aa[1]*FeatureBuffer[j]+aa[3]*FeatureBuffer[2*DIM+j]));	//第1帧
		FeatureBuffer[(FrameNum-2)*DIM+D+j]=(float)(2.0*(aa[1]*FeatureBuffer[(FrameNum-3)*DIM+j]+aa[3]*FeatureBuffer[(FrameNum-1)*DIM+j]));
		FeatureBuffer[(FrameNum-1)*DIM+D+j]=(float)(2.0*(aa[1]*FeatureBuffer[(FrameNum-2)*DIM+j]+aa[3]*FeatureBuffer[(FrameNum-1)*DIM+j]));
	}
	/* frame cep 1 */
	for(FrameNo=2;FrameNo<FrameNum-2;FrameNo++) 
	{
		for( j = 0; j < D; j++ )
			FeatureBuffer[FrameNo*DIM+D+j]=
			aa[0]*FeatureBuffer[(FrameNo-2)*DIM+j] +
			aa[1]*FeatureBuffer[(FrameNo-1)*DIM+j] +
			aa[3]*FeatureBuffer[(FrameNo+1)*DIM+j] +
			aa[4]*FeatureBuffer[(FrameNo+2)*DIM+j];
	}

	/************************************/
	//计算头两帧和最后两帧的二阶差分倒谱
	/* margin cep 2 */
	for( j = 0; j < D; j++ )
	{
		FeatureBuffer[2*D+j]=(float)(2.0*(aa[1]*FeatureBuffer[D+j]+aa[3]*FeatureBuffer[DIM+D+j]));
		FeatureBuffer[DIM+2*D+j]=(float)(2.0*(aa[1]*FeatureBuffer[D+j]+aa[3]*FeatureBuffer[2*DIM+D+j]));
		FeatureBuffer[(FrameNum-2)*DIM+2*D+j]=(float)(2.0*(aa[1]*FeatureBuffer[(FrameNum-3)*DIM+D+j]+aa[3]*FeatureBuffer[(FrameNum-1)*DIM+D+j]));
		FeatureBuffer[(FrameNum-1)*DIM+2*D+j]=(float)(2.0*(aa[1]*FeatureBuffer[(FrameNum-2)*DIM+D+j]+aa[3]*FeatureBuffer[(FrameNum-1)*DIM+D+j]));
	}
	/* FrameNo cep 2 */
	for(FrameNo=2;FrameNo<FrameNum-2;FrameNo++) 
	{
		for( j = 0; j < D; j++ )
			FeatureBuffer[FrameNo*DIM+2*D+j] = 
			aa[0]*FeatureBuffer[(FrameNo-2)*DIM+D+j] +
			aa[1]*FeatureBuffer[(FrameNo-1)*DIM+D+j] +
			aa[3]*FeatureBuffer[(FrameNo+1)*DIM+D+j] +
			aa[4]*FeatureBuffer[(FrameNo+2)*DIM+D+j];
	}
	/************************************/
	/* frame energy 0 log*/

	MaxEn = -100;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{
		if( FeatureBuffer[FrameNo*DIM+3*D] > MaxEn )
			MaxEn = FeatureBuffer[FrameNo*DIM+3*D];
	}
	enrg		 = 0.0;
	FrameCounter = 0;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{
		if( FeatureBuffer[FrameNo*DIM+3*D] > MaxEn -20 )
		{
			enrg += FeatureBuffer[FrameNo*DIM+3*D];
			FrameCounter++;
		}
	}
	enrg = enrg/FrameCounter;

	//	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )			//2001,4,19改
	//	{
	//		enrg += FeatureBuffer[FrameNo*DIM+3*D];
	//		FeatureBuffer[FrameNo*DIM+3*D] = (float)(log(FeatureBuffer[FrameNo*DIM+3*D]));
	//	}
	//	enrg = (float)(log(enrg/FrameNum));

	/* margin energy 1 */
	FeatureBuffer[3*D+1]		= (float)(0.8*(aa[1]*FeatureBuffer[3*D] + aa[3]*FeatureBuffer[DIM+3*D]));
	FeatureBuffer[DIM+3*D+1]	= (float)(0.8*(aa[1]*FeatureBuffer[3*D] + aa[3]*FeatureBuffer[2*DIM+3*D]));
	FeatureBuffer[(FrameNum-2)*DIM+3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[(FrameNum-3)*DIM+3*D] + aa[3]*FeatureBuffer[(FrameNum-1)*DIM+3*D]));
	FeatureBuffer[(FrameNum-1)*DIM+3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[(FrameNum-2)*DIM+3*D] + aa[3]*FeatureBuffer[(FrameNum-1)*DIM+3*D]));
	/*   frame energy 1     */
	for(FrameNo=2;FrameNo<FrameNum-2;FrameNo++)
		FeatureBuffer[FrameNo*DIM+3*D+1] = 
		(float)(0.4*(aa[0]*FeatureBuffer[(FrameNo-2)*DIM+3*D]
	+ aa[1]*FeatureBuffer[(FrameNo-1)*DIM+3*D]
	+ aa[3]*FeatureBuffer[(FrameNo+1)*DIM+3*D]
	+ aa[4]*FeatureBuffer[(FrameNo+2)*DIM+3*D]));
	/*  frame energy normalize */
	for(FrameNo=0;FrameNo<FrameNum;FrameNo++)
		FeatureBuffer[FrameNo*DIM+3*D] = (float)(0.5*(FeatureBuffer[FrameNo*DIM+3*D]-enrg));

	/* Margin  energy  2*/
	FeatureBuffer[3*D+2] = (float)(1.5*(FeatureBuffer[DIM+3*D+1]-FeatureBuffer[3*D+1]));
	FeatureBuffer[(FrameNum-1)*DIM+3*D+2] = (float)(1.5*(FeatureBuffer[(FrameNum-1)*DIM+3*D+1] - FeatureBuffer[(FrameNum-2)*DIM+3*D+1]));
	/* Frame energy    2 */
	for(FrameNo=1;FrameNo<FrameNum-1;FrameNo++)
		FeatureBuffer[FrameNo*DIM+3*D+2] = (float)(1.5*(FeatureBuffer[(FrameNo+1)*DIM+3*D+1] - FeatureBuffer[(FrameNo-1)*DIM+3*D+1]));

	return TRUE;
}




FEATURE *TelephoneMFCC(short *SpeechBuf,long SpeechSampleNum,long *FeaFrameNum)
{
	static FEATURE		*FeatureBuf=0;
	long		FrameNum;

	//计算语音特征的帧数	
	FrameNum = ( SpeechSampleNum-DEFAULT_FRAME_LEN+DEFAULT_FRAME_STEP )/DEFAULT_FRAME_STEP;	
	FeatureBuf = new FEATURE[FrameNum];							//为特征分配内存空间
	getAdapMfcc(SpeechBuf,&FeatureBuf[0][0],FrameNum);
	FeaFrameNum[0]=FrameNum;
	return(FeatureBuf);

}