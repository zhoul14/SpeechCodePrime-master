#include"CodeConv.h"
#include <iostream>
int cbsToxxcb(std::string filename, void* cbs){
	GMMCodebookSet* cbset = (GMMCodebookSet*)cbs;
	FILE	*fp=NULL;

	fopen_s(&fp,filename.c_str(),"wb");
	if(fp == NULL)
		return false;
	//fread(pCodeBook,sizeof(CODEBOOK),CodeBookNum,fp);
	CODEBOOK xxcb;
	for (int i = 0; i < 100; i++)
	{
		for (int t = 0; t < 2; t++)
		{
			GMMCodebook cb = cbset->getCodebook(i + t * 100);
			std::cout<< i + t *100<<",";
			for (int j = 0; j  < FEATURE_DIM; j ++)
			{
				xxcb.MeanU[j] =  cb.Mu[j];
				for (int k = 0; k < FEATURE_DIM; k++)
				{
					xxcb.InvR[j][k] = cb.InvSigma[j * FEATURE_DIM + k];
				}		
			}
			xxcb.DetR = cb.DETSIGMA;
			xxcb.DurationMean = cb.DurMean;
			xxcb.DurationVar = cb.DurVar;
			fwrite(&xxcb, sizeof(CODEBOOK),1,fp);
		}
	}

	for (int i = 0; i < 164; i++)
	{
		for (int t = 0; t < 4; t++)
		{
			GMMCodebook cb = cbset->getCodebook(i + t * 164 + 200);
			std::cout<< i + t * 164 + 200<<",";

			for (int j = 0; j  < FEATURE_DIM; j ++)
			{
				xxcb.MeanU[j] =  cb.Mu[j];
				for (int k = 0; k < FEATURE_DIM; k++)
				{
					xxcb.InvR[j][k] = cb.InvSigma[j * FEATURE_DIM + k];
				}		
			}
			xxcb.DetR = cb.DETSIGMA;
			xxcb.DurationMean = cb.DurMean;
			xxcb.DurationVar = cb.DurVar;
			fwrite(&xxcb, sizeof(CODEBOOK),1,fp);
		}
	}

	fclose(fp);

	return true;

}