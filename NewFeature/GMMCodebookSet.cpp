#include "GMMCodebookSet.h"
#include "../CommonLib/Math/MathUtility.h"
#include <cassert>
#include "iostream"
GMMCodebook GMMCodebookSet::getCodebook(int num)
{
	if (num >= CodebookNum)
	{
		printf("num not in range: 0 - %d\n", CodebookNum - 1);
		exit(-1);
		return GMMCodebook(0, 0, cbType);
	}
	else
	{
		int L = SigmaL();
		GMMCodebook cb(MixNum, FDim, cbType);
		memcpy(cb.Mu, Mu + num * MixNum * FDim, MixNum * FDim * sizeof(double));
		memcpy(cb.InvSigma, InvSigma + num * L, L * sizeof(double));
		memcpy(cb.Alpha, Alpha + num * MixNum, MixNum * sizeof(double));
		cb.DurMean = DurMean[num];
		cb.DurVar = DurVar[num];
		cb.DETSIGMA = DetSigma[num];
		return cb;

	}
}
//
//void GMMCodebookSet::printCodebookSetToFile(const char* filename)
//{
//	printf("Printing CodebookSet\n");
//	FILE* fid = fopen(filename, "w");
//	for (int i = 0; i < CodebookNum; i++)
//	{
//		GMMCodebook c (getCodebook(i));
//
//		fprintf(fid, "GMMCODEBOOK_NO_%d", i);
//		if (c.cbType == CB_TYPE_FULL_RANK)
//			fprintf(fid, " FULL\n");
//		else if (c.cbType == CB_TYPE_DIAG)
//			fprintf(fid, " DIAG\n");
//		else if (c.cbType == CB_TYPE_FULL_RANK_SHARE)
//			fprintf(fid, " FULL_SHARED\n");
//		else if (c.cbType == CB_TYPE_FULL_MIX)
//			fprintf(fid, " FULL_MIX\n");
//
//		fprintf(fid, "DURMEAN\n%f\n", c.DurMean);
//		fprintf(fid, "DURSIGMA\n%f\n", sqrt(c.DurVar));
//		fprintf(fid, "DETSIGMA\n\%f\n", c.DETSIGMA);
//
//		fprintf(fid, "ALPHA\n");
//		for (int j = 0; j < MixNum; j++)
//			fprintf(fid, "%f\t", c.Alpha[j]);
//		fprintf(fid, "\n");
//
//		for (int j = 0; j < MixNum; j++)
//		{
//			fprintf(fid, "MU_MIX_%d\n", j);
//			for (int k = 0; k < FDim; k++)
//				fprintf(fid, "%f\t", c.Mu[j * FDim + k]);
//			fprintf(fid, "\n");
//		}
//
//		if (c.cbType == CB_TYPE_FULL_MIX)
//		{
//			fprintf(fid, "BETA_%d\n",BetaNum);
//			for (int j = 0; j < BetaNum;j++)
//			{
//				fprintf(fid,"%f\t", c.Beta[j]);
//			}
//			fprintf(fid,"\n");
//		}
//
//		if (c.cbType != CB_TYPE_FULL_RANK_SHARE) {
//			for (int j = 0; j < MixNum; j++) {
//				fprintf(fid, "INV_SIGMA_MIX_%d\n", j);
//				if (c.cbType == CB_TYPE_FULL_RANK) {
//					for (int k = 0; k < FDim; k++) {
//						for (int l = 0; l < FDim; l++)
//							fprintf(fid, "%f\t", c.InvSigma[j * FDim * FDim + k * FDim + l]);
//						fprintf(fid, "\n");
//					}
//				} else if (c.cbType == CB_TYPE_DIAG) {
//					for (int k = 0; k < FDim; k++)
//						fprintf(fid, "%f\t", c.InvSigma[j * FDim + k]);
//					fprintf(fid, "\n");
//				} else if(c.cbType == CB_TYPE_FULL_MIX){
//					for (int k = 0; k < FEATURE_DIM; k++)
//					{
//						for (int l = 0; l < FEATURE_DIM; l++)
//						{
//							fprintf(fid, "%f\t", c.InvSigma[j * c.SigmaL() + k * FEATURE_DIM + l]);
//						}
//					}
//					fprintf(fid, "\n");
//					for (int k =  0; k < FDim - FEATURE_DIM; k++)
//					{
//						for (int l = 0; l < FDim - FEATURE_DIM; l++)
//						{
//							fprintf(fid, "%f\t", c.InvSigma[j * c.SigmaL() + k * (FDim - FEATURE_DIM) + l + FEATURE_DIM * FEATURE_DIM]);
//						}
//					}
//					fprintf(fid, "\n");
//
//				}
//
//
//			}
//		} else {
//			fprintf(fid, "INV_SIGMA_SHARED\n");
//			for (int j = 0; j < FDim; j++) {
//				for (int k = 0; k < FDim; k++) {
//					fprintf(fid, "%f\t", c.InvSigma[j * FDim + k]);
//				}
//				fprintf(fid, "\n");
//			}
//		}
//
//		fprintf(fid, "\n");
//
//	}
//	fclose(fid);
//	return;
//}
//

GMMCodebookSet::GMMCodebookSet(const char* initBuf, int mode) {
	if (mode == INIT_MODE_FILE) {
		initializeFromOutside = false;
		FILE* fid = fopen(initBuf, "rb");
		if (!fid) {
			printf("cannot open gmm codebook file[%s]", initBuf);;
			exit(-1);
		}

		fread(&CodebookNum, sizeof(int), 1, fid);
		fread(&FDim, sizeof(int), 1, fid);
		fread(&MixNum, sizeof(int), 1, fid);
		fread(&cbType, sizeof(int), 1, fid);

		int L = SigmaL();
		int M = isSharedSigma() ? 1 : MixNum;
		Mu = new double[CodebookNum * MixNum * FDim];
		InvSigma = new double[CodebookNum * L];
		Alpha = new double[CodebookNum * MixNum];
		DurMean = new double[CodebookNum];
		DurVar = new double[CodebookNum];
		LogDurVar = new double[CodebookNum];
		DetSigma = new double[CodebookNum * M];

		Beta = NULL;
		BetaNum = 0;

		fread(Alpha, sizeof(double), MixNum * CodebookNum, fid);
		fread(Mu, sizeof(double), MixNum * CodebookNum * FDim, fid);
		fread(InvSigma, sizeof(double), CodebookNum * L, fid);
		fread(DurMean, sizeof(double), CodebookNum, fid);
		fread(DurVar, sizeof(double), CodebookNum, fid);
		fread(LogDurVar, sizeof(double), CodebookNum, fid);
		fread(DetSigma, sizeof(double), CodebookNum * M, fid);

		fclose(fid);
	} else if (mode == INIT_MODE_MEMORY) {
		initializeFromOutside = true;
		const char* p = initBuf;
		memcpy(&CodebookNum, p, sizeof(int));
		p += sizeof(int);
		memcpy(&FDim, p, sizeof(int));
		p += sizeof(int);
		memcpy(&MixNum, p, sizeof(int));
		p += sizeof(int);
		memcpy(&cbType, p, sizeof(int));
		p += sizeof(int);


		int L = SigmaL();
		Alpha = (double*)p;
		Mu = Alpha + MixNum * CodebookNum;
		InvSigma = Mu + MixNum * CodebookNum * FDim;
		DurMean = InvSigma + CodebookNum * L;
		DurVar = DurMean + CodebookNum;
		LogDurVar = DurVar + CodebookNum;
		DetSigma = LogDurVar + CodebookNum;

	}

}

int GMMCodebookSet::getCodebookNum() const
{
	return CodebookNum;
}

int GMMCodebookSet::getFDim() const
{
	return FDim;
}

int GMMCodebookSet::getMixNum() const
{
	return MixNum;
}


GMMCodebookSet::GMMCodebookSet(int cbnum, int fdim, int mixnum, int cbType)
{
	CodebookNum = cbnum;
	FDim = fdim;
	MixNum = mixnum;
	this->cbType = cbType;
	initializeFromOutside = false;

	int L = SigmaL();
	Mu = new double[CodebookNum * MixNum * FDim];
	InvSigma = new double[CodebookNum * L];
	Alpha = new double[CodebookNum * MixNum];
	DurMean = new double[CodebookNum];
	DurVar = new double[CodebookNum];
	LogDurVar = new double[CodebookNum];


	if (cbType == CB_TYPE_FULL_RANK_SHARE)
		DetSigma = new double[CodebookNum];
	else 
	{
		DetSigma = new double[CodebookNum * MixNum];
	}

	memset(DurVar, 0, CodebookNum * sizeof(double));
	memset(LogDurVar, 0, CodebookNum * sizeof(double));
	memset(DurMean, 0, CodebookNum * sizeof(double));
	memset(Mu,0,CodebookNum * MixNum * FDim * sizeof(double));
	memset(Alpha,0,CodebookNum*MixNum *  sizeof(double));
	for(int ii=0;ii<CodebookNum;ii++)
	{
		for (int jj = 0; jj < MixNum; jj++)
		{
			Alpha[ii*jj] = 1 / MixNum;
		}
		for (int jj = 0; jj < L; jj++)
		{
			InvSigma[ii*jj] = 1;			
		}
		DetSigma[ii]=1;
	}
	return;
}

int GMMCodebookSet::getCbType() const {
	return cbType;

}

bool GMMCodebookSet::isFullRank() const {
	return cbType != CB_TYPE_DIAG ;
}



GMMCodebookSet::~GMMCodebookSet()
{
	if (!initializeFromOutside) {
		delete [] Mu;
		delete [] InvSigma;
		delete [] Alpha;
		delete [] DurMean;
		delete [] DurVar;
		delete [] LogDurVar;
		delete [] DetSigma;
		if (Beta)
		{
			delete []Beta;
		}
	}

}