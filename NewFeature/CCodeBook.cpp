
#include <stdio.h>
#include <windows.h>
#include <math.h>

#include "common.h"
#include "CCodeBook.h"
#include "WordCodeBook6HmmState.h"

CCODE_BOOK::CCODE_BOOK(long SetCodeBookNum,long SetTotalWordNum)
{
long	WordNo,HmmStateNo;

	Log2PI =log( asin(1.0)*4 );

	CodeBookNum		= SetCodeBookNum;
	TotalWordNum	= SetTotalWordNum;
	pCodeBook		= new CODEBOOK[CodeBookNum];
	m_pDspCodeBook	= new DSP_CODE_BOOK[CodeBookNum];

	

	pFeaToCodeBookProb	= new float[CodeBookNum];
	pCodeBookNo		= new long[TotalWordNum][HMM_STATE_NUM];

	for(WordNo=0;WordNo<TotalWordNum;WordNo++)
	{
		for(HmmStateNo=0;HmmStateNo<HMM_STATE_NUM;HmmStateNo++)
		{
			pCodeBookNo[WordNo][HmmStateNo] = WordCodeBook6HmmState[WordNo][HmmStateNo];
		}
	}

}

CCODE_BOOK::~CCODE_BOOK()
{
	delete []pCodeBook;
	delete []m_pDspCodeBook;
	
	delete []pFeaToCodeBookProb;
	delete []pCodeBookNo;
}

bool CCODE_BOOK::LoadWordCodeBookMap(char *FileName)
{
FILE	*fp=NULL;
long	WordNo,HmmStateNo;

	fopen_s(&fp,FileName,"rt");
	if(fp == NULL)
		return(false);

	for(WordNo=0;WordNo<TotalWordNum;WordNo++)
	{
		for(HmmStateNo=0;HmmStateNo<HMM_STATE_NUM;HmmStateNo++)
		{
			if(1 != fscanf_s(fp,"%ld ",&(pCodeBookNo[WordNo][HmmStateNo]) ) )
			{
				fclose(fp);
				return(false);
			}
		}
	}
	fclose(fp);
	return(true);
}

bool CCODE_BOOK::LoadCodeBook(char *FileName)
{
FILE	*fp=NULL;

	fopen_s(&fp,FileName,"rb");
	if(fp == NULL)
		return(false);
	fread(pCodeBook,sizeof(CODEBOOK),CodeBookNum,fp);
	fclose(fp);

	return(true);
}

bool CCODE_BOOK::LoadDspCodeBook(char *FileName)
{
FILE	*fp=NULL;

	fopen_s(&fp,FileName,"rb");
	if(fp == NULL)
		return(false);
	fread(m_pDspCodeBook,sizeof(DSP_CODE_BOOK),CodeBookNum,fp);
	fclose(fp);

	ConvertFromDspCodeBook();

	return(true);
}

void CCODE_BOOK::ConvertFromDspCodeBook(void)
{
	for( long CodeNo = 0 ; CodeNo < CODEBOOK_NUM; CodeNo++ )
	{
		for( long DimNo = 0 ; DimNo < FEATURE_DIM; DimNo++ )
		{
			pCodeBook[CodeNo].MeanU[DimNo] = (CODE_PRECISION)m_pDspCodeBook[CodeNo].MeanU[DimNo];
			for( long i = 0 ; i < FEATURE_DIM; i++ )
			{
				pCodeBook[CodeNo].InvR[DimNo][i] = (CODE_PRECISION)m_pDspCodeBook[CodeNo].InvR[DimNo][i];	
			}
		}

		pCodeBook[CodeNo].DetR			= (CODE_PRECISION)m_pDspCodeBook[CodeNo].DetR;
		pCodeBook[CodeNo].DurationMean	= (CODE_PRECISION)m_pDspCodeBook[CodeNo].DurationMean;
		pCodeBook[CodeNo].DurationVar	= (CODE_PRECISION)m_pDspCodeBook[CodeNo].DurationVar;

	}
}





void	CCODE_BOOK::FeaToAllCodeProb(FEATURE pFeature)
{
long	CodeBookNo;
	
	for(CodeBookNo=0;CodeBookNo<CodeBookNum;CodeBookNo++)
	{
		FeaToCodeBookProb(pFeature,CodeBookNo);	
	}
}

float	CCODE_BOOK::SimplifiedFeaToCodeBookProb(FEATURE Feature,long CodeBookNo)
{
long			i;
CODE_PRECISION	DotProduct,tmp;
CODE_PRECISION	*pMeanU;
	
	pMeanU = pCodeBook[CodeBookNo].MeanU;	//¾ùÖµÂë±¾
	DotProduct  = 0;
	for( i = 0; i < FEATURE_DIM; i++ )
	{
		tmp=Feature[i] - pMeanU[i];
		DotProduct += tmp * tmp;
	}
	pFeaToCodeBookProb[CodeBookNo] = float(-DotProduct);

	return(pFeaToCodeBookProb[CodeBookNo]);
}


















float CCODE_BOOK::GetDurationDist(long Duration,long CodeBookNo)
{
double	tmp,Var;
	
//	return(0);
	tmp=Duration-pCodeBook[CodeBookNo].DurationMean;
	Var=pCodeBook[CodeBookNo].DurationVar;
	if(Var<0.5)
		Var = 0.5;
	tmp=tmp*tmp/Var;
	tmp=-0.5*( tmp + Log2PI+log(Var) ); 
	return((float)tmp);
}

