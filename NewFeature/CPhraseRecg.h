#ifndef	_INCLUDE_XIAOXI_PHRASE_RECG_FILE_H_
#define	_INCLUDE_XIAOXI_PHRASE_RECG_FILE_H_

#include "windows.h"
#include "common.h"
#include "CCodeBook.h"
#include "MFCC.h"
#include "SpeechFea.h"

#define	MIN_PROBABILITY				(-1E+10)

#define	MAX_PHRASE_NUM				300
#define	MAX_WORD_NUM_EACH_PHRASE	20
#define MAX_CANDIDATE_NUM			10


class CPHRASE_RECG
{
private:
	CCODE_BOOK			*pCCodeBook;
	CMFCC				*pCMfcc;

	int		PhraseNum;
	int		CodeBookNum;
	float	PhrasePathProb[MAX_PHRASE_NUM][MAX_WORD_NUM_EACH_PHRASE*HMM_STATE_NUM];
	int		StateCodeBookNo[MAX_PHRASE_NUM][MAX_WORD_NUM_EACH_PHRASE*HMM_STATE_NUM];
	int		StateNum[MAX_PHRASE_NUM];
	int		CurrentFrameNo;
	FEATURE *TelephoneMFCC(float *SpeechBuf,int SpeechSampleNum,int *FeaFrameNum);
	BOOL	getAdapMfcc(float *SpeecBuffer,float *FeatureBuffer,int FrameNum);

public:
	char	CmdText[400][200];
	CPHRASE_RECG( CCODE_BOOK *pSetCCodeBook=0, char *PhraseFileName = NULL );
	~CPHRASE_RECG();

	float	BestProbList[MAX_CANDIDATE_NUM];
	int	*SpeechRecg(float *SpeechBuffer,int SpeechLen);
	void	FirstPassRecg(FEATURE Feature);
	void	SecondPassRecg(FEATURE *pFeature,int *CandBuffer,int CandNum);

	void	ResetFirstPassRecg(void);
	void	ResetSecondPassRecg(int *CandBuffer,int CandNum);

	int		*GetFirstPassCand(int CandNum);
	int		*GetSecondPassCand(int *CandBuffer,int CandNum);
};

#endif