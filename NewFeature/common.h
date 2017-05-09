#ifndef	_INCLUDE_XIAOXI_SPEECH_COMMON_H_
#define	_INCLUDE_XIAOXI_SPEECH_COMMON_H_

#define	TOTAL_WORD_NUM	1254
#define	CODEBOOK_NUM	856		//388
#define	HMM_STATE_NUM	6		//6
#define	FEATURE_DIM		45		//14*3+1*3 = 45

typedef float FEATURE_ELEMENT;
typedef float FEATURE[FEATURE_DIM];

typedef double	CODE_PRECISION;
typedef struct tagCodeBook{
	CODE_PRECISION	MeanU[FEATURE_DIM];
	CODE_PRECISION	InvR[FEATURE_DIM][FEATURE_DIM];
	CODE_PRECISION	DetR;
	CODE_PRECISION	DurationMean;
	CODE_PRECISION	DurationVar;
} CODEBOOK;


typedef struct tagDspCodeBook{
	float	MeanU[FEATURE_DIM];
	float	InvR[FEATURE_DIM][FEATURE_DIM];
	float	DetR;
	float	DurationMean;
	float	DurationVar;
} DSP_CODE_BOOK;



typedef struct tagFEA_INDEX{
	long	Offset;
	long	ByteSize;
} FEA_INDEX;

#endif



