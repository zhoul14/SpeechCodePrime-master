#include "GMMCodebookSet.h"
#include "../CommonLib/CommonVars.h"
typedef  double CODE_PRECISION;
typedef struct tagCodeBook{
	CODE_PRECISION	MeanU[FEATURE_DIM];
	CODE_PRECISION	InvR[FEATURE_DIM][FEATURE_DIM];
	CODE_PRECISION	DetR;
	CODE_PRECISION	DurationMean;
	CODE_PRECISION	DurationVar;
} CODEBOOK;

int cbsToxxcb(std::string filename, void* cbs);