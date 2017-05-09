#include "common.h"

#ifndef	_INCLUDE_XIAOXI_CODEBOOK_COMMON_H_
#define	_INCLUDE_XIAOXI_CODEBOOK_COMMON_H_



class CCODE_BOOK{
	private:
		double		Log2PI;
		CODE_PRECISION	x[FEATURE_DIM],y[FEATURE_DIM];
	public:
		long		TotalWordNum;
		long		CodeBookNum;

		CODEBOOK		*pCodeBook;
		DSP_CODE_BOOK	*m_pDspCodeBook;

		float		*pFeaToCodeBookProb;
		long		(*pCodeBookNo)[HMM_STATE_NUM];
		CCODE_BOOK(long	SetCodeBookNum=CODEBOOK_NUM,long SetTotalWordNum=TOTAL_WORD_NUM);
		~CCODE_BOOK();
		bool	LoadCodeBook(char *FileName);
		bool	LoadDspCodeBook(char *FileName);

		void	ConvertFromDspCodeBook(void);

		bool	LoadWordCodeBookMap(char *FileName);

		float	FeaToCodeBookProb(FEATURE pFeature,long CodeBookNo);
		float	SimplifiedFeaToCodeBookProb(FEATURE pFeature,long CodeBookNo);

		void	FeaToAllCodeProb(FEATURE pFeature);
		long	GetCodeBookNum(void)
		{
			return (CodeBookNum);
		};
		float	GetDurationDist(long Duration,long CodeBookNo);
};

#endif