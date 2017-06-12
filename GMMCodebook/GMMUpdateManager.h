#ifndef WJQ_GMM_UPDATE_MANAGER_H
#define WJQ_GMM_UPDATE_MANAGER_H

#include "GMMEstimator.h"
#include "../CommonLib/Dict/WordDict.h"
#include "GMMCodebookSet.h"
#include "FrameWarehouse.h"
#include "string"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"

class SegmentResult;
class GMMUpdateManager {
private:

	GMMCodebookSet* codebooks;

	GMMEstimator* estimator;

	double* firstMoment;

	double* secondMoment;

	double* firstMomentOfLog;

	int* successUpdateTime;

	int updateIter;

	int* durCnt;

	bool useSegmentModel;

	bool m_bCuda;

	WordDict* dict;

	double minDurVar;

	FrameWarehouse* fw;

	GMMProbBatchCalc* gbc;

	FILE* logFile;

public:

	GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool useCuda, bool useSegmentModel);

	//返回值表示执行完本次函数后累积的全部码本的总帧数
	int collect(const std::vector<int>& frameLabel, double* frames, bool bCollectFrame = true,int* clusterInfo = nullptr);

	//获取码本数
	int getUaCbnum(){return codebooks->CodebookNum;}

	//返回值为长度为cbnum的vector，vector的每个元素代表相应的码本的更新结果
	std::vector<int> update();

	//获取FrameWareHouse指针
	FrameWarehouse* getFW(){return fw;}

	//获取状态idx的成功更新次数
	int getSuccessUpdateTime(int cbidx) const;

	//从外部设置GBC
	void setGBC(GMMProbBatchCalc* pGbc){
		gbc = pGbc;
	}
	//积累前后多帧的全部码本总帧数
	int collectMultiFrames(const std::vector<int>& frameLabel, double* multiframes);

	//调用Keras训练DNN
	void trainKerasNN(bool bMakeNewModel, std::string ot, std::string it);

	~GMMUpdateManager();

	//总结训练结果
	void summaryUpdateRes(const vector<int>& r, FILE* fid, int iterNum);
};

#endif