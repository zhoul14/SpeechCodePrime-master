#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#include "../CommonLib/ReadConfig/TrainParam.h"
#include "../CommonLib/Dict/WordDict.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../SpeechSegmentAlgorithm/SimpleSTFactory.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMUpdateManager.h"
#include <windows.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <cassert>
#include <vector>
#include <string>
#include <cuda.h>
#include "assert.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include "omp.h"
#include "../CommonLib/Cluster.h"
//#include "vld.h"
using std::vector;
using std::string;

//是否5ms帧长，0表示不是
#define HALF_FRAME_LEN 0

int main(int argc, char** argv) {

	if (argc > 3) {
		printf("usage: program_name config_file [basedir]");
		exit(-1);
	}
	if (argc == 3) {
		if (GetFileAttributes(argv[2]) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(argv[2], NULL);
		}
		SetCurrentDirectory(argv[2]);
		//current_path(argv[2]);
	}

	string* configName;
	if (argc == 1) {
		configName = new string("train_config.xml");
	} else {
		configName = new string(argv[1]);
	}
	TrainParam tparam(configName->c_str());
	bool triPhone = tparam.getTriPhone();
	//初始化u中各成员
	SegmentUtility u;
	SimpleSTFactory* stfact = new SimpleSTFactory();
	u.factory = stfact;

	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);
	u.dict = dict;

	string initCb = tparam.getInitCodebook();

	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0);
	GCSTrimmer::fixSmallDurSigma(set, tparam.getMinDurSigma());

	int splitAddN = tparam.getSplitAddN();
	double splitOffset = tparam.getSplitOffset();
	if (splitAddN > 0) {
		printf("splitting codebooks, add %d mixtures\n", splitAddN);
		if (splitAddN > set->MixNum) {
			printf("splitAddN[%d] param cannot be larger than MixNum[%d]\n", splitAddN, set->MixNum);
			exit(-1);
		}
		set->splitAddN(splitAddN, splitOffset);
	}

	int cbNum = set->getCodebookNum();
	double durWeight = tparam.getDurWeight();
	bool useCuda = tparam.getUseCudaFlag();
	bool useSegmentModel = tparam.getSegmentModelFlag();
	if (useCuda) {
		printf("UseCUDA is true\n");
	} else {
		printf("UseCUDA is false\n");
	}

	if (useSegmentModel) {
		printf("UseSegmentModel is true\n");
	} else {
		printf("UseSegmentModel is false\n");
	}
	GMMProbBatchCalc* gbc =	new GMMProbBatchCalc(set, useCuda, useSegmentModel);
	printf("cbs fdim = %d, cbs statenum = %d, cbs mixture = %d\n",set->getFDim(),set->getCodebookNum(),set->getMixNum());
	gbc->setDurWeight(durWeight);
	bool clusterFlag = tparam.getCltDirName() != "";
	u.bc = gbc;//
	vector<TSpeechFile> inputs = tparam.getTrainFiles();
	SegmentAlgorithm sa;

	int maxEMIter = tparam.getEMIterNum();

	int trainIter = tparam.getTrainIterNum();

	string logPath = tparam.getLogPath();
	string lhRecordPath = logPath + "/lh_record.txt";
	string updateTimePath = logPath + "/update_time.txt";
	string updateIterPath = logPath + "/update_iter.txt";
	string summaryPath = logPath + "/summary.txt";

	FILE* lhRecordFile = fopen(lhRecordPath.c_str(), "w");
	if (!lhRecordFile) {
		printf("cannot open log file[%s]\n", lhRecordPath.c_str());
		exit(-1);
	}

	FILE* updateTimeFile = fopen(updateTimePath.c_str(), "w");
	if (!updateTimeFile) {
		printf("cannot open log file[%s]\n", updateTimePath.c_str());
		exit(-1);
	}

	FILE* summaryFile = fopen(summaryPath.c_str(), "w");
	if (!summaryFile) {
		printf("cannot open log file[%s]\n", summaryPath.c_str());
		exit(-1);
	}
	GMMUpdateManager ua(set, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel);

	for (int iter = 0; iter < trainIter ; iter++) {

		clock_t begTime = clock();
		double lhOfThisIter = 0;
		int trainCnt = -1;
		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())
				break;
			const int fDim = tparam.getFdim();
			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);
			Cluster cluster((*i).getFeatureFileName(), tparam.getCltDirName());
			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {
				printf("process file %d, speech %d    \r", trainCnt, j);
				int ansNum = input.getWordNumInSpeech(j);

#if HALF_FRAME_LEN 
				int fNum = input.getFrameNumInSpeech_half_framelen(j);
				double* frames = new double[fNum * fDim];
				input.getSpeechAt_half_framelen(j, frames, fNum);
#else
				int fNum = input.getFrameNumInSpeech(j);
				double* frames = new double[fNum * fDim];
				input.getSpeechAt(j, frames);
#endif
				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}
				int* ansList = new int[ansNum];
				input.getWordListInSpeech(j, ansList);
				//分割前完成概率的预计算
				bool* mask = new bool[dict->getTotalCbNum()];
				dict->getUsedStateIdInAns(mask, ansList, ansNum);
				gbc->setMask(mask);
				if(clusterFlag){
					double* tmpFrames = new double[fDim * fNum];
					memcpy(tmpFrames, frames, sizeof(double) * fDim * fNum);
					fNum = cluster.clusterFrame(tmpFrames, j, fDim, fNum);
					gbc->prepare(tmpFrames, fNum);
					delete tmpFrames;
				}
				else{
					gbc->prepare(frames, fNum);
				}
				SegmentResult res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
				int totalFrameNum = ua.collect(res.frameLabel, frames);
				assert(stfact->getAllocatedNum() == 0);
				lhOfThisIter += res.lh;
				delete [] ansList;
				delete [] frames;
				delete [] mask;
			}
			printf("\n");
		}

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "iter %d,\tlh = %e, segment time = %ds, TotalFrameNum = %d",
			iter, lhOfThisIter, segTime, ua.getFW()->getTotalFrameNum());
		fflush(lhRecordFile);
		vector<int> updateRes = ua.update();
		ua.summaryUpdateRes(updateRes, summaryFile, iter);
		clock_t endTime = clock();

		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);

		string allCbPath = logPath + "/all_codebooks.txt";
		set->saveCodebook(tparam.getOutputCodebook());
		set->printCodebookSetToFile(allCbPath.c_str());
	}

	for (int i = 0; i < ua.getUaCbnum(); i++) {
		fprintf(updateTimeFile, "%d\t%d\n", i, ua.getSuccessUpdateTime(i));
	}

	fclose(lhRecordFile);
	fclose(updateTimeFile);
	fclose(summaryFile);
	delete set;
	delete stfact;
	delete dict;
	delete gbc;	
	delete configName;
	return 0;
}