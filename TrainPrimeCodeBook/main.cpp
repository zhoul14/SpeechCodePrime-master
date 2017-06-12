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
using std::vector;
using std::string;

#define PRIME 1//首次
#define CODEBOOK_NUM 3206//新码本多少个状态
#define HALF_FRAME_LEN 0//是否半帧长


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


	string *configName = nullptr, *configName2 = nullptr;
	if (argc == 3) {
		configName = new string(argv[1]);
		configName2 = new string(argv[2]);
	} else {
		printf("prime codebook train need two train_config files, first is seed config file, second is prime codebook config file.");
	}


	TrainParam tparam(configName->c_str());
	TrainParam tparam2(configName2->c_str());
	const int fDim2 = tparam2.getFdim();

	bool triPhone = tparam.getTriPhone();

	SegmentUtility u;
	SimpleSTFactory* stfact = new SimpleSTFactory();
	u.factory = stfact;

	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);
	u.dict = dict;
	dict->makeUsingCVidWordList();
	string initCb = tparam.getInitCodebook();


	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0);

	GMMCodebookSet* mySet = new GMMCodebookSet(CODEBOOK_NUM, fDim2, 1/*mixnum*/,GMMCodebookSet::CB_TYPE_FULL_RANK/*type*/);//

	GCSTrimmer::fixSmallDurSigma(set, tparam.getMinDurSigma());

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
	u.bc = gbc;//

	vector<TSpeechFile> inputs = tparam.getTrainFiles();
	vector<TSpeechFile> inputs2 = tparam2.getTrainFiles();//

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

	GMMUpdateManager ua(mySet, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel);//
	trainIter = 1;

	bool* trainedCb = new bool[cbNum];

	double lhOfLastIter = -1e300; 

	for (int iter = 0; iter < trainIter ; iter++) {

		clock_t begTime = clock();
		clock_t labTime = 0;
		clock_t prepareTime = 0;
		double lhOfThisIter = 0;

		int trainCnt = -1;

		/************************************************start segment*************************************************/
		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())
				break;
			const int fDim = tparam.getFdim();

			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);

			Cluster cluster((*i).getFeatureFileName(), tparam.getCltDirName());

			int ii=trainCnt;
			FeatureFileSet input2(inputs2[ii].getFeatureFileName(),inputs2[ii].getMaskFileName(),inputs2[ii].getAnswerFileName(),fDim2);
			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {

				printf("process file %d, speech %d    \r", trainCnt, j);


				int fNum = input.getFrameNumInSpeech(j);
				int ansNum = input.getWordNumInSpeech(j);
				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}
				double* frames = new double[fNum * fDim];
				input.getSpeechAt(j, frames);

#if HALF_FRAME_LEN
				int fNum2 = input2.getFrameNumInSpeech_half_framelen(j);

				double* frames2 = new double[fDim2*fNum2];
				input2.getSpeechAt_half_framelen(j, frames2, fNum*2);
				std::cout << fNum << ":"<<fNum2;
				fNum2 = min(fNum*2,fNum2);
#else
				double* frames2 = new double[fDim2 * fNum];
				input2.getSpeechAt(j, frames2);//
#endif
				int* ansList = new int[ansNum];
				input.getWordListInSpeech(j, ansList);
				bool* mask = new bool[dict->getTotalCbNum()];
				dict->getUsedStateIdInAns(mask, ansList, ansNum);
				gbc->setMask(mask);
				clock_t t1 = clock();
				//分割前完成概率的预计算
				gbc->prepare(frames, fNum);
				clock_t t2 = clock();
				prepareTime += t2 - t1;
				t1 = clock();
				int answer = ansList[0];
				int* usedFrames = NULL;
				int totalFrameNum;
				SegmentResult res;

				res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
#if HALF_FRAME_LEN
				vector<int> half_framelen_label;
				for(auto i = 0; i != res.frameLabel.size(); i++ ){
					half_framelen_label.push_back(res.frameLabel[i]);
					half_framelen_label.push_back(res.frameLabel[i]);
				}
				if (fNum2 !=half_framelen_label.size()){
					cout << fNum2 << " and " << half_framelen_label.size();
					if(half_framelen_label.size() > fNum2){
						half_framelen_label.resize(fNum2);
					}
				}
				totalFrameNum = ua.collect(half_framelen_label, frames2, true, nullptr);//
#else
				totalFrameNum = ua.collect(res.frameLabel, frames2, true, nullptr);//

#endif

				assert(stfact->getAllocatedNum() == 0);
				lhOfThisIter += res.lh;

				delete [] frames2;
				delete [] ansList;
				delete [] frames;
				delete [] mask;
			}
			printf("\n");
		}
		/*******************************segment end****************************************/

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "iter %d,\tlh = %e, segment time = %ds(%ds, %ds), TotalFrameNum = %d", iter, lhOfThisIter, segTime, labTime / CLOCKS_PER_SEC, prepareTime / CLOCKS_PER_SEC, ua.getFW()->getTotalFrameNum());
		fflush(lhRecordFile);

		vector<int> updateRes;
		updateRes = ua.update();
		ua.summaryUpdateRes(updateRes, summaryFile, iter);

		clock_t endTime = clock();
		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);
		
		lhOfLastIter = lhOfThisIter;
		string allCbPath = logPath + "/all_codebooks.txt";
		mySet->saveCodebook(tparam2.getOutputCodebook());//
		mySet->printCodebookSetToFile(allCbPath.c_str());//
		delete mySet;
	}

	for (int i = 0; i < ua.getUaCbnum(); i++) {
		fprintf(updateTimeFile, "%d\t%d\n", i, ua.getSuccessUpdateTime(i));
	}

	fclose(lhRecordFile);
	fclose(updateTimeFile);
	fclose(summaryFile);
	delete [] trainedCb;	
	delete set;
	delete stfact;
	delete dict;
	delete gbc;	
	delete configName;
	delete configName2;
	return 0;
}
