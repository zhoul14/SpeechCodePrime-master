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
#include "omp.h"
#include "../CommonLib/Cluster.h"
using std::vector;
using std::string;

//是否用GMM作为种子码本，生成NN模型
#define MAKE_NN_MODEL 0

double* getClusterMultiFrames(const int& j, const int& fDim, int& fNum, Cluster& cluster, FeatureFileSet& input){
	int primeFLen = input.getPrimeFrameLen(j);
	int maskLen = input.getMaskLen(j);

	double* features = new double[primeFLen * fDim];
	int *maskInfo = new int[maskLen * 2];

	input.getMaskData(maskInfo, j);
	input.getPrimeSpeechAt(j, features);
	double* multiFrames = new double[fNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];

	fNum = cluster.clusterMultiFrame(features, j, fDim, primeFLen, maskInfo, maskLen, multiFrames, 1);

	delete [] maskInfo;
	delete [] features;

	return multiFrames;
}

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
	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);//字典位置
	u.dict = dict;
	string initCb = tparam.getInitCodebook();//初始码本名字
	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0);//初始化种子码本
	GCSTrimmer::fixSmallDurSigma(set, tparam.getMinDurSigma());//设置最小段长方差
	int cbNum = set->getCodebookNum();

	bool useCuda = tparam.getUseCudaFlag();//是否gpu
	bool useSegmentModel =  false;//是否段长
	string NNname = tparam.getInitNNname();//初始dnn模型的名字
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
	GMMProbBatchCalc* gbc =	new GMMProbBatchCalc(set, useCuda, useSegmentModel);//初始化计算类
	printf("cbs fdim = %d, cbs statenum = %d, cbs mixture = %d\n",set->getFDim(),set->getCodebookNum(),set->getMixNum());
	u.bc = gbc;//把gbc给更新器

	gbc->initPython();//初始化python模块
	if(MAKE_NN_MODEL == false)gbc->getNNModel(tparam.getInitNNname());//如果不是第一次生产dnn，则给gbc种子nn模型

	//初始化输入数据集，各种Log文件
	vector<TSpeechFile> inputs = tparam.getTrainFiles();
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

	SegmentAlgorithm sa;
	GMMUpdateManager ua(set, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel);
	for (int iter = 0; iter < trainIter ; iter++) {

		int trainCnt = -1;
		clock_t begTime = clock();
		clock_t prepareTime = 0;

		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())break;

			const int fDim = tparam.getFdim();//feature dim
			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);//给文件夹集
			Cluster cluster((*i).getFeatureFileName(), tparam.getCltDirName());//初始化cluster

			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {

				printf("process file %d, speech %d    \r", trainCnt, j);

				int fNum = input.getFrameNumInSpeech(j);//帧数
				int ansNum = input.getWordNumInSpeech(j);//答案数
				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}

				int* ansList = new int[ansNum];//答案list
				input.getWordListInSpeech(j, ansList);//获取答案list
				bool* mask = new bool[dict->getTotalCbNum()];
				dict->getUsedStateIdInAns(mask, ansList, ansNum);//获取mask
				if(MAKE_NN_MODEL == true)gbc->setMask(mask);

				double* frames = nullptr, *multiFrames = nullptr;
				int totalFrameNum = 0;

				if(tparam.getCltDirName() != "")
				{
					multiFrames = getClusterMultiFrames(j, fDim, fNum, cluster,input);//如果cluster，处理cluster多帧
				}
				else{
					multiFrames = new double[fNum * fDim * (2 * MULTIFRAMES_COLLECT + 1)];
					input.getMultiSpeechAt(j, multiFrames);//取前后多帧特征矢量。
				}
				clock_t t1;
				//分割前完成概率的预计算
				if(MAKE_NN_MODEL && iter == 0){
					frames = new double[fNum * fDim];
					input.getSpeechAt(j, frames);
					t1 = clock();
					gbc->prepare(frames, fNum);		
					delete []frames;
				}
				else{
					t1 = clock();
					gbc->preparePy(multiFrames, fNum);		//用python计算概率
				}
				prepareTime += clock() - t1;

				//分割force-alignment
				SegmentResult res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
				//分割后，保存HMM状态对应的帧特征向量，等待最后一起更新
				totalFrameNum = ua.collectMultiFrames(res.frameLabel, multiFrames);

				delete [] multiFrames;
				delete [] ansList;
				delete [] mask;
			}
			printf("\n");
		}
		/*******************************segment end****************************************/

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fflush(lhRecordFile);

		//将保存的HMM状态对应帧特征向量，flush到文件中
		ua.getFW()->flush();
		printf("flushing!\n");

		midTime = clock();
		//用文件中HMM状态对应的帧特征向量，更新码本模型
		ua.trainKerasNN(MAKE_NN_MODEL, tparam.getOutNNname(), tparam.getInitNNname());//用keras更新
		clock_t endTime = clock();

		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "prepare time = %ds", prepareTime);
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);
	}


	fclose(lhRecordFile);
	fclose(updateTimeFile);
	fclose(summaryFile);
	delete set;
	delete stfact;
	delete dict;
	delete gbc;	
	delete configName;
	gbc->FinalizePython();

	return 0;
}