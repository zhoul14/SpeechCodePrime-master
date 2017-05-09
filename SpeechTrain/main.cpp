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
#include "../StateProbabilityMap/StateProbabilityMap.h"
#include "assert.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include "omp.h"
#include "../CommonLib/Cluster.h"
//#include "vld.h"
using std::vector;
using std::string;

#define USEDURA 1
#define PRIME 0
#define HALF_FRAME_LEN 1
#define MULTIFRAME 0
#define PREPARE_PY 0
#define MAKE_NN_MODEL 0

#define FRAMECOLLECT 0
#define STATEPROBMAP 0
#define PRIME_DIM 45
#define WORD_LEVEL_MMIE 0
#define CLUSTER 1
void summaryUpdateRes(const vector<int>& r, FILE* fid, int iterNum) {
	if (r.size() == 0)
		return;

	vector<int> suc, sne, anz, ic;
	for (int i = 0; i < r.size(); i++) {
		if (r[i] == GMMEstimator::SUCCESS) {
			suc.push_back(i);
		} else if (r[i] == GMMEstimator::SAMPLE_NOT_ENOUGH) {
			sne.push_back(i);
		} else if (r[i] == GMMEstimator::ALPHA_NEAR_ZERO) {
			anz.push_back(i);
		} else if (r[i] == GMMEstimator::ILL_CONDITIONED) {
			ic.push_back(i);
		}
	}
	fprintf(fid, "Train Iter %d\n", iterNum);
	fprintf(fid, "SUCCESS: %d\n", suc.size());
	fprintf(fid, "ALPHA_NEAR_ZERO: %d\n", anz.size());
	for (int i = 0; i < anz.size(); i++) {
		fprintf(fid, "%d ", anz[i]);
	}
	fprintf(fid, "\n");

	fprintf(fid, "SAMPLE_NOT_ENOUGH: %d\n", sne.size());
	for (int i = 0; i < sne.size(); i++) {
		fprintf(fid, "%d ", sne[i]);
	}
	fprintf(fid, "\n");

	fprintf(fid, "ILL_CONDITION: %d\n", ic.size());
	for (int i = 0; i < ic.size(); i++) {
		fprintf(fid, "%d ", ic[i]);
	}
	fprintf(fid, "\n");


	fprintf(fid, "TOTAL: %d\n", r.size());

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


	string* configName,* configName2;
	if (argc == 1) {
		configName = new string("train_config.xml");
	} else {
		configName = new string(argv[1]);

#if PRIME
		configName2 = new string(argv[2]);//
#endif

		//configName3 = new string(argv[3]);
	}


	TrainParam tparam(configName->c_str());

#if PRIME
	TrainParam tparam2(configName2->c_str());
	//const int fDim2 = tparam2.getFdim();//
	const int fDim2 = PRIME_DIM;
#endif 

	bool triPhone = tparam.getTriPhone();
	//////////////////////////////////////////////////////////////////////////
	//for (int crossfitIter = 0; crossfitIter < 50 ;crossfitIter++){
	//////////////////////////////////////////////////////////////////////////	
	//初始化u中各成员
	SegmentUtility u;
	SegmentUtility uHelper;

	SimpleSTFactory* stfact = new SimpleSTFactory();
	SimpleSTFactory* stfactHelper = NULL;

	u.factory = stfact;
	uHelper.factory = stfactHelper;

	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);
	u.dict = dict;
	uHelper.dict = dict;
	//dict->setTriPhone(triPhone);
	dict->makeUsingCVidWordList();
	string initCb = tparam.getInitCodebook();
	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0);
	NBestRecAlgorithm* reca = new NBestRecAlgorithm();

	//set->saveCodebookDSP("preLarge_Model");
#if PRIME
	GMMCodebookSet* mySet = new GMMCodebookSet(3206,fDim2,1,1);//
#endif
	//mySet->mergeIsoCbs2BetaCbs(set);
	//set->printCodebookSetToFile("primeCBSet.txt");
	//mySet->printCodebookSetToFile("MixCBSet.txt");
	/*GMMCodebookSet* testSet= new GMMCodebookSet(857,48,1,3,0);
	testSet->printCodebookSetToFile("testCodebooks.txt");
	*/
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
	bool useSegmentModel = USEDURA ? tparam.getSegmentModelFlag(): USEDURA;
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
#if PREPARE_PY
	gbc->initPython();
#if !MAKE_NN_MODEL
	gbc->getNNModel(tparam.getInitCodebook());
#endif
	//初始化输入数据集
#endif
	vector<TSpeechFile> inputs = tparam.getTrainFiles();
#if PRIME
	vector<TSpeechFile> inputs2 = tparam2.getTrainFiles();//

#endif 
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
#if PRIME
	GMMUpdateManager ua(mySet, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel);//
	trainIter = 1;
#else
	GMMUpdateManager ua(set, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel, STATEPROBMAP, 25);

#endif

	/*ua.getMMIEmatrix("ConMatrix.txt",false);
	ua.getMMIEmatrix("DataMatrix.txt",true);*/
	//set->printCodebookSetToFile(string(logPath + "/all_codebooks.txt").c_str());

	bool* trainedCb = new bool[cbNum];

	double lhOfLastIter = -1e300; 

	for (int iter = 0; iter < trainIter ; iter++) {

		clock_t begTime = clock();
		clock_t labTime = 0;
		clock_t prepareTime = 0;
		double lhOfThisIter = 0;

		int trainCnt = -1;

#if STATEPROBMAP
		CStateProbMap CSPM(cbNum);
#endif

		/************************************************start segment*************************************************/
		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())
				break;
			const int fDim = tparam.getFdim();

			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);

			Cluster cluster((*i).getFeatureFileName(),input, tparam.getCltDirName());

#if PRIME
			int ii=trainCnt;
			FeatureFileSet input2(inputs2[ii].getFeatureFileName(),inputs2[ii].getMaskFileName(),inputs2[ii].getAnswerFileName(),fDim2);
#endif
			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {

				printf("process file %d, speech %d    \r", trainCnt, j);


#if HALF_FRAME_LEN && !PRIME

				int fNum = input.getFrameNumInSpeech_half_framelen(j);

				//if(fNum!=input2.getFrameNumInSpeech(j))//
				//printf("fNum1:%d does not match fNum2:%d",fNum,input2.getFrameNumInSpeech(j));

				int ansNum = input.getWordNumInSpeech(j);

				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}


				double* frames = new double[fNum * fDim];
				input.getSpeechAt_half_framelen(j, frames, fNum);
#else
				int fNum = input.getFrameNumInSpeech(j);

				//if(fNum!=input2.getFrameNumInSpeech(j))//
				//printf("fNum1:%d does not match fNum2:%d",fNum,input2.getFrameNumInSpeech(j));

				int ansNum = input.getWordNumInSpeech(j);

				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}


				double* frames = new double[fNum * fDim];
				input.getSpeechAt(j, frames);
#endif
#if MULTIFRAME

#if CLUSTER

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
#elif PREPARE_PY
				double* multiFrames = new double[fNum * fDim * (2 * MULTIFRAMES_COLLECT + 1)];
				input.getMultiSpeechAt(j, multiFrames);
#endif
#endif


#if PRIME
#if HALF_FRAME_LEN
				int fNum2 = input2.getFrameNumInSpeech_half_framelen(j);

				double* frames2 = new double[fDim2*fNum2];
				input2.getSpeechAt_half_framelen(j, frames2, fNum*2);
				std::cout << fNum << ":"<<fNum2;
				fNum2 = min(fNum*2,fNum2);
				/*if(fNum2 > 0){
					for(int i = 0; i != fNum2 * fDim2; i++){
						if (frames2[i] != frames2[i] ||  !_finite(frames2[i]))
							int debug = 1;

					}
				}*/
#else
				double* frames2 = new double[fDim2 * fNum];
				input2.getSpeechAt(j, frames2);//
#endif
#endif
				int* ansList = new int[ansNum];
				input.getWordListInSpeech(j, ansList);
				//分割前完成概率的预计算
				bool* mask = new bool[dict->getTotalCbNum()];
#if !STATEPROBMAP&&!WORD_LEVEL_MMIE
				dict->getUsedStateIdInAns(mask, ansList, ansNum);
				gbc->setMask(mask);
#endif
				clock_t t1 = clock();

#if PREPARE_PY && MULTIFRAME

				if(iter==0 && MAKE_NN_MODEL)
				{
					gbc->prepare(frames, fNum);
				}
				else
				{
					gbc->preparePy(multiFrames, fNum);		
				}
#else

#if 1/*
				double* tmpFrames = new double[fNum * fDim];
				memcpy(tmpFrames, frames, sizeof(double) * fNum * fDim);
				int cl = cluster.clusterFrame(tmpFrames,j,fDim,fNum);
				int clusterNum = cluster.getClusterNum(j);
				int* clustInfo = cluster.getClusterInfo(j,fNum);
				gbc->prepare(tmpFrames, clusterNum);*/
				//gbc->mergeClusterInfo(clustInfo,clusterNum);


#else
				//int clusterNum = cluster.getClusterNum(j);
				//int* clustInfo = cluster.getClusterInfo(j,fNum);

				gbc->prepare(frames, fNum);

				//gbc->mergeClusterInfo(clustInfo,clusterNum);

#endif



#endif

				clock_t t2 = clock();
				prepareTime += t2 - t1;
				//vector<vector<SWord> > res0 = reca->recSpeech(fNum, fDim, dict, gbc, 4, useSegmentModel);//isoword
				vector<SegmentResult>res0;
				if (WORD_LEVEL_MMIE)
				{
					res0.resize(TOTAL_WORD_NUM);
				}
				t1 = clock();
				int answer = ansList[0];
				int* usedFrames = NULL;
				/*usedFrames = new int[fNum * cbNum];
				memset(usedFrames, 1, sizeof(int) * fNum * cbNum);
				if (WORD_LEVEL_MMIE)
				{	
				double LhList[TOTAL_WORD_NUM]; 
				omp_set_dynamic(true);
				#pragma omp parallel for 
				for (int segIdx = 0; segIdx < TOTAL_WORD_NUM; segIdx++)
				{
				//ansList[0] = segIdx;
				int a[1];
				a[0]= segIdx;
				SegmentResult res1;
				SegmentAlgorithm sa;
				SimpleSTFactory mst;
				sa.setFactory(&mst);
				res1 = sa.segmentSpeech(fNum, fDim, ansNum, a, u);
				assert(mst.getAllocatedNum() == 0);
				if (segIdx == answer)
				{
				res = res1;
				}
				res0[segIdx] = (res1);
				LhList[segIdx] = res1.lh;
				}
				sort(LhList,LhList + TOTAL_WORD_NUM);

				for (int segIdx = 0; segIdx != TOTAL_WORD_NUM; segIdx++){
				if(res0[segIdx].lh < LhList[TOTAL_WORD_NUM - 25] && res0[segIdx].lh != res.lh)continue;
				int totalFrameNum = ua.collect(res0[segIdx].frameLabel, frames, usedFrames, res0[segIdx].lh);
				}
				//printf("res.lh:[%lf], minLh:[%lf]\n", res.lh , LhList[TOTAL_WORD_NUM - 25]);

				//cout<<shit<<endl;
				}

				//input.SaveSegmentPointToBuf(j,res.frameLabel);
				t2 = clock();
				labTime += t2 - t1;*/
				int totalFrameNum;
				SegmentResult res;

#if PRIME
				gbc->prepare(frames, fNum);
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
#else
				if(WORD_LEVEL_MMIE)
				{
					totalFrameNum = ua.collect(res.frameLabel, frames, FRAMECOLLECT);
					//ua.collectWordGamma(res0,res.frameLabel,usedFrames, res.lh);
				}
				else
				{
					//res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
#if  MULTIFRAME
					res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);

					totalFrameNum = ua.collectMultiFrames(res.frameLabel, multiFrames);

#else 
					if(clusterFlag){
						double* tmpFrames = new double[fDim * fNum];
						memcpy(tmpFrames, frames, sizeof(double) * fDim * fNum);
						fNum = cluster.clusterFrameSame(tmpFrames, j, fDim, fNum);
						gbc->prepare(tmpFrames, fNum);
						delete tmpFrames;
					}
					else{
						gbc->prepare(frames, fNum);
					}
					res = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
					totalFrameNum = ua.collect(res.frameLabel, frames);
#endif // MULTIFRAMES_COLLECT

				}
				//if(!ua.collectWordGamma(res.frameLabel,res0,ansList[0],res.lh))printf("shit !ans is not in recognition result List!\n\n");
				//ua.collectWordGamma(res0, ansList[0], usedFrames);
				delete []usedFrames;

#endif
#if STATEPROBMAP
				CSPM.pushToMap(gbc,res.frameLabel);
#endif

#if !WORD_LEVEL_MMIE
				assert(stfact->getAllocatedNum() == 0);
#endif
				lhOfThisIter += res.lh;

#if PRIME
				delete [] frames2;
#endif

#if MULTIFRAME
				delete []multiFrames;
#endif
				delete [] ansList;
				delete [] frames;
				delete [] mask;
			}
			//input.PrintSegmentPointBuf("SegMent48_log.txt");
#if WORD_LEVEL_MMIE
			printf("No.%dFile,",trainCnt);
			ua.printfObjectFunVal();
#endif
			printf("\n");
		}
		/*******************************segment end****************************************/

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "iter %d,\tlh = %e, segment time = %ds(%ds, %ds), TotalFrameNum = %d", iter, lhOfThisIter, segTime, labTime / CLOCKS_PER_SEC, prepareTime / CLOCKS_PER_SEC, ua.getFW()->getTotalFrameNum());
		fflush(lhRecordFile);

#if MULTIFRAME
		ua.getFW()->flush();
		cout<<"TotalFrameNum:"<<ua.getFW()->getTotalFrameNum()<<endl;
		//system("pause");
		//return 0;
#endif

#if STATEPROBMAP
		std::vector<std::vector<double>> m;
		m.resize(cbNum);
		CSPM.mergeMapToMatrix(m, ua.getFW());
		CSPM.saveOutMaptoFile(tparam.getInitCodebook() + "Map.txt", ua.getFW());

		int CorrelateCnt = ua.setMMIEmatrix(m);

		fprintf(lhRecordFile, ", Correlate num = %d", CorrelateCnt);
		fflush(lhRecordFile);
		/*return 0;*/
#endif
		vector<int> updateRes;
		if(STATEPROBMAP)
		{
			for (int uptime = 0; uptime != maxEMIter; uptime++)
			{
				ua.setGBC(gbc);
				updateRes = ua.updateStateLvMMIE();
				set->saveCodebook(tparam.getOutputCodebook());
				summaryUpdateRes(updateRes, summaryFile, iter);
			}
		}
		else if(WORD_LEVEL_MMIE)
		{
			fprintf(lhRecordFile, ", object value = %lf", ua.getObjFV());
			updateRes = ua.updateWordLvMMIE();
		}
		else if(PREPARE_PY)
		{
			if(iter == 0)ua.trainKerasNN(MAKE_NN_MODEL, tparam.getOutputCodebook(), tparam.getInitCodebook());
			else{
				ua.trainKerasNN(MAKE_NN_MODEL, tparam.getOutputCodebook(), tparam.getOutputCodebook());
			}
		}
		else
		{
			updateRes = ua.update();
		}
		clock_t endTime = clock();

		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);


		lhOfLastIter = lhOfThisIter;

		string allCbPath = logPath + "/all_codebooks.txt";
#if PRIME
		mySet->saveCodebook(tparam2.getOutputCodebook());//
		mySet->printCodebookSetToFile(allCbPath.c_str());//
		delete mySet;
#elif WORD_LEVEL_MMIE
		string ss = tparam.getOutputCodebook()+(char)(iter+48);
		cout<<ss<<endl;
		set->saveCodebook(ss);
		set->printCodebookSetToFile(allCbPath.c_str());
#else
		set->saveCodebook(tparam.getOutputCodebook());
		set->printCodebookSetToFile(allCbPath.c_str());
#endif
	}

	gbc->FinalizePython();
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
	return 0;
}