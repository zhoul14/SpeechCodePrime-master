#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
//#include "boost/filesystem.hpp"
//#include "CTriPhoneDDBRec.h"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include <direct.h>
#include "../CommonLib/Cluster.h"
#define REC_PY 1
#define CLUSTER 0
//#define DROPFRAME 0
using namespace std;
void printRecFile(FILE* fid, const vector<vector<SWord> >& res) {
	fprintf(fid, "%4d\n", res.size());
	for (int i = 0; i < res.size(); i++) {
		for (int j = 0; j < res[i].size(); j++) {
			SWord r = res[i][j];
			if (r.wordId == -1)
				continue;
			fprintf(fid, "%4d %.2f ", r.wordId, r.lh);
		}
		fprintf(fid, "\n");
	}
	fprintf(fid, "\n");
	fflush(fid);
}

void printRecResToScreen(const vector<vector<SWord> >& res, int* ansId, int ansNum, WordDict* dict) {
	if (res.size() == 0)
		return;

	for (int i = 0; i < res.size(); i++) {
		for (int j = 0; j < min(res[i].size(), 8); j++) {
			string tw = dict->wordToText(res[i][j].wordId);
			printf("%s\t", tw.c_str());
		}
		printf("\n");
	}

	printf("(");
	for (int i = 0; i < ansNum; i++) {
		string tw = dict->wordToText(ansId[i]);
		printf("%s ", tw.c_str());
	}
	printf(")\n");

}


int main(int argc,char *argv[]) {

	char *recg;
	if(argc < 2 || argc > 3) {
		printf("usage:program_name config_file [basedir]\n");
		exit(-1);
	} else {
		recg = argv[1];
		if (argc == 3) {
			//boost::filesystem::current_path(argv[2]);
			_chdir(argv[2]);
		}
	}

	RecParam rparam(recg);
	std::string CodeBookName = rparam.getCodebookFileName();
	int recFileNum = rparam.getRecNum();
	int multiJump = rparam.getMultiJump();

	const int fDim = rparam.getFdim();

	const int bestN = BEST_N;

	if (recFileNum < 1) {
		printf("Recognition File Number error! FileNum:%ld\n", recFileNum);
		return -1;
	}
	printf("FileNum=%ld\n", recFileNum);

	GMMCodebookSet* cbset = new GMMCodebookSet(rparam.getCodebookFileName().c_str());

	bool useCuda = rparam.getUseCudaFlag();
	bool useSegmentModel = false && rparam.getSegmentModelFlag();

	if (useCuda)
		printf("CUDA is used\n");
	else 
		printf("CUDA is not used\n");

	if (useSegmentModel)
		printf("SegmentModel is used\n");
	else
		printf("SegmentModel is not used\n");

	GMMProbBatchCalc* gbc = new GMMProbBatchCalc(cbset, useCuda, useSegmentModel);
#if REC_PY
	gbc->initPython();
	gbc->getNNModel(rparam.getCodebookFileName());
#endif
	WordDict* dict = new WordDict(rparam.getWordDictFileName().c_str(),rparam.getTriPhone());

	dict->setTriPhone(rparam.getTriPhone());

	NBestRecAlgorithm* reca = new NBestRecAlgorithm();

	gbc->setDurWeight(rparam.getDurWeight());
	printf("CodeBook:%s  DurWeight=%f\n",CodeBookName.c_str(),rparam.getDurWeight());

	vector<RSpeechFile> inputs = rparam.getRecFiles();
	double totalLh = 0;

	for (int i = 0; i < recFileNum; i++) {
		RSpeechFile input = inputs.at(i);
		FeatureFileSet fs(input.getFeatureFileName(), input.getMaskFileName(), input.getAnswerFileName(), rparam.getFdim());


#if	CLUSTER
		Cluster cluster(input.getFeatureFileName(),fs);
#endif

		string pdir = input.getRecResultFileName();
		pdir = pdir.substr(0, pdir.find_last_of("/\\"));
		if (GetFileAttributes(pdir.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(pdir.c_str(), NULL);
		}

		FILE* recf = fopen(input.getRecResultFileName().c_str(), "w");
		if (!recf) {
			printf("Cannot open file: %s\n", input.getRecResultFileName().c_str());
			exit(-1);
		}

		//读入语音特征文件
		int SentenceNum = fs.getSpeechNum();
		fprintf(recf, "%4d\n", SentenceNum);

#ifdef DROPFRAME
		char tagFileName[4];
		tagFileName[3] = 0;
		sprintf(tagFileName,"M%.2d",i + 1);
		FILE* df = fopen(string(string()).c_str(),"rb");
		int n = fs.fileByteNum(df);
		char* efBuf = new char[n];
		fread(&n, 1, n, df);
		fclose(df);
#endif		


		for (int j = 0; j < SentenceNum ; j++) {

			printf("recognizing file %d: %d/%d\n", i, j, SentenceNum);
			int wordNum = fs.getWordNumInSpeech(j);
			int* pWordList = new int[wordNum];
			fs.getWordListInSpeech(j, pWordList);

			int fNum = fs.getFrameNumInSpeech(j);


#ifdef DROPFRAME
			ClusterIndex* clustidx = (ClusterIndex*)(efBuf + sizeof(ClusterIndex) * j);
			float* energyBuf = (float*)(clustidx->offset + efBuf);
			int frameLen = clustidx->ClusterNum;
			if(frameLen != fNum){
				printf("frameLen:%d != fNum:%d\n",frameLen,fNum);
			}
			int newfNum= 0;
			for (int t = 0; t != fNum; t++)
			{

				if(energyBuf[t] > 10000){
					newfNum ++;
				}
			}
#endif
#if REC_PY
			double* featuresMulti = new double[fNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];
			fs.getMultiSpeechAt(j, featuresMulti);
#if DROPFRAME
			double* temp = new double[newfNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];
			int counter = 0;
			int outCnt = 0;
			for (int t = 0; t !=fNum; t++)
			{

				if(energyBuf[t] > 10000 && counter%2 == 0){
					counter++;
					continue;
				}
				memcpy(temp + (outCnt++) * fDim *(MULTIFRAMES_COLLECT * 2 + 1), featuresMulti + t * fDim *(MULTIFRAMES_COLLECT * 2 + 1), fDim *(MULTIFRAMES_COLLECT * 2 + 1));
			}
			fNum = newfNum;
			memcpy(featuresMulti, temp, newfNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1));
			delete[] temp;
#endif
			gbc->preparePy(featuresMulti, fNum);
#else
			double* features = new double[fNum * fDim];
			fs.getSpeechAt(j, features);
#if DROPFRAME
			double* temp = new double[newfNum * fDim];
			int counter = 0;
			int outCnt = 0;
			for (int t = 0; t !=fNum; t++)
			{

				if(energyBuf[t] > 10000 && counter%2 == 0){
					counter++;
					continue;
				}
				memcpy(temp + (outCnt++) * fDim, features + t * fDim , fDim);
			}
			fNum = newfNum;
			memcpy(features, temp, newfNum * fDim);
			delete[] temp;
#endif
#if CLUSTER
			fNum = cluster.clusterFrame(features,j,fDim,fNum);
			/*int clusterNum = cluster.getClusterNum(j);
			int* clustInfo = cluster.getClusterInfo(j,fNum);
			gbc->mergeClusterInfo(clustInfo,clusterNum);
			fNum = clusterNum;*/
#endif
			gbc->prepare(features, fNum);
#endif


			vector<vector<SWord> > res = reca->recSpeech(fNum, fDim, dict, gbc, multiJump, useSegmentModel);
			printRecResToScreen(res, pWordList, wordNum, dict);

			int resLen = res.size();
			if (resLen > 0 && res[resLen - 1].size() > 0) {
				totalLh += res[resLen - 1][0].lh;
			}

			delete [] pWordList;

#if REC_PY
			delete [] featuresMulti;
#else
			delete [] features;
#endif
			printRecFile(recf, res);
		}
		fclose(recf);

#ifdef DROPFRAME
		delete []efBuf;
#endif
	}
	FILE* lhFile = fopen("lhrecord.txt", "w");
	fprintf(lhFile, "lh = %e\n", totalLh);
	fclose(lhFile);
#if REC_PY
	//gbc->FinalizePython();
#endif
	delete cbset;
	delete dict;
	delete reca;
	delete gbc;


	return 0;
}

