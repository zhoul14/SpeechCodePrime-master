#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
//#include "boost/filesystem.hpp"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../OneBestRecAlgorithm/SimpleSpeechRec.h"
//#include "SRecTokenFactory.h"
#include <windows.h>
#include "../CommonLib/Droper.h"
#include "../CommonLib/Cluster.h"
#include <time.h>

using namespace std;
void printRecFile(FILE* fid, const vector<SWord>& res) {
	fprintf(fid, "%4d\n", res.size());
	for (int i = 0; i < res.size(); i++) {
		SWord r = res[i];
		if (r.wordId == -1)
			continue;
		fprintf(fid, "%4d %.2f \n", r.wordId, r.lh);
	}
	fprintf(fid, "\n");
	fflush(fid);
}

void printRecResToScreen(const vector<SWord>& res, int* ansId, int ansNum, WordDict* dict) {
	if (res.size() == 0)
		return;

	for (int i = 0; i < res.size(); i++) {
		string tw = dict->wordToText(res[i].wordId);
		printf("%s[%.2f] ", tw.c_str(), res[i].lh);
	}
	printf("\n");

	int s = res.size();
	printf("LH = %f\n", res[s - 1].lh);

	printf("[T]");
	for (int i = 0; i < ansNum; i++) {
		string tw = dict->wordToText(ansId[i]);
		printf("%s ", tw.c_str());
	}
	printf("\n\n");

}
void calSpeechLh(FeatureFileSet &fs, Droper* droper, Cluster* cluster, const int& j, int fNum, const int& useClusterFlag, const int& fDim, GMMProbBatchCalc* gbc){

	double* features = new double[fNum * fDim];
	fs.getSpeechAt(j, features);
	if(droper != nullptr)fNum = droper->dropFrame(features, j, fDim, fNum);
	if(cluster != nullptr)fNum = cluster->clusterFrame(features, j, fDim, fNum, useClusterFlag);//如果聚类
	gbc->prepare(features, fNum);
	delete [] features;
}

void calPYSpeechLh(FeatureFileSet &fs, Cluster* cluster, const int& j, const int& useClusterFlag, const int& fDim, GMMProbBatchCalc* gbc){

	int fNum = fs.getFrameNumInSpeech(j);
	double* featuresMulti = nullptr;
	if(cluster == nullptr){
		featuresMulti = new double[fNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];
		fs.getMultiSpeechAt(j, featuresMulti);
		gbc->preparePy(featuresMulti, fNum);
	}
	else{
		int primeFLen = fs.getPrimeFrameLen(j);
		double* features = new double[primeFLen * fDim];
		int maskLen = fs.getMaskLen(j);
		int *maskInfo = new int[maskLen * 2];
		fs.getMaskData(maskInfo, j);
		fs.getPrimeSpeechAt(j, features);
		featuresMulti = new double[fNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];
		fNum = cluster->clusterMultiFrame(features, j, fDim, primeFLen, maskInfo, maskLen, featuresMulti, useClusterFlag);
		gbc->preparePy(featuresMulti, fNum);
		delete [] maskInfo;
		delete [] features;

	}

	delete [] featuresMulti;
}

void calHalfFrameLenSpeechLh(FeatureFileSet &fs, Droper* droper, Cluster* cluster, const int& j, const int& useClusterFlag, const int& fDim, GMMProbBatchCalc* gbc){

	int fNum = fs.getFrameNumInSpeech_half_framelen(j);
	double* features = new double[fNum * fDim];
	fs.getSpeechAt_half_framelen(j, features, fNum);
	if(cluster!= nullptr)fNum = cluster->clusterFrame(features, j, fDim, fNum, useClusterFlag);//如果聚类
	gbc->prepare(features, fNum);
	delete []features;
}


int main(int argc,char *argv[]) {

	char *recg;
	string cltDirname="";
	if(argc < 2 || argc > 3) {
		printf("usage:program_name config_file [basedir]\n");
		exit(-1);
	} else {
		recg = argv[1];
		if (argc == 3) {
			cltDirname = string(argv[2]);
			//etCurrentDirectory(argv[2]);
		}
	}

	RecParam rparam(recg);
	std::string CodeBookName = rparam.getCodebookFileName();
	int recFileNum = rparam.getRecNum();

	const int bestN = BEST_N;

	if (recFileNum < 1) {
		printf("Recognition File Number error! FileNum:%ld\n", recFileNum);
		return -1;
	}
	printf("FileNum=%ld\n", recFileNum);

	GMMCodebookSet* cbset = new GMMCodebookSet(rparam.getCodebookFileName().c_str());

	bool useSegmentModel = rparam.getSegmentModelFlag();
	bool useCuda = rparam.getUseCudaFlag();
	bool usePy = rparam.getUsePYFlag();
	bool useDropFrame = rparam.getUseDropFlag();
	bool useHalfLenStep = rparam.getUseHalfLenFlag();
	int useClusterFlag = rparam.getClusterFlag();

	GMMProbBatchCalc* gbc = new GMMProbBatchCalc(cbset, useCuda, useSegmentModel);

	if(usePy){
		gbc->initPython();
		gbc->getNNModel(rparam.getNNmodelName());
	}
	WordDict* dict = new WordDict(rparam.getWordDictFileName().c_str(),rparam.getTriPhone());

	const int fDim = cbset->FDim;

	SimpleSpeechRec* reca = new SimpleSpeechRec();

	gbc->setDurWeight(rparam.getDurWeight());
	printf("CodeBook:%s  DurWeight=%f\n",CodeBookName.c_str(),rparam.getDurWeight());

	vector<RSpeechFile> inputs = rparam.getRecFiles();
	double totalLh = 0;
	double time1 = clock();
	for (int i = 0; i < recFileNum; i++) {

		RSpeechFile input = inputs.at(i);
		FeatureFileSet fs(input.getFeatureFileName(), input.getMaskFileName(), input.getAnswerFileName(), cbset->FDim);

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
		Droper* droper = nullptr;
		if(useDropFrame){
			droper = new Droper(input.getFeatureFileName(),fs,400000);
		}
		Cluster* cluster = nullptr;
		if(useClusterFlag > 0){
			cltDirname = cltDirname == "" ? rparam.getCltDirName() : cltDirname;
			cluster = new Cluster (input.getFeatureFileName(),cltDirname);
			cout<< "CLT DIR NAME:"<<cltDirname<<endl;
		}

		for (int j = 0; j < SentenceNum ; j++) {

			printf("recognizing file %d: %d/%d\n", i, j, SentenceNum);
			int wordNum = fs.getWordNumInSpeech(j);
			int* pWordList = new int[wordNum];
			fs.getWordListInSpeech(j, pWordList);
			int fNum = fs.getFrameNumInSpeech(j);

			if(droper != nullptr)droper->getNewFnum(j,fNum);
			
			if(useHalfLenStep){
				calHalfFrameLenSpeechLh(fs, droper, cluster, j, useClusterFlag, fDim, gbc);
			}
			else if (usePy){
				calPYSpeechLh(fs, cluster, j, useClusterFlag, fDim, gbc);
			}
			else{
				calSpeechLh(fs, droper, cluster, j, fNum, useClusterFlag, fDim, gbc);
			}
			vector<SWord> res = reca->recSpeech(fNum, fDim, dict, gbc, useSegmentModel,rparam.getBHeadNoise());
			printRecResToScreen(res, pWordList, wordNum, dict);

			int resLen = res.size();
			totalLh += res[resLen - 1].lh;

			delete [] pWordList;
			printRecFile(recf, res);
		}
		if(cluster != nullptr)delete cluster;
		if(droper != nullptr) delete droper;
		fclose(recf);

	}
	time1 -= clock();
	printf("time:%lf",-time1/1000);
	FILE* lhFile = fopen("timer.txt", "a+");
	fprintf(lhFile, "file:%s, timer = %lf\n", cltDirname.c_str(), -time1/1000);
	fclose(lhFile);

	delete cbset;
	delete dict;
	delete reca;
	delete gbc;
	return 0;
}

