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
int calSpeechLh(const int& j, FeatureFileSet& fs, GMMProbBatchCalc* gbc,const int& fNum, const int& fDim, Cluster* cluster){

	double* features = new double[fNum * fDim];
	fs.getSpeechAt(j, features);
	int temp_fNum = fNum;
	if(cluster != nullptr)temp_fNum = cluster->clusterFrame(features, j, fDim, fNum);
	gbc->prepare(features, temp_fNum);
	delete [] features;
	return temp_fNum;
}

void calPySpeechLh(const int& j, FeatureFileSet& fs, GMMProbBatchCalc* gbc, const int& fNum, const int& fDim){

	double* featuresMulti = new double[fNum * fDim * (MULTIFRAMES_COLLECT * 2 + 1)];
	fs.getMultiSpeechAt(j, featuresMulti);
	gbc->preparePy(featuresMulti, fNum);
	delete [] featuresMulti;
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
	bool usePy = rparam.getUsePYFlag();
	bool useCluster = rparam.getClusterFlag();
	string cltDirname = rparam.getCltDirName();
	if (useCuda)
		printf("CUDA is used\n");
	else 
		printf("CUDA is not used\n");

	if (useSegmentModel)
		printf("SegmentModel is used\n");
	else
		printf("SegmentModel is not used\n");

	GMMProbBatchCalc* gbc = new GMMProbBatchCalc(cbset, useCuda, useSegmentModel);
	if(usePy){
		gbc->initPython();
		gbc->getNNModel(rparam.getNNmodelName());
	}
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

		Cluster* cluster = nullptr;
		if(useCluster > 0){
			cltDirname = cltDirname == "" ? rparam.getCltDirName() : cltDirname;
			cluster = new Cluster (input.getFeatureFileName(),cltDirname);
			cout<< "CLT DIR NAME:"<<cltDirname<<endl;
		}
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

		//�������������ļ�
		int SentenceNum = fs.getSpeechNum();
		fprintf(recf, "%4d\n", SentenceNum);

		for (int j = 0; j < SentenceNum ; j++) {

			printf("recognizing file %d: %d/%d\n", i, j, SentenceNum);
			int wordNum = fs.getWordNumInSpeech(j);
			int* pWordList = new int[wordNum];
			fs.getWordListInSpeech(j, pWordList);

			int fNum = fs.getFrameNumInSpeech(j);

			if(usePy){
				calPySpeechLh(j, fs, gbc, fNum, fDim);
			}
			else{
				fNum = calSpeechLh(j, fs, gbc, fNum, fDim, cluster);
			}

			vector<vector<SWord> > res = reca->recSpeech(fNum, fDim, dict, gbc, multiJump, useSegmentModel);
			printRecResToScreen(res, pWordList, wordNum, dict);

			int resLen = res.size();
			if (resLen > 0 && res[resLen - 1].size() > 0) {
				totalLh += res[resLen - 1][0].lh;
			}

			delete [] pWordList;
			printRecFile(recf, res);
		}
		if(cluster != nullptr)delete cluster;
		fclose(recf);
	}
	FILE* lhFile = fopen("lhrecord.txt", "w");
	fprintf(lhFile, "lh = %e\n", totalLh);
	fclose(lhFile);

	delete cbset;
	delete dict;
	delete reca;
	delete gbc;


	return 0;
}

