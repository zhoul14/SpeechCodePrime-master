#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../OneBestRecAlgorithm/SimpleSpeechRec.h"
#include <windows.h>
#include "../SpeechSegmentAlgorithm/SimpleSTFactory.h"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
using namespace std;
void printSeg2RecFile(FILE* fid, const int* res, const vector<double>& resLh,const int& len) {
	fprintf(fid, "%4d\n", len);
	for (int i = 0; i < len; i++) {
		int r = res[i];
		if (r == -1)
			continue;
		fprintf(fid, "%4d %.2f \n", r, 0.5);
	}
	fprintf(fid, "\n");
	fflush(fid);
}

void printSegResToScreen(int* res, int resNum,int* ansId, int ansNum, WordDict* dict) {
	if (resNum == 0)
		return;

	for (int i = 0; i < resNum; i++) {
		string tw = dict->wordToText(res[i]);
		printf("%s  ", tw.c_str());
	}
	printf("\n");

	int s = ansNum;

	printf("[T]");
	for (int i = 0; i < ansNum; i++) {
		string tw = dict->wordToText(ansId[i]);
		printf("%s ", tw.c_str());
	}
	printf("\n\n");

}
void printSegResToFile(FILE* f, int* res, int resNum,int* ansId, int ansNum, WordDict* dict) {
	if (resNum == 0)
		return;

	for (int i = 0; i < resNum; i++) {
		string tw = dict->wordToText(res[i]);
		fprintf(f,"%s  ", tw.c_str());
	}
	fprintf(f,"\n");

	int s = ansNum;

	fprintf(f,"[T]");
	for (int i = 0; i < ansNum; i++) {
		string tw = dict->wordToText(ansId[i]);
		fprintf(f,"%s ", tw.c_str());
	}
	fprintf(f,"\n\n");

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
	int multiJump = rparam.getMultiJump();
	SegmentAlgorithm sa;
	SegmentUtility u;
	SimpleSTFactory* stfact = new SimpleSTFactory();

	u.factory = stfact;

	const int bestN = BEST_N;

	if (recFileNum < 1) {
		printf("Recognition File Number error! FileNum:%ld\n", recFileNum);
		return -1;
	}
	printf("FileNum=%ld\n", recFileNum);

	GMMCodebookSet* cbset = new GMMCodebookSet(rparam.getCodebookFileName().c_str());

	bool useSegmentModel = rparam.getSegmentModelFlag();
	bool useCuda = rparam.getUseCudaFlag();
	int useClusterFlag = rparam.getClusterFlag();

	GMMProbBatchCalc* gbc = new GMMProbBatchCalc(cbset, useCuda, useSegmentModel);

	WordDict* dict = new WordDict(rparam.getWordDictFileName().c_str(),rparam.getTriPhone());
	u.dict = dict;
	u.bc = gbc;//

	const int fDim = cbset->FDim;

	SimpleSpeechRec* reca = new SimpleSpeechRec();

	gbc->setDurWeight(rparam.getDurWeight());
	printf("CodeBook:%s  DurWeight=%f\n",CodeBookName.c_str(),rparam.getDurWeight());

	vector<RSpeechFile> inputs = rparam.getRecFiles();
	FILE* f = fopen("CmdWord_log.txt","w");

	double totalLh = 0;
	int sumErrorCnt = 0;
	int sumRecCnt = 0;
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
		int ** pWordMap = new int*[SentenceNum];
		int errorCnt = 0;
		int recCnt = 0;

		fprintf(recf, "%4d\n", SentenceNum);
		for (int j = 0; j < SentenceNum ; j++) {
			int wordNum = fs.getWordNumInSpeech(j);
			int* pWordList = new int[wordNum];
			fs.getWordListInSpeech(j, pWordList);
			pWordMap[j] = pWordList;
		}



		for (int j = 0; j < SentenceNum ; j++) {

			printf("recognizing file %d: %d/%d\r", i, j, SentenceNum);
			int fNum = fs.getFrameNumInSpeech(j);
			double* features = new double[fNum * fDim];

			fs.getSpeechAt(j, features);
			gbc->prepare(features, fNum);
			SegmentResult res;
			double maxLh = -INT_MAX;
			int maxIdx = -1;
			for (int k = 0; k < SentenceNum; k++)
			{

				int* pWordList = pWordMap[k];
				int wordNum = fs.getWordNumInSpeech(k);
				SegmentResult tres = sa.segmentSpeech(fNum, fDim, wordNum, pWordList, u);

				if (maxLh < tres.lh && tres.frameLabel.empty() == false)
				{
					res = tres;
					maxLh = tres.lh;
					maxIdx = k;
				}
			}

			
			int wordNum = fs.getWordNumInSpeech(maxIdx);
			int* pWordList = pWordMap[maxIdx];
			//printSegResToScreen(res.frameLabel, res.stateLh, pWordList, wordNum, dict);

			if(maxIdx != j){
				errorCnt ++;
				int ansNum = fs.getWordNumInSpeech(j);
				int* ansList = pWordMap[j];
				printSegResToScreen(pWordList, wordNum, ansList, ansNum, dict);
				fprintf(f, "Error idx:%d", j);
				printSegResToFile(f,pWordList, wordNum, ansList, ansNum, dict);
			}

			int resLen = res.frameLabel.size();
			totalLh += res.lh;

			delete [] features;
			printSeg2RecFile(recf, pWordList, res.stateLh, wordNum);
			recCnt++;
		}

		for (int j = 0; j < SentenceNum ; j++)
		{
			delete []pWordMap[j];
		}
		delete []pWordMap;
		fclose(recf);
		fprintf(f,"ErrorCnt:[%d], TotalCnt:[%d], ErrorRate:[%f]\n", errorCnt,recCnt,(float)errorCnt/recCnt);
		printf("ErrorCnt:[%d], TotalCnt:[%d], ErrorRate:[%f]\n", errorCnt,recCnt,(float)errorCnt/recCnt);
		sumErrorCnt += errorCnt;
		sumRecCnt += recCnt;
	}
	fprintf(f,"=============================================================");
	fprintf(f,"{Sum}:\nErrorCnt:[%d], TotalCnt:[%d], ErrorRate:[%f]\n", sumErrorCnt,sumRecCnt,(float)sumErrorCnt/sumRecCnt);

	fclose(f);
	delete stfact;
	delete cbset;
	delete dict;
	delete reca;
	delete gbc;
	return 0;
}

