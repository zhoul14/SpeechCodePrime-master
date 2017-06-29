#include "GMMUpdateManager.h"
#include "../CommonLib/Math/MathUtility.h"
#include <string>
#include <iostream>
#include <fstream>
#include "algorithm"
#include "sstream"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#include <Python.h>
#include <arrayobject.h>
#include "omp.h"
//#include "vld.h"
#define  MAX_CNUM 7
#define KAI 0.0453
//GMMEstimator(int fDim, int mixNum, int maxIter);

GMMUpdateManager::GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool cudaFlag, bool useSegmentModel) {
	this->codebooks = codebooks;
	this->dict = dict;
	this->minDurVar = minDurSigma * minDurSigma;
	this->useSegmentModel = useSegmentModel;
	this->updateIter = 0;
	this->m_bCuda = cudaFlag;

	//reiniter = new KMeansReinit(10, 100);
	logFile = fopen(logFileName, "w");
	if (!logFile) {
		printf("cannot open log file[%s] in update manager\n", logFileName);
		exit(-1);
	}
	estimator = new GMMEstimator(codebooks->getFDim(), codebooks->getMixNum(), codebooks->getCbType(), maxIter, cudaFlag, 0);
	estimator->setOutputFile(logFile);

	std::string fwpath = "FWTMP";
	int maxFNum = 30000000;
	fw = new FrameWarehouse(fwpath, codebooks->getCodebookNum(), codebooks->getFDim(), maxFNum);

	int cbNum = codebooks->getCodebookNum();

	if (useSegmentModel) {
		firstMoment = new double[cbNum];
		memset(firstMoment, 0, cbNum * sizeof(double));

		firstMomentOfLog = new double[cbNum];
		memset(firstMomentOfLog, 0, cbNum * sizeof(double));

		secondMoment = new double[cbNum];
		memset(secondMoment, 0, cbNum * sizeof(double));

		durCnt = new int[cbNum];
		memset(durCnt, 0, cbNum * sizeof(int));
	}


	successUpdateTime = new int[cbNum];
	memset(successUpdateTime, 0, cbNum * sizeof(int));

}

int GMMUpdateManager::collect(const std::vector<int>& frameLabel, double* frames, bool bCollectFrame, int* clusterInfo) {
	if (frameLabel.size() == 0) {
		return 0;
	}

	int fDim = codebooks->getFDim();
	int cbNum = codebooks->getCodebookNum();

	//统计duration的一阶矩和二阶矩
	if (useSegmentModel) {
		int currentCb = *(frameLabel.begin());

		if(currentCb>=cbNum)
			currentCb=cbNum-1;

		int currentDur = 0;
		for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
			int cb = *i;

			if(cb>=cbNum)
				cb=cbNum-1;

			if (cb == currentCb) {
				currentDur++;
			} else {
				firstMoment[currentCb] += currentDur;
				firstMomentOfLog[currentCb] += log((double)currentDur);
				secondMoment[currentCb] += currentDur * currentDur;
				durCnt[currentCb]++;

				currentCb = cb;
				currentDur = 1;
			}
		}
		//处理最后一段duration
		firstMoment[currentCb] += currentDur;
		firstMomentOfLog[currentCb] += log((double)currentDur);
		secondMoment[currentCb] += currentDur * currentDur;
		durCnt[currentCb]++;
	}


	//统计各码本对应的帧
	int time = -1;
	if(bCollectFrame){
		if(clusterInfo == nullptr){
			for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
				time++;

				double* ft = frames + fDim * time;
				int cbid = (*i);
				if(cbid>=cbNum)
					cbid=cbNum-1;
				fw->pushFrame(cbid, ft);
			}
		}
		else{
			int idx = 0;
			time++;

			for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {

				while (time<clusterInfo[idx])
				{
					double* ft = frames + fDim * time;
					int cbid = (*i);
					if(cbid>=cbNum)
						cbid=cbNum-1;
					fw->pushFrame(cbid, ft);
					time++;

				}
				idx++;
			}
		}
	}
	int totalFrameNum = fw->getTotalFrameNum();
	return max(totalFrameNum,time);
}

int GMMUpdateManager::collectMultiFrames(const std::vector<int>& frameLabel, double* multiframes){
	if (frameLabel.size() == 0) {
		return 0;
	}

	int fDim = codebooks->getFDim();
	int cbNum = codebooks->getCodebookNum();

	//统计duration的一阶矩和二阶矩
	if (useSegmentModel) {
		int currentCb = *(frameLabel.begin());

		if(currentCb>=cbNum)
			currentCb=cbNum-1;

		int currentDur = 0;
		for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
			int cb = *i;

			if(cb>=cbNum)
				cb=cbNum-1;

			if (cb == currentCb) {
				currentDur++;
			} else {
				firstMoment[currentCb] += currentDur;
				firstMomentOfLog[currentCb] += log((double)currentDur);
				secondMoment[currentCb] += currentDur * currentDur;
				durCnt[currentCb]++;

				currentCb = cb;
				currentDur = 1;
			}
		}
		//处理最后一段duration
		firstMoment[currentCb] += currentDur;
		firstMomentOfLog[currentCb] += log((double)currentDur);
		secondMoment[currentCb] += currentDur * currentDur;
		durCnt[currentCb]++;
	}


	//统计各码本对应的帧
	int time = -1;
	for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
		time++;

		double* ft = multiframes + fDim * time * (2 * MULTIFRAMES_COLLECT + 1);
		int cbid = (*i);
		if(cbid>=cbNum)
			cbid=cbNum-1;
		for(int j = 0; j != (2 * MULTIFRAMES_COLLECT + 1); j++){
			fw->pushFrame(cbid, ft + fDim * j);
		}
	}
	int totalFrameNum = fw->getTotalFrameNum();
	return totalFrameNum;
}

GMMUpdateManager::~GMMUpdateManager() {

	fclose(logFile);
	delete estimator;
	fw->clearFrames();
	delete fw;
	if (useSegmentModel) {
		delete [] firstMoment;
		delete [] firstMomentOfLog;
		delete [] secondMoment;
		delete [] durCnt;
	}

	delete [] successUpdateTime;
}

int GMMUpdateManager::getSuccessUpdateTime(int cbidx) const {
	return successUpdateTime[cbidx];
}

std::vector<int> GMMUpdateManager::update() {
	using std::vector;
	int cbNum = codebooks->getCodebookNum();
	int fDim = codebooks->getFDim();

	int* cbFrameCnt = new int[cbNum];

	for (int i = 0; i < cbNum; i++) {
		cbFrameCnt[i] = fw->getFrameNum(i);
	}

	vector<int> res;
	for (int i = 0; i < cbNum; i++) {
		printf("updating codebook %d, %d samples\n", i, cbFrameCnt[i]);
		fprintf(logFile, "updating codebook %d, %d samples\n", i, cbFrameCnt[i]);

		if (cbFrameCnt[i] == 0) {
			int errCode = GMMEstimator::SAMPLE_NOT_ENOUGH;
			std::string t = GMMEstimator::errorInfo(errCode);
			printf("--- UPDATE FAIL: %s ---\n", t.c_str());
			fprintf(logFile, "--- UPDATE FAIL: %s ---\n", t.c_str());
			res.push_back(errCode);
			continue;
		}
		GMMCodebook cb = codebooks->getCodebook(i);
		int errCode;
		if (useSegmentModel) {
			//高斯分布建模duration
			cb.DurMean = firstMoment[i] / durCnt[i];
			cb.DurVar = secondMoment[i] / durCnt[i] - cb.DurMean * cb.DurMean;

			if (cb.DurVar < minDurVar) {
				cb.DurVar = minDurVar;
			}
		}


		double* allFramesOfCb = new double[cbFrameCnt[i] * fDim];
		fw->loadFrames(i, allFramesOfCb);

		estimator->setCbId(i);
		estimator->loadParam(cb.Alpha, cb.Mu, cb.InvSigma);
		estimator->loadData(allFramesOfCb, cbFrameCnt[i]);
		errCode = ((GMMEstimator*)estimator)->estimate();
		res.push_back(errCode);

		delete [] allFramesOfCb;

		if (errCode != GMMEstimator::SUCCESS) {
			successUpdateTime[i] = 0;
			std::string t = GMMEstimator::errorInfo(errCode);
			printf("--- UPDATE FAIL: %s ---\n", t.c_str());
			fprintf(logFile, "--- UPDATE FAIL: %s ---\n", t.c_str());
			continue;
		}

		estimator->saveParam(cb.Alpha, cb.Mu, cb.InvSigma);
		codebooks->updateCodebook(i, cb);
		successUpdateTime[i]++;

	}
	fw->clearFrames();
	delete [] cbFrameCnt;
	return res;
}

void GMMUpdateManager::trainKerasNN(bool bMakeNewModel, std::string ot = "",std::string it = ""){
	printf("trainKerasNN\n");
	static int flag = 0;
	int codebookNum = codebooks->CodebookNum;
	int ll = (MULTIFRAMES_COLLECT * 2 + 1);
	int pyFdim = codebooks->getFDim() * ll;
	static int time = 1;
	cout<<"TotalFrameNum"<<fw->getTotalFrameNum()/ll<<endl;
	if (flag == 0 && bMakeNewModel)
	{

		flag++;
		PyObject * pModule = NULL;    //声明变量  
		PyObject * pFunc = NULL;      //声明变量  
		pModule =PyImport_ImportModule(PYTHONFILE); 
		pFunc= PyObject_GetAttrString(pModule, "makeModel");   

		PyObject *pArgs = PyTuple_New(6);        
		//函数调用的参数
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("s",(ot + MODEL_JSON_FILE).c_str()));
		PyTuple_SetItem(pArgs, 1, Py_BuildValue("s",(ot + MODEL_H5_FILE).c_str()));
		PyTuple_SetItem(pArgs, 2, Py_BuildValue("s","FWTMP"));
		PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", codebookNum));
		PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", fw->getTotalFrameNum()/ll));
		PyTuple_SetItem(pArgs, 5, Py_BuildValue("i", pyFdim));

		PyObject *pReturn = PyEval_CallObject(pFunc, pArgs);

		if (!pReturn)
		{
			printf("python error");
			PyObject* excType, *excValue, *excTraceback;
			PyErr_Fetch(&excType, &excValue, &excTraceback);
			PyErr_NormalizeException(&excType, &excValue, &excTraceback);

			PyTracebackObject* traceback = (PyTracebackObject*)excTraceback;
			while (traceback->tb_next != NULL){
				std::cout<<excValue->ob_type->tp_name<<endl;
				traceback = traceback->tb_next;
			}
			exit(-1);
		}
		fw->clearFrames();
		//gbc->FinalizePython();
		//gbc->initPython();
		//gbc->getNNModel(it);
		return;
	}

	PyObject * pModule = NULL;    //声明变量  
	PyObject * pFunc = NULL;      //声明变量  
	pModule =PyImport_ImportModule(PYTHONFILE); 
	pFunc= PyObject_GetAttrString(pModule, "trainModel");   

	PyObject *pArgs = PyTuple_New(9);        
	//函数调用的参数
	time++;
	float lr = time > 2 ? 0.001 : 0.001/time ;

	std::cout<<fw->getTotalFrameNum()<<endl;
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("s",(it+MODEL_JSON_FILE).c_str()));
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("s",(it+MODEL_H5_FILE).c_str()));
	PyTuple_SetItem(pArgs, 2, Py_BuildValue("s",(ot+MODEL_JSON_FILE).c_str()));
	PyTuple_SetItem(pArgs, 3, Py_BuildValue("s",(ot+MODEL_H5_FILE).c_str()));
	PyTuple_SetItem(pArgs, 4, Py_BuildValue("s","FWTMP"));
	PyTuple_SetItem(pArgs, 5, Py_BuildValue("i", codebookNum));
	PyTuple_SetItem(pArgs, 6, Py_BuildValue("i", fw->getTotalFrameNum()/ll));
	PyTuple_SetItem(pArgs, 7, Py_BuildValue("f", lr));
	PyTuple_SetItem(pArgs, 8, Py_BuildValue("i", pyFdim));

	PyObject *pReturn = PyEval_CallObject(pFunc, pArgs);
	if (!pReturn)
	{
		printf("python error ! function: trainModel!\n");
		PyObject* excType, *excValue, *excTraceback;
		PyErr_Fetch(&excType, &excValue, &excTraceback);
		PyErr_NormalizeException(&excType, &excValue, &excTraceback);

		PyTracebackObject* traceback = (PyTracebackObject*)excTraceback;
		while (traceback->tb_next != NULL)
			traceback = traceback->tb_next;
		exit(-1);
	}

	fw->clearFrames();




	//gbc->FinalizePython();
	//gbc->initPython();
	//gbc->getNNModel(it);

}

void GMMUpdateManager::summaryUpdateRes(const vector<int>& r, FILE* fid, int iterNum) {
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